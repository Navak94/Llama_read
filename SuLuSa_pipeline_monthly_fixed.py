from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# Global config
# =============================================================================
EARTH_RADIUS_KM = 6371.0088
RECHECK_LABELS = {"unclear", "neutral"}
ARTICLE_MIN_LEN = 300
ARTICLE_CHAR_LIMIT = 9000

QWEN_7B = "qwen2.5:7b"
QWEN_72B = "qwen2.5:72b"

BASE_FEATURES = [
    "gpi",
    "hdi",
    "life_expectancy",
    "expected_schooling",
    "gdp_per_capita",
    "distance",
    "push_count",
    "pull_count",
    "mixed_count",
    "unclear_count",
    "n_articles",
] #mean_schooling to be re-added (was after expected_schooling)

FEATURE_KEYWORDS = {
    "gpi": [
        "war", "conflict", "violence", "attack", "invasion", "troop", "military",
        "shelling", "missile", "bomb", "crisis", "security", "peace", "instability",
        "displacement", "refugee", "combat", "front line", "frontline", "occupation",
    ],
    "hdi": [
        "development", "human development", "living standard", "quality of life",
        "poverty", "healthcare", "education", "wellbeing", "welfare", "inequality",
    ],
    "life_expectancy": [
        "life expectancy", "mortality", "death rate", "public health", "healthcare",
        "hospital", "disease", "epidemic", "pandemic",
    ],
    "expected_schooling": [
        "school", "schooling", "education", "student", "university", "teacher",
        "classroom", "learning",
    ],
    "gdp_per_capita": [
        "gdp", "economy", "economic", "jobs", "job", "employment", "unemployment",
        "income", "salary", "wage", "inflation", "recession", "investment", "growth",
        "market", "trade", "business", "poverty",
    ],
    "distance": [
        "border", "cross-border", "cross border", "neighboring", "neighbouring",
        "route", "journey", "travel", "crossing", "frontier", "corridor",
    ],
}

# =============================================================================
# Generic helpers
# =============================================================================
def normalize_country(s: str) -> str:
    return str(s).strip().lower()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def run_ollama(
    prompt: str,
    model: str,
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.2,
    timeout_sec: int = 180,
    max_retries: int = 2,
    format_json: bool = False,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if format_json:
        payload["format"] = "json"

    url = host.rstrip("/") + "/api/generate"
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout_sec)
            r.raise_for_status()
            response_text = (r.json().get("response") or "").strip()
            if response_text:
                return response_text
            return "[model_error] empty_output"
        except requests.Timeout:
            if attempt == max_retries:
                return "[model_error] timeout"
            time.sleep(2**attempt)
        except Exception as e:
            if attempt == max_retries:
                return f"[model_error] {e}"
            time.sleep(2**attempt)
    return "[model_error] unknown"


def extract_json_object(text: str) -> dict:
    text = (text or "").strip()

    if not text:
        raise ValueError("LLM returned empty output.")
    if text.startswith("[model_error]"):
        raise ValueError(f"LLM error: {text}")

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not find a JSON object in LLM response. Raw output starts: {text[:300]!r}")


def sanitize_article_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    rename_candidates = {
        "sourceurl": "URL",
        "url": "URL",
        "link": "URL",
        "published_month": "month",
        "year_month": "month",
    }
    for old_name, new_name in rename_candidates.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    if "URL" not in df.columns:
        raise ValueError("Article CSV must contain one of: URL, sourceurl, url, link")

    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
    elif "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
    elif "year" in df.columns:
        y = df["year"].astype(str).str.strip()
        if y.str.fullmatch(r"\d{4}-\d{2}").all():
            df["month"] = y
        elif y.str.fullmatch(r"\d{4}").all():
            df["month"] = y + "-01"
        else:
            df["month"] = pd.to_datetime(y, errors="coerce").dt.to_period("M").astype(str)
    else:
        raise ValueError("Article CSV must contain month, date, or year information.")

    for col in ["corridor", "year", "title", "article_text", "snippet"]:
        if col not in df.columns:
            df[col] = ""

    if "year" not in df.columns:
        df["year"] = df["month"]

    return df


def normalize_panel_month_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
    elif "year" in df.columns:
        y = df["year"].astype(str).str.strip()
        if y.str.fullmatch(r"\d{4}-\d{2}").all():
            df["month"] = y
        elif y.str.fullmatch(r"\d{4}").all():
            df["month"] = y + "-01"
        else:
            df["month"] = pd.to_datetime(y, errors="coerce").dt.to_period("M").astype(str)
    elif "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
    elif "period" in df.columns:
        df["month"] = pd.to_datetime(df["period"], errors="coerce").dt.to_period("M").astype(str)
    elif "year_month" in df.columns:
        df["month"] = pd.to_datetime(df["year_month"], errors="coerce").dt.to_period("M").astype(str)
    else:
        raise ValueError("Panel must contain one of: month, year, date, period, year_month")
    return df


def extract_url_text_tokens(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    parts = [parsed.netloc or "", parsed.path or "", parsed.query or ""]
    raw = " ".join(parts)
    raw = unquote(raw).lower()
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return " ".join(raw.split())


def count_feature_mentions_in_text(text: str) -> Dict[str, int]:
    text = str(text or "").lower()
    counts: Dict[str, int] = {}
    for feature, keywords in FEATURE_KEYWORDS.items():
        total = 0
        for kw in keywords:
            total += len(re.findall(rf"\b{re.escape(kw.lower())}\b", text))
        counts[feature] = total
    return counts


def add_feature_mention_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mention_cols = [f"{feat}_mentions" for feat in FEATURE_KEYWORDS]
    for col in mention_cols:
        if col not in df.columns:
            df[col] = 0

    for idx, row in df.iterrows():
        url_text = extract_url_text_tokens(row.get("URL", ""))
        title_text = str(row.get("title", "") or "")
        article_text = str(row.get("article_text", "") or "")
        snippet_text = str(row.get("snippet", "") or "")
        combined_text = " ".join([url_text, title_text, snippet_text, article_text]).strip()
        counts = count_feature_mentions_in_text(combined_text)
        for feature, value in counts.items():
            df.at[idx, f"{feature}_mentions"] = int(value)

    return df


def monthly_mentions_to_penalties(monthly_feature_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_feature_df.copy()
    mention_cols = [f"{feat}_mentions" for feat in FEATURE_KEYWORDS]
    for feature in FEATURE_KEYWORDS:
        mention_col = f"{feature}_mentions"
        penalty_col = f"w_{feature}"
        if mention_col in df.columns:
            df[penalty_col] = 1.0 / (1.0 + df[mention_col].fillna(0).astype(float))
    return df


# =============================================================================
# Mode-3 style article reading and classification
# =============================================================================
def get_article_text_newspaper(url: str, char_limit: int = ARTICLE_CHAR_LIMIT) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = (article.text or "").strip()
        return text[:char_limit]
    except Exception:
        return ""


def get_article_text_fallback(url: str, char_limit: int = ARTICLE_CHAR_LIMIT) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        return text[:char_limit]
    except Exception:
        return ""


def get_article_text(url: str, char_limit: int = ARTICLE_CHAR_LIMIT) -> str:
    text = get_article_text_newspaper(url, char_limit=char_limit)
    if len(text) >= ARTICLE_MIN_LEN:
        return text
    return get_article_text_fallback(url, char_limit=char_limit)

def build_base_feature_descriptions() -> Dict[str, str]:
    return {
        "gpi": "Global Peace Index. Higher conflict or instability can drive migration pressure.",
        "hdi": "Human Development Index. Broad development and quality-of-life measure.",
        "life_expectancy": "Life expectancy at birth. Welfare and public-health proxy.",
        "expected_schooling": "Expected years of schooling. Future educational opportunity proxy.",
        "mean_schooling": "Mean years of schooling. Realized educational attainment proxy.",
        "gdp_per_capita": "GDP per capita. Economic opportunity and living-standard proxy.",
        "distance": "Geographic distance between origin and destination. Migration friction proxy.",
        "push_count": "Number of articles classified as push in the time period.",
        "pull_count": "Number of articles classified as pull in the time period.",
        "mixed_count": "Number of articles classified as mixed in the time period.",
        "unclear_count": "Number of articles classified as unclear in the time period.",
        "n_articles": "Total number of migration-related articles in the time period.",
    }

def build_url_prompt(url_items: List[dict]) -> str:
    prompt = (
        "You are a URL-only migration-pressure classifier.\n"
        "Use ONLY lexical tokens found in the URL string. No outside knowledge. No invented details.\n\n"
        "Country-level interpretation:\n"
        "Assume the article concerns migration pressure affecting the country referenced in the URL.\n"
        "- push: signals conditions that may cause people to LEAVE that country\n"
        "- pull: signals conditions that may cause people to ARRIVE in that country\n"
        "Interpret pressure relative to the country mentioned or implied by the URL.\n\n"
        "Goal:\n"
        "For each URL, classify migration pressure:\n"
        "- push: conditions encouraging emigration (war, crisis, crackdown, collapse, unemployment, disaster, displacement)\n"
        "- pull: conditions encouraging immigration (jobs, hiring, growth, investment, visa, asylum, residency, aid, safety)\n"
        "- mixed: both push and pull indicators appear\n"
        "- neutral: no clear migration-related tokens\n"
        "- unclear: URL too short, generic, ID-like, or ambiguous\n\n"
        "Language rule:\n"
        "Reasons MUST be written in English even if the URL tokens are non-English.\n"
        "Quote the exact token(s) from the URL that triggered the classification.\n\n"
        "Output (STRICT JSONL, one object per line, no extra text):\n"
        '{"row_id": <int>, "pressure": "push|pull|mixed|neutral|unclear", "reason": "<=12 English words; include exact URL token(s)"}\n\n'
        "ITEMS (tab-separated):\n"
        "<row_id>\\t<url>\n"
    )
    for it in url_items:
        prompt += f"{int(it['row_id'])}\t{str(it['url'])}\n"
    return prompt


def build_article_prompt(article_text: str) -> str:
    return (
        "You are an article-based migration-pressure classifier.\n"
        "Use ONLY the provided article text. No outside knowledge. No invented details.\n\n"
        "Country-level interpretation:\n"
        "Classify migration pressure relative to the main country discussed in the article.\n"
        "- push: conditions that may cause people to LEAVE that country\n"
        "- pull: conditions that may cause people to ARRIVE in that country\n\n"
        "Goal:\n"
        "Classify the article as one of:\n"
        "- push\n- pull\n- mixed\n- neutral\n- unclear\n\n"
        "Definitions:\n"
        "- push: war, violence, persecution, repression, unemployment, collapse, disaster, displacement\n"
        "- pull: jobs, hiring, growth, investment, visas, asylum access, residency, aid, safety, stability\n"
        "- mixed: both push and pull are meaningfully present\n"
        "- neutral: article is not clearly about migration-driving conditions\n"
        "- unclear: article text is too vague or insufficient\n\n"
        "Language rule:\nReason MUST be written in English.\n\n"
        "Output (STRICT JSON ONLY, no markdown, no extra text):\n"
        '{"pressure": "push|pull|mixed|neutral|unclear", "reason": "<=16 English words"}\n\n'
        f"ARTICLE:\n{article_text}\n"
    )


def run_batch_url_pressure(
    url_items: List[dict], model: str, host: str, timeout_sec: int, temperature: float
) -> List[dict]:
    out = run_ollama(
        build_url_prompt(url_items),
        model=model,
        host=host,
        timeout_sec=timeout_sec,
        temperature=temperature,
    )
    if not out or out.startswith("[model_error]"):
        return []

    results = []
    for line in out.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "row_id" in obj and "pressure" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results


def batch_process_urls(
    url_items: List[dict],
    model: str,
    host: str,
    batch_size: int,
    timeout_sec: int,
    temperature: float,
) -> Dict[int, dict]:
    pressure_map: Dict[int, dict] = {}
    for start in range(0, len(url_items), batch_size):
        batch = url_items[start : start + batch_size]
        batch_res = run_batch_url_pressure(batch, model=model, host=host, timeout_sec=timeout_sec, temperature=temperature)
        for r in batch_res:
            pressure_map[int(r["row_id"])] = {
                "pressure": (r.get("pressure") or "unclear").strip().lower(),
                "reason": (r.get("reason") or "").strip(),
            }
        time.sleep(0.2)
    return pressure_map


def run_article_pressure(article_text: str, model: str, host: str, timeout_sec: int, temperature: float) -> dict:
    out = run_ollama(
        build_article_prompt(article_text),
        model=model,
        host=host,
        timeout_sec=timeout_sec,
        temperature=temperature,
    )
    if not out or out.startswith("[model_error]"):
        return {"pressure": "unclear", "reason": ""}
    try:
        obj = json.loads(out)
        return {
            "pressure": (obj.get("pressure") or "unclear").strip().lower(),
            "reason": (obj.get("reason") or "").strip(),
        }
    except Exception:
        pass
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            return {
                "pressure": (obj.get("pressure") or "unclear").strip().lower(),
                "reason": (obj.get("reason") or "").strip(),
            }
        except Exception:
            continue
    return {"pressure": "unclear", "reason": ""}


def run_mode3_articles(
    article_csv: str | Path,
    output_dir: str | Path,
    ollama_host: str = "http://127.0.0.1:11434",
    url_model_1: str = QWEN_7B,
    url_model_2: str = QWEN_72B,
    article_model: str = QWEN_7B,
    temperature: float = 0.2,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = sanitize_article_df(pd.read_csv(article_csv))

    url_items = []
    for i, row in df.iterrows():
        url = str(row.get("URL", "")).strip()
        if url and url.lower() != "nan":
            url_items.append({"row_id": i, "url": url})

    map_url_1 = batch_process_urls(url_items, model=url_model_1, host=ollama_host, batch_size=10, timeout_sec=180, temperature=temperature)

    df["pressure_url_1"] = [map_url_1.get(i, {"pressure": "unclear"})["pressure"] for i in range(len(df))]
    df["reason_url_1"] = [map_url_1.get(i, {"reason": ""})["reason"] for i in range(len(df))]

    unclear_after_url_1 = [it for it in url_items if df.at[int(it["row_id"]), "pressure_url_1"] in RECHECK_LABELS]
    map_url_2: Dict[int, dict] = {}
    if unclear_after_url_1:
        map_url_2 = batch_process_urls(unclear_after_url_1, model=url_model_2, host=ollama_host, batch_size=5, timeout_sec=300, temperature=temperature)

    df["pressure_url_2"] = [map_url_2.get(i, {}).get("pressure", "") for i in range(len(df))]
    df["reason_url_2"] = [map_url_2.get(i, {}).get("reason", "") for i in range(len(df))]

    article_candidates = []
    for it in url_items:
        rid = int(it["row_id"])
        p1 = df.at[rid, "pressure_url_1"]
        p2 = df.at[rid, "pressure_url_2"]
        if p1 in RECHECK_LABELS and (not p2 or p2 in RECHECK_LABELS):
            article_candidates.append(it)

    article_texts: Dict[int, str] = {}
    article_results: Dict[int, dict] = {}
    for it in article_candidates:
        rid = int(it["row_id"])
        url = str(it["url"])
        existing_text = str(df.at[rid, "article_text"] or "").strip() if "article_text" in df.columns else ""
        article_text = existing_text if len(existing_text) >= ARTICLE_MIN_LEN else get_article_text(url)
        article_texts[rid] = article_text
        if not article_text or len(article_text) < ARTICLE_MIN_LEN:
            article_results[rid] = {"pressure": "unclear", "reason": ""}
            continue
        article_results[rid] = run_article_pressure(article_text, model=article_model, host=ollama_host, timeout_sec=240, temperature=temperature)
        time.sleep(0.5)

    # Try to fetch article text even for non-fallback rows if missing, for later evidence prompt.
    for it in url_items:
        rid = int(it["row_id"])
        if rid not in article_texts:
            existing_text = str(df.at[rid, "article_text"] or "").strip() if "article_text" in df.columns else ""
            article_texts[rid] = existing_text if existing_text else get_article_text(str(it["url"]))

    df["article_text"] = [article_texts.get(i, str(df.at[i, "article_text"] or "")) for i in range(len(df))]
    df["pressure_article"] = [article_results.get(i, {}).get("pressure", "") for i in range(len(df))]
    df["reason_article"] = [article_results.get(i, {}).get("reason", "") for i in range(len(df))]

    final_pressure = []
    final_reason = []
    final_source = []
    for i in range(len(df)):
        p1 = df.at[i, "pressure_url_1"]
        r1 = df.at[i, "reason_url_1"]
        p2 = df.at[i, "pressure_url_2"]
        r2 = df.at[i, "reason_url_2"]
        pa = df.at[i, "pressure_article"]
        ra = df.at[i, "reason_article"]

        if p1 not in RECHECK_LABELS:
            final_pressure.append(p1)
            final_reason.append(r1)
            final_source.append("url_1")
        elif p2 and p2 not in RECHECK_LABELS:
            final_pressure.append(p2)
            final_reason.append(r2)
            final_source.append("url_2")
        elif pa and pa not in RECHECK_LABELS:
            final_pressure.append(pa)
            final_reason.append(ra)
            final_source.append("article")
        else:
            final_pressure.append("unclear")
            final_reason.append(r1 or r2 or ra)
            final_source.append("unresolved")

    df["pressure_final"] = final_pressure
    df["reason_final"] = final_reason
    df["final_source"] = final_source
    df["mode_used"] = 3
    df = add_feature_mention_columns(df)

    out_path = output_dir / "mode3_article_outputs.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved article outputs to {out_path}")
    return df


def build_monthly_article_features(processed_articles_df: pd.DataFrame) -> pd.DataFrame:
    df = sanitize_article_df(processed_articles_df)

    labels = ["push", "pull", "mixed", "unclear"]
    for label in labels:
        df[f"is_{label}"] = (df["pressure_final"] == label).astype(int)

    agg_dict = {
        "push_count": ("is_push", "sum"),
        "pull_count": ("is_pull", "sum"),
        "mixed_count": ("is_mixed", "sum"),
        "unclear_count": ("is_unclear", "sum"),
        "n_articles": ("pressure_final", "size"),
    }
    for feature in FEATURE_KEYWORDS:
        mention_col = f"{feature}_mentions"
        if mention_col in df.columns:
            agg_dict[mention_col] = (mention_col, "sum")

    monthly = df.groupby("month", as_index=False).agg(**agg_dict)

    monthly["push_ratio"] = monthly["push_count"] / monthly["n_articles"]
    monthly["pull_ratio"] = monthly["pull_count"] / monthly["n_articles"]
    monthly["mixed_ratio"] = monthly["mixed_count"] / monthly["n_articles"]
    monthly["unclear_ratio"] = monthly["unclear_count"] / monthly["n_articles"]
    monthly = monthly_mentions_to_penalties(monthly)

    return monthly

def merge_article_features_into_panel(panel_df: pd.DataFrame, monthly_article_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_panel_month_column(panel_df)

    article_cols = [
        "push_count", "pull_count", "mixed_count", "unclear_count",
        "n_articles", "push_ratio", "pull_ratio", "mixed_ratio", "unclear_ratio"
    ] + [f"{feat}_mentions" for feat in FEATURE_KEYWORDS] + [f"w_{feat}" for feat in FEATURE_KEYWORDS]
    df = df.drop(columns=[c for c in article_cols if c in df.columns], errors="ignore")

    df = df.merge(monthly_article_df, on="month", how="left")

    for col in article_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

# =============================================================================
# Penalty generation from processed article corpus
# =============================================================================
def build_feature_descriptions(feature_names: List[str]) -> Dict[str, str]:
    descriptions = {
        "log_origin_population": "Log of origin-country population. Larger origin populations can create more potential migrants.",
        "log_dest_population": "Log of destination-country population. Larger destinations may absorb more migrants.",
        "log_origin_gdp_pc": "Log of origin-country GDP per capita. Lower origin income can act as push pressure.",
        "log_dest_gdp_pc": "Log of destination-country GDP per capita. Higher destination income can act as pull pressure.",
        "origin_expected_schooling": "Expected years of schooling in the origin country, a development and opportunity proxy.",
        "dest_expected_schooling": "Expected years of schooling in the destination country, a development and opportunity proxy.",
        "origin_life_expectancy": "Origin-country life expectancy, a welfare and development proxy.",
        "dest_life_expectancy": "Destination-country life expectancy, a welfare and development proxy.",
        "origin_gpi": "Origin-country Global Peace Index. Worse peace conditions may push migration.",
        "dest_gpi": "Destination-country Global Peace Index. Safer destinations may attract migrants.",
        "origin_hdi": "Origin-country HDI, a broad development proxy.",
        "dest_hdi": "Destination-country HDI, a broad development proxy.",
        "log_distance_km": "Log geographic distance between origin and destination. Distance usually increases migration friction.",
        "gdp_pc_diff": "Destination GDP per capita minus origin GDP per capita, an economic gap feature.",
        "hdi_diff": "Destination HDI minus origin HDI, a development gap feature.",
        "gpi_diff": "Destination GPI minus origin GPI, a safety or instability gap feature.",
        "push_ratio": "Share of migration-related articles labeled as push pressure for the corridor or period.",
        "pull_ratio": "Share of migration-related articles labeled as pull pressure for the corridor or period.",
        "mixed_ratio": "Share of articles labeled mixed, meaning both push and pull cues are present.",
        "unclear_ratio": "Share of articles labeled unclear, meaning weak or ambiguous migration evidence.",
        "n_articles": "Count of migration-related articles used to summarize the corridor or period.",
        "push_count": "Raw count of push-labeled articles.",
        "pull_count": "Raw count of pull-labeled articles.",
        "mixed_count": "Raw count of mixed-labeled articles.",
        "unclear_count": "Raw count of unclear-labeled articles.",
    }
    return {name: descriptions.get(name, f"Feature named {name} in the migration flow model.") for name in feature_names}


def build_penalty_prompt_from_articles(processed_articles_df: pd.DataFrame, feature_names: List[str]) -> str:
    feature_desc = build_feature_descriptions(feature_names)
    feature_block = "\n".join(f"- {name}: {feature_desc[name]}" for name in feature_names)

    article_blocks = []
    for idx, row in processed_articles_df.iterrows():
        excerpt = str(row.get("article_text", "") or "")[:1600]
        article_blocks.append(
            f"ARTICLE {idx+1}\n"
            f"Corridor: {row.get('corridor', '')}\n"
            f"Year: {row.get('year', '')}\n"
            f"Title: {row.get('title', '')}\n"
            f"URL: {row.get('URL', '')}\n"
            f"Pressure final: {row.get('pressure_final', '')}\n"
            f"Reason final: {row.get('reason_final', '')}\n"
            f"Excerpt:\n{excerpt}\n"
        )
    article_context = "\n---\n".join(article_blocks)

    required_json = {
        "scores": [
            {"feature": feat, "rank": 1, "reason": "brief English reason"}
            for feat in feature_names
        ]
    }

    return f"""
You are helping build a migration-flow prediction model using LLM-Lasso.

Your task is to assign each base feature concept to a relevance tier based on the migration-related article evidence.

Important:
- Score the feature concepts themselves, before any origin/destination expansion.
- Do NOT score origin and destination separately.
- Do NOT score difference columns separately.
- Later, these tiers will be converted into penalties for derived model columns.

Interpretation:
- Tier 1 = highly relevant
- Tier 2 = moderately relevant
- Tier 3 = weakly relevant

Rules:
- You MUST score EVERY feature listed below.
- Ties ARE NOT allowed.
- You should use more than one tier whenever the evidence supports it.
- Do NOT assign the same tier to every feature.
- Use article evidence plus migration reasoning.
- Features do not need to be mentioned literally in the articles to be relevant.
- Think comparatively: which feature concepts seem most central vs less central for explaining migration in this corridor and time period?

How to think:
- If the article evidence suggests conflict, insecurity, persecution, displacement, war, or instability, features like GPI, distance, and push_count may be more relevant.
- If the article evidence suggests destination attractiveness, aid, opportunity, safety, or support, features like GDP/capita, HDI, pull_count, and related development features may be more relevant.
- If a feature is only weakly connected to the article evidence, it should receive a worse tier.
- Structural variables can still be important even if not explicitly named in the articles.

Features:
{feature_block}

Article evidence:
{article_context}

Return ONLY valid JSON.
The JSON must contain exactly {len(feature_names)} score entries.
The "feature" names must exactly match the feature names listed above.

Required output structure:
{json.dumps(required_json, indent=2)}
""".strip()


def generate_penalties_from_articles(
    processed_articles_df: pd.DataFrame,
    feature_names: List[str],
    output_dir: str | Path,
    ollama_model: str = QWEN_7B,
    ollama_host: str = "http://127.0.0.1:11434",
) -> Dict[str, float]:
    output_dir = Path(output_dir)
    prompt = build_penalty_prompt_from_articles(processed_articles_df, feature_names)

    print("\n===== FINAL PENALTY PROMPT START =====\n")
    print(prompt[:12000])
    print("\n===== FINAL PENALTY PROMPT END =====\n")

    out = run_ollama(
        prompt,
        model=ollama_model,
        host=ollama_host,
        timeout_sec=300,
        temperature=0,
        format_json=False,
    )
    raw = extract_json_object(out)

    with open(output_dir / "llm_feature_scores_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    rows = raw.get("scores", [])
    if not isinstance(rows, list):
        raise ValueError("LLM output does not contain a valid 'scores' list.")

    score_map = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        feat = str(r.get("feature", "")).strip()
        if not feat:
            continue
        score_map[feat] = float(r.get("penalty", 1.0))

    missing = [feat for feat in feature_names if feat not in score_map]
    extra = [feat for feat in score_map if feat not in feature_names]

    if missing:
        raise ValueError(f"LLM did not return penalties for all features. Missing: {missing}")
    if extra:
        raise ValueError(f"LLM returned unknown features: {extra}")

    clean_scores = {}
    for feat in feature_names:
        value = float(score_map[feat])
        clean_scores[feat] = min(max(value, 0.1), 1.0)

    pd.DataFrame({
        "feature": list(clean_scores.keys()),
        "llm_penalty_score": list(clean_scores.values()),
    }).to_csv(output_dir / "llm_feature_scores.csv", index=False)

    with open(output_dir / "llm_feature_scores.json", "w", encoding="utf-8") as f:
        json.dump(clean_scores, f, indent=2)

    return clean_scores


# =============================================================================
# Paper-style LLM-Lasso
# =============================================================================
def load_country_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["Country_key"] = df["Country"].map(normalize_country)
    rename_map = {
        "Global_Peace_Index": "gpi",
        "Human Development Index (HDI) ": "hdi",
        "Life expectancy at birth": "life_expectancy",
        "Expected years of schooling": "expected_schooling",
        "Mean years of schooling": "mean_schooling",
        "GDP/Capita": "gdp_per_capita",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    keep = ["Country_key"] + list(rename_map.keys())
    return df[keep].rename(columns=rename_map)


def enrich_panel_with_country_features(panel_df: pd.DataFrame, country_df: pd.DataFrame) -> pd.DataFrame:
    df = panel_df.copy()
    for col in ["origin", "destination"]:
        if col not in df.columns:
            raise ValueError(f"Input panel CSV must contain a '{col}' column.")
    if "migration_flow" not in df.columns:
        raise ValueError("Input panel CSV must contain a 'migration_flow' column.")

    df["origin_key"] = df["origin"].map(normalize_country)
    df["destination_key"] = df["destination"].map(normalize_country)

    origin_features = country_df.add_prefix("origin_")
    dest_features = country_df.add_prefix("dest_")
    df = df.merge(origin_features, left_on="origin_key", right_on="origin_Country_key", how="left")
    df = df.merge(dest_features, left_on="destination_key", right_on="dest_Country_key", how="left")

    df["distance_km"] = df.apply(
        lambda r: haversine_km(r["origin_latitude"], r["origin_longitude"], r["dest_latitude"], r["dest_longitude"])
        if pd.notna(r["origin_latitude"]) and pd.notna(r["origin_longitude"]) and pd.notna(r["dest_latitude"]) and pd.notna(r["dest_longitude"])
        else np.nan,
        axis=1,
    )

    if "origin_population" in df.columns:
        df["log_origin_population"] = np.log1p(df["origin_population"].clip(lower=0))
    if "dest_population" in df.columns:
        df["log_dest_population"] = np.log1p(df["dest_population"].clip(lower=0))

    df["log_origin_gdp_pc"] = np.log1p(df["origin_gdp_per_capita"].clip(lower=0))
    df["log_dest_gdp_pc"] = np.log1p(df["dest_gdp_per_capita"].clip(lower=0))
    df["log_distance_km"] = np.log1p(df["distance_km"].clip(lower=0))

    df["gdp_pc_diff"] = df["dest_gdp_per_capita"] - df["origin_gdp_per_capita"]
    df["hdi_diff"] = df["dest_hdi"] - df["origin_hdi"]
    df["gpi_diff"] = df["dest_gpi"] - df["origin_gpi"]
    df["pair_id"] = df["origin_key"] + "__" + df["destination_key"]
    df["log_migration_flow"] = np.log1p(df["migration_flow"].clip(lower=0))
    return df


def choose_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_candidates = [
        "log_origin_gdp_pc",
        "log_dest_gdp_pc",
        "origin_expected_schooling",
        "dest_expected_schooling",
        "origin_life_expectancy",
        "dest_life_expectancy",
        "origin_gpi",
        "dest_gpi",
        "origin_hdi",
        "dest_hdi",
        "log_distance_km",
        "gdp_pc_diff",
        "hdi_diff",
        "gpi_diff",
        "push_count",
        "pull_count",
        "mixed_count",
        "unclear_count",
        "n_articles",
    ]
    numeric_features = [c for c in numeric_candidates if c in df.columns]
    categorical_features = [c for c in ["pair_id"] if c in df.columns]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def build_base_penalty_vector(transformed_feature_names: List[str], llm_numeric_scores: Dict[str, float]) -> np.ndarray:
    base_penalties = []
    for feat in transformed_feature_names:
        if feat.startswith("num__"):
            raw_name = feat.replace("num__", "", 1)
            penalty = float(llm_numeric_scores.get(raw_name, 1.0))
        else:
            penalty = 1.0
        base_penalties.append(penalty)
    return np.asarray(base_penalties, dtype=float)


def apply_penalty_transform(X: np.ndarray, penalties_v: np.ndarray, eta: int, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    w = np.power(np.maximum(penalties_v, eps), eta)
    X_tilde = X / w
    return X_tilde, w

def build_base_penalty_prompt_from_articles(
    processed_articles_df: pd.DataFrame,
    base_feature_names: List[str]
) -> str:
    desc = build_base_feature_descriptions()
    feature_block = "\n".join(
        f"- {name}: {desc.get(name, name)}" for name in base_feature_names
    )

    article_blocks = []
    for idx, row in processed_articles_df.iterrows():
        excerpt = str(row.get("article_text", "") or "")[:1200]
        article_blocks.append(
            f"ARTICLE {idx+1}\n"
            f"Corridor: {row.get('corridor', '')}\n"
            f"Year: {row.get('year', '')}\n"
            f"Title: {row.get('title', '')}\n"
            f"Pressure final: {row.get('pressure_final', '')}\n"
            f"Reason final: {row.get('reason_final', '')}\n"
            f"Excerpt:\n{excerpt}\n"
        )

    article_context = "\n---\n".join(article_blocks)

    required_json = {
        "scores": [
            {"feature": feat, "rank": 1, "reason": "brief English reason"}
            for feat in base_feature_names
        ]
    }

    return f"""
You are helping build a migration-flow prediction model using LLM-Lasso.

Your task is to assign each base feature concept to a relevance tier based on the migration-related article evidence.

Important:
- Score the feature concepts themselves, before any origin/destination expansion.
- Do NOT score origin and destination separately.
- Do NOT score difference columns separately.
- Later, these tiers will be converted into penalties for derived model columns.

Interpretation:
- Tier 1 = highly relevant
- Tier 2 = moderately relevant
- Tier 3 = weakly relevant

Rules:
- You MUST score EVERY feature listed below.
- Ties ARE NOT allowed.
- You should use more than one tier whenever the evidence supports it.
- Do NOT assign the same tier to every feature.
- Use article evidence plus migration reasoning.
- Features do not need to be mentioned literally in the articles to be relevant.
- Think comparatively: which feature concepts seem most central vs less central for explaining migration in this corridor and time period?

Formatting rules:
- Return ONLY valid JSON.
- The JSON must contain exactly {len(base_feature_names)} score entries.
- The "feature" names must exactly match the feature names listed below.
- Each score entry must be its own separate object inside the "scores" array.
- Do NOT nest one feature inside another.
- Each object must contain exactly these keys: "feature", "rank", "reason".
- Do not include markdown fences.
- Do not include trailing commas.

How to think:
- If the article evidence suggests conflict, insecurity, persecution, displacement, war, or instability, features like GPI, distance, and push_count may be more relevant.
- If the article evidence suggests destination attractiveness, aid, opportunity, safety, or support, features like GDP/capita, HDI, pull_count, and related development features may be more relevant.
- If a feature is only weakly connected to the article evidence, it should receive a worse tier.
- Structural variables can still be important even if not explicitly named in the articles.

Features:
{feature_block}

Article evidence:
{article_context}

Required output structure:
{json.dumps(required_json, indent=2)}
""".strip()


def fit_paper_style_llm_lasso(
    df: pd.DataFrame,
    llm_scores: Dict[str, float],
    output_dir: str | Path,
    eta_max: int = 10,run_label="model"
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_features, categorical_features = choose_feature_columns(df)
    if not numeric_features and not categorical_features:
        raise ValueError("No usable features were found. Add numeric variables to the panel CSV.")

    X_df = df[numeric_features + categorical_features].copy()
    y = df["log_migration_flow"].astype(float).to_numpy()

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_proc = preprocessor.fit_transform(X_df)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    transformed_feature_names = list(preprocessor.get_feature_names_out())
    base_penalties_v = build_base_penalty_vector(transformed_feature_names, llm_scores)

    n_rows = len(df)
    outer_folds = max(2, min(5, n_rows // 3 if n_rows >= 6 else 2))
    inner_folds = outer_folds
    eta_grid = list(range(1, eta_max + 1))

    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    eta_results = []
    for eta in eta_grid:
        fold_mse = []
        fold_mae = []
        for train_idx, val_idx in outer_cv.split(X_proc):
            X_train = X_proc[train_idx]
            X_val = X_proc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            X_train_tilde, w = apply_penalty_transform(X_train, base_penalties_v, eta=eta)
            X_val_tilde = X_val / w

            inner_model = LassoCV(cv=inner_folds, random_state=42, max_iter=50000, n_alphas=200)
            inner_model.fit(X_train_tilde, y_train)
            val_pred = inner_model.predict(X_val_tilde)
            fold_mse.append(mean_squared_error(y_val, val_pred))
            fold_mae.append(mean_absolute_error(y_val, val_pred))

        eta_results.append({"eta": eta, "cv_mse": float(np.mean(fold_mse)), "cv_mae": float(np.mean(fold_mae))})

    #eta_df = pd.DataFrame(eta_results).sort_values(["cv_mse", "cv_mae"], ascending=True)
    best_eta = 1
    eta_df = pd.DataFrame([{
        "eta": best_eta,
        "cv_mse": None,
        "cv_mae": None,
        "note": "eta forced to 1 so LLM penalties are always used"
    }])


    X_tilde_full, final_w = apply_penalty_transform(X_proc, base_penalties_v, eta=best_eta)
    final_model = LassoCV(cv=inner_folds, random_state=42, max_iter=50000, n_alphas=200)
    final_model.fit(X_tilde_full, y)

    pred_log = final_model.predict(X_tilde_full)
    pred_flow = np.expm1(pred_log)
    theta = final_model.coef_
    beta = theta / final_w

    coef_df = pd.DataFrame({
        "feature": transformed_feature_names,
        "base_penalty_v": base_penalties_v,
        "best_eta": best_eta,
        "final_penalty_weight": np.power(base_penalties_v, best_eta),
        "coefficient": beta,
        "abs_coefficient": np.abs(beta),
    }).sort_values("abs_coefficient", ascending=False)
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

    print("\n=== TOP COEFFICIENTS ===")
    print(coef_df.sort_values("abs_coefficient", ascending=False).head(10))

    pred_df = df.copy()
    pred_df["pred_log_migration_flow"] = pred_log
    pred_df["pred_migration_flow"] = pred_flow
    pred_df["residual_log"] = pred_df["log_migration_flow"] - pred_df["pred_log_migration_flow"]

    metrics = {
        "rows": int(len(df)),
        "outer_cv_folds": int(outer_folds),
        "inner_cv_folds": int(inner_folds),
        "best_eta": int(best_eta),
        "alpha": float(final_model.alpha_),
        "r2_train": float(r2_score(y, pred_log)),
        "rmse_train_log": float(np.sqrt(mean_squared_error(y, pred_log))),
        "mae_train_log": float(mean_absolute_error(y, pred_log)),
        "numeric_features_scored_by_llm": numeric_features,
        "categorical_features_neutral_penalty": categorical_features,
    }

    pred_df.to_csv(output_dir / f"{run_label}_predictions.csv", index=False)
    coef_df.to_csv(output_dir / f"{run_label}_coefficients.csv", index=False)
    eta_df.to_csv(output_dir / f"{run_label}_eta_cv.csv", index=False)
    with open(output_dir / f"{run_label}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(f"  {output_dir / f'{run_label}_predictions.csv'}")
    print(f"  {output_dir / f'{run_label}_coefficients.csv'}")
    print(f"  {output_dir / f'{run_label}_eta_cv.csv'}")
    print(f"  {output_dir / f'{run_label}_metrics.json'}")
    print("\nBest eta:", best_eta)
    print("\nTop coefficients:")
    print(coef_df.head(20).to_string(index=False))
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

def normalize_score_rows(rows, expected_features):
    fixed_rows = []

    for r in rows:
        if not isinstance(r, dict):
            continue

        # Normal case:
        # {"feature": "...", "rank": 2, "reason": "..."}
        if "feature" in r:
            feat = str(r.get("feature", "")).strip()
            if feat:
                fixed_rows.append({
                    "feature": feat,
                    "rank": r.get("rank"),
                    "reason": r.get("reason", ""),
                })

        # Recovery case:
        # {"feature":"mixed_count","rank":3,"reason":"...",
        #  "unclear_count":{"rank":5,"reason":"..."},
        #  "n_articles":{"rank":1,"reason":"..."}}
        for k, v in r.items():
            if k in {"feature", "rank", "reason"}:
                continue
            if k in expected_features and isinstance(v, dict):
                fixed_rows.append({
                    "feature": k,
                    "rank": v.get("rank"),
                    "reason": v.get("reason", ""),
                })

    # Deduplicate by feature; last one wins
    dedup = {}
    for row in fixed_rows:
        feat = str(row.get("feature", "")).strip()
        if feat:
            dedup[feat] = row

    return list(dedup.values())

def generate_base_penalties_from_articles(
    processed_articles_df: pd.DataFrame,
    base_feature_names: List[str],
    output_dir: str | Path,
    ollama_model: str = QWEN_7B,
    ollama_host: str = "http://127.0.0.1:11434",
) -> Dict[str, float]:
    output_dir = Path(output_dir)
    prompt = build_base_penalty_prompt_from_articles(processed_articles_df, base_feature_names)

    print("\n===== FINAL BASE PENALTY PROMPT START =====\n")
    print(prompt[:12000])
    print("\n===== FINAL BASE PENALTY PROMPT END =====\n")

    out = run_ollama(
        prompt,
        model=ollama_model,
        host=ollama_host,
        timeout_sec=300,
        temperature=0,
        format_json=True,
    )

    print("\n===== RAW BASE LLM OUTPUT START =====\n")
    print(repr(out[:5000]))
    print("\n===== RAW BASE LLM OUTPUT END =====\n")

    raw = extract_json_object(out)

    with open(output_dir / "llm_base_feature_scores_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    rows = raw.get("scores", [])
    if not isinstance(rows, list):
        raise ValueError("LLM output does not contain a valid 'scores' list.")

    rows = normalize_score_rows(rows, set(base_feature_names))

    rank_map = {}
    for r in rows:
        if not isinstance(r, dict):
            continue

        feat = str(r.get("feature", "")).strip()
        rank = r.get("rank", None)

        if feat not in base_feature_names:
            continue
        if rank is None:
            continue

        try:
            rank_map[feat] = int(rank)
        except Exception:
            continue

    missing = [feat for feat in base_feature_names if feat not in rank_map]
    extra = [feat for feat in rank_map if feat not in base_feature_names]

    if missing:
        print("\n===== NORMALIZED SCORE ROWS =====\n")
        print(json.dumps(rows, indent=2))
        raise ValueError(f"Missing base features in LLM output: {missing}")
    if extra:
        raise ValueError(f"Unknown base features in LLM output: {extra}")

    ranks = [int(rank_map[f]) for f in base_feature_names]
    unique_ranks = sorted(set(ranks))

    if len(unique_ranks) < 2:
        raise ValueError(
            f"LLM returned no meaningful rank variation: {unique_ranks}"
        )

    def rank_to_penalty(rank: int, unique_ranks: List[int]) -> float:
        if len(unique_ranks) == 1:
            return 0.1
        pos = unique_ranks.index(rank)
        return 0.1 + 0.9 * (pos / (len(unique_ranks) - 1))

    clean_scores = {
        feat: rank_to_penalty(int(rank_map[feat]), unique_ranks)
        for feat in base_feature_names
    }

    pd.DataFrame({
        "base_feature": base_feature_names,
        "rank": [rank_map[f] for f in base_feature_names],
        "llm_penalty_score": [clean_scores[f] for f in base_feature_names],
    }).to_csv(output_dir / "llm_base_feature_scores.csv", index=False)

    return clean_scores


def get_base_to_model_feature_map() -> Dict[str, List[str]]:
    return {
        "gpi": ["origin_gpi", "dest_gpi", "gpi_diff"],
        "hdi": ["origin_hdi", "dest_hdi", "hdi_diff"],
        "life_expectancy": ["origin_life_expectancy", "dest_life_expectancy"],
        "expected_schooling": ["origin_expected_schooling", "dest_expected_schooling"],
        "mean_schooling": [],  # fill this later if you add it to country_df
        "gdp_per_capita": ["log_origin_gdp_pc", "log_dest_gdp_pc", "gdp_pc_diff"],
        "distance": ["log_distance_km"],
        "push_count": ["push_count"],
        "pull_count": ["pull_count"],
        "mixed_count": ["mixed_count"],
        "unclear_count": ["unclear_count"],
        "n_articles": ["n_articles"],
    }

def expand_base_penalties_to_model_features(
    base_scores: Dict[str, float],
    numeric_feature_names: List[str]
) -> Dict[str, float]:
    mapping = get_base_to_model_feature_map()
    expanded = {}

    for base_feat, cols in mapping.items():
        penalty = float(base_scores.get(base_feat, 1.0))
        for col in cols:
            expanded[col] = penalty

    for feat in numeric_feature_names:
        if feat not in expanded:
            expanded[feat] = 1.0

    return expanded

def build_uniform_penalties(numeric_features: List[str]) -> Dict[str, float]:
    return {feat: 1.0 for feat in numeric_features}


def extract_monthly_base_penalties(monthly_feature_row: pd.Series) -> Dict[str, float]:
    penalties = {}
    for feature in BASE_FEATURES:
        col = f"w_{feature}"
        penalties[feature] = float(monthly_feature_row[col]) if col in monthly_feature_row and pd.notna(monthly_feature_row[col]) else 1.0
    return penalties

# =============================================================================
# Combined pipeline
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combined Mode-3 + article-aware LLM-Lasso pipeline. "
            "Reads article URLs, runs Mode-3 style article understanding, generates global feature penalties, "
            "then fits paper-style LLM-Lasso on the migration panel."
        )
    )
    parser.add_argument("--article_csv", required=True, help="CSV with corridor, year, title, URL columns.")
    parser.add_argument("--panel_csv", required=True, help="CSV with origin, destination, migration_flow, and optional news features.")
    parser.add_argument("--country_csv", default="HDI_GPI_with_GDP_LATLON.csv", help="Country feature lookup CSV.")
    parser.add_argument("--output_dir", default="mode3_llm_lasso_output", help="Directory where outputs are written.")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434", help="Base URL for Ollama API.")
    parser.add_argument("--mode3_url_model_1", default=QWEN_7B, help="Mode-3 first URL model.")
    parser.add_argument("--mode3_url_model_2", default=QWEN_72B, help="Mode-3 second URL model.")
    parser.add_argument("--mode3_article_model", default=QWEN_7B, help="Mode-3 article model.")
    parser.add_argument("--penalty_model", default=QWEN_7B, help="LLM used for final feature-penalty generation.")
    parser.add_argument("--eta_max", type=int, default=10, help="Maximum eta in the inverse-importance family.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_articles_df = run_mode3_articles(
        article_csv=args.article_csv,
        output_dir=output_dir,
        ollama_host=args.ollama_host,
        url_model_1=args.mode3_url_model_1,
        url_model_2=args.mode3_url_model_2,
        article_model=args.mode3_article_model,
        temperature=0.2,
    )

    monthly_article_df = build_monthly_article_features(processed_articles_df)
    print("\n=== MONTHLY ARTICLE FEATURES ===")
    print(monthly_article_df.head())


    panel_df = pd.read_csv(args.panel_csv)
    panel_df = merge_article_features_into_panel(panel_df, monthly_article_df)

    print("\n=== PANEL AFTER MERGE ===")
    print(panel_df[[
        "month",
        "push_count",
        "pull_count",
        "mixed_count",
        "unclear_count",
        "n_articles"
    ]].head(12))

    country_df = load_country_features(args.country_csv)
    model_df = enrich_panel_with_country_features(panel_df, country_df)

    print("\n=== MODEL DF FEATURE VARIATION ===")
 # 1. Get model features
    numeric_features, _ = choose_feature_columns(model_df)

    # 2. Build uniform (plain Lasso) penalties
    uniform_scores = {feat: 1.0 for feat in numeric_features}

    # 3. Build LLM penalties
    base_scores = generate_base_penalties_from_articles(
        processed_articles_df=processed_articles_df,
        base_feature_names=BASE_FEATURES,
        output_dir=output_dir,
        ollama_model=args.penalty_model,
        ollama_host=args.ollama_host,
    )

    llm_scores = expand_base_penalties_to_model_features(
        base_scores, numeric_features
    )

    # 4. Run BOTH models
    print("-------------------NO LLM------------------")
    print("uniform_scores:", uniform_scores)
    print("-------------------NO LLM------------------")
    print("-------------------USING LLM------------------")
    print("llm_scores:", llm_scores)
    print("-------------------USING LLM------------------")
    print
    # Plain Lasso (no LLM influence)
    fit_paper_style_llm_lasso(
        df=model_df,
        llm_scores=uniform_scores,
        output_dir=output_dir / "plain_lasso",
        eta_max=args.eta_max,
        run_label="plain_lasso",
    )

    fit_paper_style_llm_lasso(
        df=model_df,
        llm_scores=llm_scores,
        output_dir=output_dir / "llm_lasso",
        eta_max=args.eta_max,
        run_label="llm_lasso",
    )




if __name__ == "__main__":
    main()
