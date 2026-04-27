from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# HARD-CODED URL ROWS
# Put ONE row per article. migration_flow is your target for that row.
# You can change these values later.
# =============================================================================
EVENTS = [
    {
        "event_id": "ukr_pol_2022_11_npr",
        "origin": "Ukraine",
        "destination": "Poland",
        "url": "https://www.npr.org/2022/11/01/1132167234/russia-ukraine-war-unemployment-displaced-economy",
        "migration_flow": 1500000,
        "shared_border": 1,
    },
    {
        "event_id": "ukr_pol_2022_05_npr",
        "origin": "Ukraine",
        "destination": "Poland",
        "url": "https://www.npr.org/2022/05/10/1093066817/ukraine-war-gas-prices-refugees",
        "migration_flow": 3200000,
        "shared_border": 1,
    },
    {
        "event_id": "ukr_pol_2025_nbc",
        "origin": "Ukraine",
        "destination": "Poland",
        "url": "https://www.nbcnews.com/world/ukraine/ukrainian-refugees-years-after-russia-invasion-fear-return-rcna250438",
        "migration_flow": 950000,
        "shared_border": 1,
    },
    {
        "event_id": "mex_usa_2022_nbc",
        "origin": "Mexico",
        "destination": "United States",
        "url": "https://www.nbcnews.com/politics/immigration/migrant-border-crossings-fiscal-year-2022-topped-276-million-breaking-rcna53517",
        "migration_flow": 2760000,
        "shared_border": 1,
    },
]

COUNTRY_CSV = "HDI_GPI_with_GDP_LATLON.csv"
OUTPUT_DIR = "mode3_model4_output"

# =============================================================================
# MODE 3 CONFIG
# Keep the same logic:
# qwen7b URL -> qwen72b URL -> qwen7b ARTICLE if still unclear/neutral
# =============================================================================
GLOBAL_TEMP = 1.8
RECHECK_LABELS = ["unclear", "neutral"]

QWEN_7B = "qwen2.5:7b"
QWEN_72B = "qwen2.5:72b"

URL_BATCH_SIZE_SMALL = 10
URL_BATCH_SIZE_LARGE = 5
URL_TIMEOUT_SMALL = 180
URL_TIMEOUT_LARGE = 300
ARTICLE_TIMEOUT = 240
ARTICLE_MIN_LEN = 300
ARTICLE_CHAR_LIMIT = 9000

EARTH_RADIUS_KM = 6371.0088


# =============================================================================
# OLLAMA
# =============================================================================
def run_ollama(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    max_retries: int = 2,
    timeout_sec: int = 180,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    for attempt in range(max_retries + 1):
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
            out = (data.get("response") or "").strip()
            if out:
                return out
            return "[model_error] empty_output"

        except requests.Timeout:
            if attempt == max_retries:
                return "[model_error] timeout"
            time.sleep(2 ** attempt)

        except Exception as e:
            if attempt == max_retries:
                return f"[model_error] {e}"
            time.sleep(2 ** attempt)

    return "[model_error] unknown"


# =============================================================================
# MODE 3 PROMPTS
# =============================================================================
def build_url_prompt(url_items: List[Dict]) -> str:
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
        "{\"row_id\": <int>, \"pressure\": \"push|pull|mixed|neutral|unclear\", \"reason\": \"<=12 English words; include exact URL token(s)\"}\n\n"
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
        "- push\n"
        "- pull\n"
        "- mixed\n"
        "- neutral\n"
        "- unclear\n\n"
        "Definitions:\n"
        "- push: war, violence, persecution, repression, unemployment, collapse, disaster, displacement\n"
        "- pull: jobs, hiring, growth, investment, visas, asylum access, residency, aid, safety, stability\n"
        "- mixed: both push and pull are meaningfully present\n"
        "- neutral: article is not clearly about migration-driving conditions\n"
        "- unclear: article text is too vague or insufficient\n\n"
        "Language rule:\n"
        "Reason MUST be written in English.\n\n"
        "Output (STRICT JSON ONLY, no markdown, no extra text):\n"
        "{\"pressure\": \"push|pull|mixed|neutral|unclear\", \"reason\": \"<=16 English words\"}\n\n"
        "ARTICLE:\n"
        f"{article_text}\n"
    )


# =============================================================================
# MODE 3 URL + ARTICLE STEPS
# =============================================================================
def run_batch_url_pressure(url_items, model: str, timeout_sec: int = 180, temperature: float = 0.2):
    out = run_ollama(
        build_url_prompt(url_items),
        model=model,
        timeout_sec=timeout_sec,
        temperature=temperature,
    )

    if not out or out.startswith("[model_error]"):
        print(f"   -> {model} error: {out}")
        return []

    results = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue
        if line.lower().startswith("here are") or line.lower().startswith("here is"):
            continue
        if not line.startswith("{"):
            continue

        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "row_id" in obj and "pressure" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    return results


def batch_process_urls(
    url_items,
    model: str,
    batch_size: int,
    timeout_sec: int = 180,
    sleep_between_batches: float = 0.2,
    temperature: float = 0.2,
):
    pressure_map = {}

    for start in range(0, len(url_items), batch_size):
        batch = url_items[start:start + batch_size]
        print(f"[{model}] URL batch {start}..{start + len(batch) - 1} (size={len(batch)})")
        batch_res = run_batch_url_pressure(
            batch,
            model=model,
            timeout_sec=timeout_sec,
            temperature=temperature,
        )

        for r in batch_res:
            rid = r.get("row_id")
            if rid is None:
                continue
            pressure_map[int(rid)] = {
                "pressure": (r.get("pressure") or "unclear").strip().lower(),
                "reason": (r.get("reason") or "").strip(),
            }

        time.sleep(sleep_between_batches)

    return pressure_map


def get_article_text_newspaper(url: str, char_limit: int = ARTICLE_CHAR_LIMIT) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return (article.text or "").strip()[:char_limit]
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
        return " ".join(text.split())[:char_limit]
    except Exception:
        return ""


def get_article_text(url: str, char_limit: int = ARTICLE_CHAR_LIMIT) -> str:
    text = get_article_text_newspaper(url, char_limit=char_limit)
    if len(text) >= ARTICLE_MIN_LEN:
        return text
    return get_article_text_fallback(url, char_limit=char_limit)


def run_article_pressure(article_text: str, model: str, timeout_sec: int = ARTICLE_TIMEOUT, temperature: float = 0.2):
    out = run_ollama(
        build_article_prompt(article_text),
        model=model,
        timeout_sec=timeout_sec,
        temperature=temperature,
    )

    if not out or out.startswith("[model_error]"):
        return {"pressure": "unclear", "reason": ""}

    try:
        obj = json.loads(out.strip())
        return {
            "pressure": (obj.get("pressure") or "unclear").strip().lower(),
            "reason": (obj.get("reason") or "").strip(),
        }
    except Exception:
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


def run_mode3_on_events(events: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(events).copy()
    df["row_id"] = np.arange(len(df))

    url_items = [{"row_id": int(r["row_id"]), "url": r["url"]} for _, r in df.iterrows()]

    map_url_1 = batch_process_urls(
        url_items,
        model=QWEN_7B,
        batch_size=URL_BATCH_SIZE_SMALL,
        timeout_sec=URL_TIMEOUT_SMALL,
        temperature=GLOBAL_TEMP,
    )
    df["pressure_url_1"] = [map_url_1.get(i, {"pressure": "unclear"})["pressure"] for i in df["row_id"]]
    df["reason_url_1"] = [map_url_1.get(i, {"reason": ""})["reason"] for i in df["row_id"]]

    unclear_after_url_1 = [
        {"row_id": int(r["row_id"]), "url": r["url"]}
        for _, r in df.iterrows()
        if r["pressure_url_1"] in RECHECK_LABELS
    ]

    map_url_2 = {}
    if unclear_after_url_1:
        map_url_2 = batch_process_urls(
            unclear_after_url_1,
            model=QWEN_72B,
            batch_size=URL_BATCH_SIZE_LARGE,
            timeout_sec=URL_TIMEOUT_LARGE,
            temperature=GLOBAL_TEMP,
        )

    df["pressure_url_2"] = [map_url_2.get(i, {}).get("pressure", "") for i in df["row_id"]]
    df["reason_url_2"] = [map_url_2.get(i, {}).get("reason", "") for i in df["row_id"]]

    article_texts = {}
    article_results = {}

    for _, row in df.iterrows():
        rid = int(row["row_id"])
        needs_article = (
            row["pressure_url_1"] in RECHECK_LABELS
            and (not row["pressure_url_2"] or row["pressure_url_2"] in RECHECK_LABELS)
        )

        if not needs_article:
            article_texts[rid] = ""
            article_results[rid] = {"pressure": "", "reason": ""}
            continue

        print(f"[ARTICLE] Fetching row_id={rid}: {row['url']}")
        article_text = get_article_text(str(row["url"]))
        article_texts[rid] = article_text

        if not article_text or len(article_text) < ARTICLE_MIN_LEN:
            article_results[rid] = {"pressure": "unclear", "reason": ""}
            continue

        article_results[rid] = run_article_pressure(
            article_text,
            model=QWEN_7B,
            timeout_sec=ARTICLE_TIMEOUT,
            temperature=GLOBAL_TEMP,
        )
        time.sleep(0.5)

    df["article_text"] = [article_texts.get(i, "") for i in df["row_id"]]
    df["pressure_article"] = [article_results.get(i, {}).get("pressure", "") for i in df["row_id"]]
    df["reason_article"] = [article_results.get(i, {}).get("reason", "") for i in df["row_id"]]

    final_pressure = []
    final_reason = []
    final_source = []

    for _, row in df.iterrows():
        p1, r1 = row["pressure_url_1"], row["reason_url_1"]
        p2, r2 = row["pressure_url_2"], row["reason_url_2"]
        pa, ra = row["pressure_article"], row["reason_article"]

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
    return df


# =============================================================================
# MODEL 4-LIKE LASSO SIDE
# =============================================================================
def normalize_country(s: str) -> str:
    return str(s).strip().lower()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


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


def extract_year_month_from_url(url: str) -> tuple[Optional[int], Optional[int]]:
    m = re.search(r"/(20\\d{2})/(\\d{2})(?:/(\\d{2}))?/", url)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def add_article_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    labels = ["push", "pull", "mixed", "neutral", "unclear"]

    for lbl in labels:
        df[f"{lbl}_count"] = (df["pressure_final"] == lbl).astype(int)

    df["n_articles"] = 1

    for lbl in labels:
        df[f"{lbl}_ratio"] = df[f"{lbl}_count"] / df["n_articles"]

    score_map = {
        "push": 1.0,
        "pull": 1.0,
        "mixed": 0.75,
        "neutral": 0.0,
        "unclear": 0.0,
    }
    df["article_relevance_score"] = df["pressure_final"].map(score_map).fillna(0.0)

    return df


def enrich_panel_with_country_features(panel_df: pd.DataFrame, country_df: pd.DataFrame) -> pd.DataFrame:
    df = panel_df.copy()
    df["origin_key"] = df["origin"].map(normalize_country)
    df["destination_key"] = df["destination"].map(normalize_country)

    origin_features = country_df.add_prefix("origin_")
    dest_features = country_df.add_prefix("dest_")

    df = df.merge(origin_features, left_on="origin_key", right_on="origin_Country_key", how="left")
    df = df.merge(dest_features, left_on="destination_key", right_on="dest_Country_key", how="left")

    df["distance_km"] = df.apply(
        lambda r: haversine_km(
            r["origin_latitude"], r["origin_longitude"], r["dest_latitude"], r["dest_longitude"]
        )
        if pd.notna(r["origin_latitude"]) and pd.notna(r["origin_longitude"]) and pd.notna(r["dest_latitude"]) and pd.notna(r["dest_longitude"])
        else np.nan,
        axis=1,
    )

    df["log_origin_gdp_pc"] = np.log1p(df["origin_gdp_per_capita"].clip(lower=0))
    df["log_dest_gdp_pc"] = np.log1p(df["dest_gdp_per_capita"].clip(lower=0))
    df["log_distance_km"] = np.log1p(df["distance_km"].clip(lower=0))

    df["gdp_pc_diff"] = df["dest_gdp_per_capita"] - df["origin_gdp_per_capita"]
    df["hdi_diff"] = df["dest_hdi"] - df["origin_hdi"]
    df["gpi_diff"] = df["dest_gpi"] - df["origin_gpi"]

    df["pair_id"] = df["origin_key"] + "__" + df["destination_key"]
    df["log_migration_flow"] = np.log1p(df["migration_flow"].clip(lower=0))

    return df


def choose_feature_columns(df: pd.DataFrame):
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
        "shared_border",
        "article_relevance_score",
        "push_ratio",
        "pull_ratio",
        "mixed_ratio",
        "unclear_ratio",
        "n_articles",
        "push_count",
        "pull_count",
        "mixed_count",
        "unclear_count",
    ]
    numeric_features = [c for c in numeric_candidates if c in df.columns]
    categorical_features = [c for c in ["pair_id"] if c in df.columns]
    return numeric_features, categorical_features


def fit_lasso(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_features, categorical_features = choose_feature_columns(df)
    X = df[numeric_features + categorical_features].copy()
    y = df["log_migration_flow"].astype(float)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    n_rows = len(df)
    cv_folds = max(2, min(3, n_rows)) if n_rows >= 2 else 2

    model = Pipeline([
        ("prep", preprocessor),
        ("lasso", LassoCV(cv=cv_folds, random_state=42, max_iter=50000, n_alphas=100)),
    ])

    model.fit(X, y)
    pred_log = model.predict(X)
    pred_flow = np.expm1(pred_log)

    metrics = {
        "rows": int(len(df)),
        "cv_folds": int(cv_folds),
        "alpha": float(model.named_steps["lasso"].alpha_),
        "r2_train": float(r2_score(y, pred_log)),
        "rmse_train_log": float(np.sqrt(mean_squared_error(y, pred_log))),
        "mae_train_log": float(mean_absolute_error(y, pred_log)),
    }

    feature_names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["lasso"].coef_
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    pred_df = df.copy()
    pred_df["pred_log_migration_flow"] = pred_log
    pred_df["pred_migration_flow"] = pred_flow
    pred_df["residual_log"] = pred_df["log_migration_flow"] - pred_df["pred_log_migration_flow"]

    pred_df.to_csv(output_dir / "lasso_predictions.csv", index=False)
    coef_df.to_csv(output_dir / "lasso_coefficients.csv", index=False)

    with open(output_dir / "lasso_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Mode 3 on hardcoded events...")
    mode3_df = run_mode3_on_events(EVENTS)
    mode3_df["year"], mode3_df["month"] = zip(*mode3_df["url"].map(extract_year_month_from_url))
    mode3_df = add_article_feature_columns(mode3_df)
    mode3_df.to_csv(out_dir / "mode3_event_rows.csv", index=False)

    print("Merging country features...")
    country_df = load_country_features(COUNTRY_CSV)
    model_df = enrich_panel_with_country_features(mode3_df, country_df)
    model_df.to_csv(out_dir / "model_ready_panel.csv", index=False)

    print("Running Lasso...")
    fit_lasso(model_df, out_dir)

    print("Done.")
    print(f"Saved: {out_dir / 'mode3_event_rows.csv'}")
    print(f"Saved: {out_dir / 'model_ready_panel.csv'}")
    print(f"Saved: {out_dir / 'lasso_predictions.csv'}")
    print(f"Saved: {out_dir / 'lasso_coefficients.csv'}")
    print(f"Saved: {out_dir / 'lasso_metrics.json'}")


if __name__ == "__main__":
    main()