import time
import requests
import pandas as pd
import json
from newspaper import Article
from bs4 import BeautifulSoup
from googletrans import Translator
from urllib.parse import unquote

# =========================================================
# CONFIG
# =========================================================
GLOBAL_TEMP = 1.8
RECHECK_LABELS = ["unclear", "neutral"]

CSV_IN = "gdelt_events_basic.csv"
CSV_OUT = "migration_mode_3.csv"
MANUAL_EVAL_OUT = "manual_eval_mode_3.csv"
URL_COL = "sourceurl"

# MODE 3 ONLY:
# qwen7b URL -> qwen72b URL -> qwen7b ARTICLE if still unclear/neutral
QWEN_7B = "qwen2.5:7b"
QWEN_72B = "qwen2.5:72b"

TRANSLATE_REASONS = True
TRANSLATION_COLUMNS = ["reason_url_1", "reason_url_2", "reason_article", "reason_final"]

URL_BATCH_SIZE_SMALL = 10
URL_BATCH_SIZE_LARGE = 5
URL_TIMEOUT_SMALL = 180
URL_TIMEOUT_LARGE = 300
ARTICLE_TIMEOUT = 240
ARTICLE_MIN_LEN = 300
ARTICLE_CHAR_LIMIT = 9000

translator = Translator()


# =========================================================
# OLLAMA RUNNER
# =========================================================
def run_ollama(prompt: str, model: str, temperature: float = 0.2, max_retries: int = 2, timeout_sec: int = 180) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    for attempt in range(max_retries + 1):
        try:
            r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout_sec)
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


# =========================================================
# PROMPTS
# =========================================================
def build_url_prompt(url_items):
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
        rid = int(it["row_id"])
        url = str(it["url"])
        prompt += f"{rid}\t{url}\n"

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


# =========================================================
# URL BATCH CLASSIFICATION
# =========================================================
def run_batch_url_pressure(url_items, model: str, timeout_sec: int = 180, temperature: float = 0.2):
    prompt = build_url_prompt(url_items)
    out = run_ollama(prompt, model=model, timeout_sec=timeout_sec, temperature=temperature)

    if out and not out.startswith("[model_error]"):
        print("   -> first 300 chars:", out[:300].replace("\n", "\\n"))
    else:
        print(f"   -> {model} raw out: {out[:300] if out else '<EMPTY>'}")

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


def batch_process_urls(url_items, model: str, batch_size: int, timeout_sec: int = 180, sleep_between_batches: float = 0.2, temperature: float = 0.2):
    pressure_map = {}

    for start in range(0, len(url_items), batch_size):
        batch = url_items[start:start + batch_size]
        print(f"[{model}] URL batch {start}..{start + len(batch) - 1} (size={len(batch)})")
        batch_res = run_batch_url_pressure(
            batch,
            model=model,
            timeout_sec=timeout_sec,
            temperature=temperature
        )

        if not batch_res:
            print("   -> batch returned 0 results (timeout/error/empty).")
            time.sleep(sleep_between_batches)
            continue

        for r in batch_res:
            rid = r.get("row_id")
            if rid is None:
                continue
            pressure_map[int(rid)] = {
                "pressure": (r.get("pressure") or "unclear").strip().lower(),
                "reason": (r.get("reason") or "").strip()
            }

        time.sleep(sleep_between_batches)

    return pressure_map


# =========================================================
# ARTICLE EXTRACTION
# =========================================================
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


# =========================================================
# ARTICLE CLASSIFICATION
# =========================================================
def run_article_pressure(article_text: str, model: str, timeout_sec: int = ARTICLE_TIMEOUT, temperature: float = 0.2):
    prompt = build_article_prompt(article_text)
    out = run_ollama(prompt, model=model, timeout_sec=timeout_sec, temperature=temperature)

    if not out or out.startswith("[model_error]"):
        return {"pressure": "unclear", "reason": ""}

    out = out.strip()

    try:
        obj = json.loads(out)
        return {
            "pressure": (obj.get("pressure") or "unclear").strip().lower(),
            "reason": (obj.get("reason") or "").strip()
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
                "reason": (obj.get("reason") or "").strip()
            }
        except Exception:
            continue

    return {"pressure": "unclear", "reason": ""}


# =========================================================
# TRANSLATION
# =========================================================
def translate_if_needed(text):
    if pd.isna(text):
        return text

    text = unquote(str(text)).strip()
    if not text:
        return text

    try:
        detected = translator.detect(text).lang
        if detected != "en":
            translated = translator.translate(text, dest="en").text
            time.sleep(0.5)
            return translated
        return text
    except Exception:
        return text


def translate_reason_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in TRANSLATION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(translate_if_needed)
    return df


# =========================================================
# MAIN - MODE 3 ONLY
# =========================================================
def main():
    overall_start = time.perf_counter()

    df = pd.read_csv(CSV_IN)

    if URL_COL not in df.columns:
        raise ValueError(f"CSV missing expected column '{URL_COL}'. Found: {list(df.columns)}")

    total_rows = len(df)

    url_items = []
    skipped_no_url = 0
    for i, row in df.iterrows():
        url = str(row.get(URL_COL, "")).strip()
        if not url or url.lower() == "nan":
            skipped_no_url += 1
            continue
        url_items.append({"row_id": i, "url": url})

    print("MODE = 3")
    print(f"Total rows: {total_rows}")
    print(f"Rows with URL: {len(url_items)}")
    print(f"Skipped (no URL): {skipped_no_url}")

    # -------------------------------------------------
    # PASS 1: qwen7b over all URLs
    # -------------------------------------------------
    map_url_1 = batch_process_urls(
        url_items=url_items,
        model=QWEN_7B,
        batch_size=URL_BATCH_SIZE_SMALL,
        timeout_sec=URL_TIMEOUT_SMALL,
        temperature=GLOBAL_TEMP
    )

    df["pressure_url_1"] = [map_url_1.get(i, {"pressure": "unclear"})["pressure"] for i in range(total_rows)]
    df["reason_url_1"] = [map_url_1.get(i, {"reason": ""})["reason"] for i in range(total_rows)]
    df["model_url_1"] = QWEN_7B
    df["temp_url_1"] = GLOBAL_TEMP

    # -------------------------------------------------
    # PASS 2: qwen72b for unclear/neutral from pass 1
    # -------------------------------------------------
    unclear_after_url_1 = []
    for it in url_items:
        rid = int(it["row_id"])
        if df.at[rid, "pressure_url_1"] in RECHECK_LABELS:
            unclear_after_url_1.append(it)

    print(f"Unclear/neutral after URL pass 1: {len(unclear_after_url_1)}")

    map_url_2 = {}
    if unclear_after_url_1:
        map_url_2 = batch_process_urls(
            url_items=unclear_after_url_1,
            model=QWEN_72B,
            batch_size=URL_BATCH_SIZE_LARGE,
            timeout_sec=URL_TIMEOUT_LARGE,
            temperature=GLOBAL_TEMP
        )

    df["pressure_url_2"] = [map_url_2.get(i, {}).get("pressure", "") for i in range(total_rows)]
    df["reason_url_2"] = [map_url_2.get(i, {}).get("reason", "") for i in range(total_rows)]
    df["model_url_2"] = QWEN_72B
    df["temp_url_2"] = GLOBAL_TEMP

    # -------------------------------------------------
    # ARTICLE FALLBACK:
    # only if URL1 is unclear/neutral AND URL2 is blank or unclear/neutral
    # -------------------------------------------------
    article_candidates = []

    for it in url_items:
        rid = int(it["row_id"])
        p1 = df.at[rid, "pressure_url_1"]
        p2 = df.at[rid, "pressure_url_2"]

        needs_article = (p1 in RECHECK_LABELS and (not p2 or p2 in RECHECK_LABELS))
        if needs_article:
            article_candidates.append(it)

    print(f"Rows needing article fallback: {len(article_candidates)}")

    article_texts = {}
    article_results = {}

    for idx, it in enumerate(article_candidates, start=1):
        rid = int(it["row_id"])
        url = str(it["url"]).strip()

        print(f"[ARTICLE {idx}/{len(article_candidates)}] Fetching row_id={rid}: {url}")
        article_text = get_article_text(url)
        article_texts[rid] = article_text

        if not article_text or len(article_text) < ARTICLE_MIN_LEN:
            print("   -> extract failed/too short")
            article_results[rid] = {"pressure": "unclear", "reason": ""}
            continue

        result = run_article_pressure(
            article_text,
            model=QWEN_7B,
            timeout_sec=ARTICLE_TIMEOUT,
            temperature=GLOBAL_TEMP
        )
        article_results[rid] = result
        print(f"   -> article pressure={result['pressure']} reason={result['reason']}")
        time.sleep(0.5)

    df["article_text"] = [article_texts.get(i, "") for i in range(total_rows)]
    df["pressure_article"] = [article_results.get(i, {}).get("pressure", "") for i in range(total_rows)]
    df["reason_article"] = [article_results.get(i, {}).get("reason", "") for i in range(total_rows)]
    df["model_article"] = QWEN_7B
    df["temp_article"] = GLOBAL_TEMP

    # -------------------------------------------------
    # FINAL DECISION - MODE 3
    # -------------------------------------------------
    final_pressure = []
    final_reason = []
    final_source = []

    for i in range(total_rows):
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

    if TRANSLATE_REASONS:
        df = translate_reason_columns(df)

    df.to_csv(CSV_OUT, index=False)
    print(f"\nSaved output to {CSV_OUT}")

    # -------------------------------------------------
    # MANUAL EVAL FILE
    # -------------------------------------------------
    manual_eval = df[
        (
            df["pressure_url_1"].isin(["neutral", "unclear"])
        ) &
        (
            (df["pressure_url_2"].isin(["push", "pull", "neutral"])) |
            (df["pressure_article"].isin(["push", "pull", "neutral"]))
        )
    ].copy()

    manual_cols = [
        "sourceurl",
        "pressure_url_1", "reason_url_1",
        "pressure_url_2", "reason_url_2",
        "pressure_article", "reason_article",
        "pressure_final", "reason_final",
        "final_source", "mode_used"
    ]

    manual_cols = [c for c in manual_cols if c in manual_eval.columns]
    manual_eval[manual_cols].head(25).to_csv(MANUAL_EVAL_OUT, index=False)
    print(f"Saved manual eval to {MANUAL_EVAL_OUT}")

    overall_elapsed = time.perf_counter() - overall_start
    mins, secs = divmod(overall_elapsed, 60)
    hours, mins = divmod(mins, 60)
    print(f"\nTotal runtime: {int(hours):02d}:{int(mins):02d}:{secs:05.2f}")


if __name__ == "__main__":
    main()