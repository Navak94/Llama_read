import time
import subprocess
import requests
import pandas as pd
import json
from newspaper import Article
from bs4 import BeautifulSoup

CSV_IN = "gdelt_events_basic.csv"
CSV_OUT = "ALL_AT_ONCE_TEST_Qwen_7B_article_and_URL.csv"
URL_COL = "sourceurl"

# ---- 1) Pull article text ----
def get_article_text_newspaper(url: str, char_limit: int = 9000) -> str:
    """Best effort: newspaper3k extraction."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = (article.text or "").strip()
        return text[:char_limit]
    except Exception:
        return ""

def get_article_text_fallback(url: str, char_limit: int = 9000) -> str:
    """Fallback: requests + BeautifulSoup."""
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

def get_article_text(url: str, char_limit: int = 9000) -> str:
    text = get_article_text_newspaper(url, char_limit=char_limit)
    if len(text) >= 300:
        return text
    return get_article_text_fallback(url, char_limit=char_limit)

# ---- 2) Run qwen locally via Ollama ----
def run_qwen(prompt: str, model: str = "qwen2.5:7b", max_retries: int = 2) -> str:
    """
    Calls: ollama run qwen2.5:7b
    Returns stdout text (best effort).
    """
    for attempt in range(max_retries + 1):
        try:
            process = subprocess.Popen(
                ["bash", "-c", f"ollama run {model}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(prompt, timeout=180)

            if process.returncode == 0 and stdout.strip():
                return stdout.strip()

            msg = (stderr or "").strip()
            if msg:
                return f"[qwen_error] {msg}"

        except subprocess.TimeoutExpired:
            return "[qwen_error] timeout"
        except Exception as e:
            if attempt == max_retries:
                return f"[qwen_error] {e}"
            time.sleep(2 ** attempt)

    return "[qwen_error] unknown"

# ---- 3) Batch URL sentiment ----
def run_qwen_batch_url_sentiment(url_items, model="qwen2.5:7b"):
    """
    url_items: list of dicts: [{"row_id": int, "url": str}, ...]
    Returns list of dicts: [{"row_id": int, "sentiment": str, "reason": str}, ...]
    """
    prompt = (
        "You are doing URL-only sentiment triage.\n"
        "For each item, infer sentiment from ONLY the URL text.\n"
        "If not enough info, use 'unclear'.\n"
        "Allowed sentiment: positive|negative|neutral|mixed|unclear\n\n"
        "Return STRICT JSONL (one JSON object per line) with keys:\n"
        "row_id (int), sentiment (string), reason (string <= 15 words).\n"
        "No extra text.\n\n"
        "ITEMS:\n"
    )

    for it in url_items:
        rid = int(it["row_id"])
        url = str(it["url"])
        prompt += f"{rid}\t{url}\n"

    out = run_qwen(prompt, model=model)

    results = []
    if not out or out.startswith("[qwen_error]"):
        return results

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "row_id" in obj and "sentiment" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    return results

# ---- 4) Main ----
def main():
    df = pd.read_csv(CSV_IN)

    # ✅ CHECK COLUMN FIRST
    if URL_COL not in df.columns:
        raise ValueError(f"CSV missing expected column '{URL_COL}'. Found: {list(df.columns)}")

    # ============================
    # ✅ A) BATCH URL SENTIMENT HERE
    # ============================
    url_items = []
    for i, row in df.iterrows():
        url = str(row.get(URL_COL, "")).strip()
        if url and url.lower() != "nan":
            url_items.append({"row_id": i, "url": url})

    BATCH_SIZE = 80
    sentiment_map = {}

    for start in range(0, len(url_items), BATCH_SIZE):
        batch = url_items[start:start + BATCH_SIZE]
        print(f"Processing URL batch {start} to {start + len(batch) - 1}")
        batch_res = run_qwen_batch_url_sentiment(batch)

        for r in batch_res:
            rid = r.get("row_id")
            if rid is not None:
                sentiment_map[int(rid)] = {
                    "sentiment": r.get("sentiment", "unclear"),
                    "reason": r.get("reason", "")
                }

    print("Finished URL sentiment batching.")
    # ============================

    # ============================
    # ✅ B) NOW DO YOUR ARTICLE LOOP
    # ============================
    total_rows = len(df)
    urls_present = 0
    extracted_ok = 0
    qwen_ok = 0
    skipped_no_url = 0
    failed_extract = 0
    failed_qwen = 0

    processed_rows = []
    run_start = time.perf_counter()

    for i, row in df.iterrows():
        url = str(row.get(URL_COL, "")).strip()
        if not url or url.lower() == "nan":
            skipped_no_url += 1
            continue

        urls_present += 1
        row_start = time.perf_counter()
        print(f"[{i+1}/{total_rows}] Fetching: {url}")

        article_text = get_article_text(url)

        if not article_text or len(article_text) < 300:
            failed_extract += 1
            print("   -> extract failed/too short")
            continue

        extracted_ok += 1

        prompt = (
            "You are helping with event/news triage.\n"
            "Task:\n"
            "1) Give a 3-5 sentence summary.\n"
            "2) List 3-8 key entities (people/orgs/places) as bullet points.\n"
            "3) Give 1 short 'why it matters' sentence.\n\n"
            "ARTICLE:\n"
            f"{article_text}\n"
        )

        qwen_out = run_qwen(prompt)

        if not qwen_out or qwen_out.startswith("[qwen_error]"):
            failed_qwen += 1
            print(f"   -> qwen failed: {qwen_out[:120]}")
            continue

        qwen_ok += 1

        # ✅ LOOK UP PRE-BATCHED URL SENTIMENT
        s = sentiment_map.get(i, {"sentiment": "unclear", "reason": ""})
        qwen_url_out = f"Sentiment: {s['sentiment']}\nReason: {s['reason']}"

        processed_rows.append({
            **row.to_dict(),
            "article_text": article_text,
            "qwen_full_Fed_output": qwen_out,
            "qwen_just_URL": qwen_url_out
        })

        row_end = time.perf_counter()
        print(f"   -> OK (#{qwen_ok}) row_time={row_end - row_start:.2f}s")

        time.sleep(1.0)

    run_end = time.perf_counter()
    elapsed = run_end - run_start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)

    out_df = pd.DataFrame(processed_rows)
    out_df.to_csv(CSV_OUT, index=False)

    print("\n==== RUN SUMMARY ====")
    print(f"Total rows in input:      {total_rows}")
    print(f"Rows with URL present:    {urls_present}")
    print(f"Extracted text OK:        {extracted_ok}")
    print(f"qwen processed OK:        {qwen_ok}")
    print(f"Skipped (no URL):         {skipped_no_url}")
    print(f"Failed extraction:        {failed_extract}")
    print(f"Failed qwen:              {failed_qwen}")
    print(f"Output rows written:      {len(out_df)}  (should equal qwen OK)")
    print(f"Total runtime:            {int(h):02d}:{int(m):02d}:{s:05.2f}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    print(f"\nTotal runtime: {int(hours):02d}:{int(mins):02d}:{secs:05.2f} (hh:mm:ss)")
