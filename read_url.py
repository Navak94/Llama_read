import time
import subprocess
import requests
import pandas as pd
from newspaper import Article
from bs4 import BeautifulSoup
import time


CSV_IN = "gdelt_events_basic.csv"
CSV_OUT = "gdelt_events_with_qwen.csv"
URL_COL = "sourceurl"

# ---- 1) Pull article text (reuse your approach) ----
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
    """Fallback: requests + BeautifulSoup (less clean but sometimes works when newspaper fails)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove obvious junk
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # crude cleanup
        text = " ".join(text.split())
        return text[:char_limit]
    except Exception:
        return ""

def get_article_text(url: str, char_limit: int = 9000) -> str:
    text = get_article_text_newspaper(url, char_limit=char_limit)
    if len(text) >= 300:  # "good enough"
        return text
    return get_article_text_fallback(url, char_limit=char_limit)

# ---- 2) Run qwen locally via Ollama (reuse qwen_read.py pattern) ----
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

            # Sometimes ollama errors land in stderr
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

# ---- 3) Main loop: CSV -> fetch -> qwen -> write ----
def main():
    df = pd.read_csv(CSV_IN)

    if URL_COL not in df.columns:
        raise ValueError(f"CSV missing expected column '{URL_COL}'. Found: {list(df.columns)}")

    total_rows = len(df)
    urls_present = 0
    extracted_ok = 0
    qwen_ok = 0
    skipped_no_url = 0
    failed_extract = 0
    failed_qwen = 0

    # Only store SUCCESSFUL rows here
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

        # slightly different prompt but since the task is different, doesnt make sense for it to be the same for reading urls
        URL_prompt = (
            "You are helping with event/news triage.\n"
            "Given ONLY this URL string, infer sentiment if possible.\n"
            "If there isn't enough info in the URL, say 'unclear'.\n"
            "Return:\n"
            "- Sentiment: positive|negative|neutral|mixed|unclear\n"
            "- 1 short reason (max 15 words)\n\n"
            f"URL:\n{url}\n"
        )


        qwen_out = run_qwen(prompt)
        qwen_url_out = run_qwen(URL_prompt)


        if not qwen_out or qwen_out.startswith("[qwen_error]"):
            failed_qwen += 1
            print(f"   -> qwen failed: {qwen_out[:120]}")
            continue

        qwen_ok += 1  # ✅ count successful article Qwen runs

        if (not qwen_url_out) or qwen_url_out.startswith("[qwen_error]"):
            qwen_url_out = ""   # best-effort, don't fail the row

        # ONLY append successful ones
        processed_rows.append({
            **row.to_dict(),
            "article_text": article_text,
            "qwen_full_Fed_output": qwen_out,
            "qwen_just_URL":qwen_url_out
        })

        row_end = time.perf_counter()
        print(f"   -> OK (#{qwen_ok}) row_time={row_end - row_start:.2f}s")

        time.sleep(1.0)

    run_end = time.perf_counter()
    elapsed = run_end - run_start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)

    # Write ONLY successful processed rows
    out_df = pd.DataFrame(processed_rows)
    out_df.to_csv(CSV_OUT, index=False)

    print("\n==== RUN SUMMARY ====")
    print(f"Total rows in input:      {total_rows}")
    print(f"Rows with URL present:    {urls_present}")
    print(f"Extracted text OK:        {extracted_ok}")
    print(f"qwen processed OK:       {qwen_ok}")
    print(f"Skipped (no URL):         {skipped_no_url}")
    print(f"Failed extraction:        {failed_extract}")
    print(f"Failed qwen:             {failed_qwen}")
    print(f"Output rows written:      {len(out_df)}  (should equal qwen OK)")
    print(f"Total runtime:            {int(h):02d}:{int(m):02d}:{s:05.2f}")

if __name__ == "__main__":
    start_time = time.perf_counter()

    main()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)

    print(
        f"\nTotal runtime: "
        f"{int(hours):02d}:{int(mins):02d}:{secs:05.2f} "
        f"(hh:mm:ss)"
    )

