import time
import subprocess
import pandas as pd
from urllib.parse import urlparse, unquote
import re

CSV_IN = "gdelt_events_basic.csv"
CSV_OUT = "Qwen_72B_FILTERED_URL_ONLY.csv"
URL_COL = "sourceurl"

# --- NEW: URL -> clean text (no clicking, no fetching) ---
def url_to_clean_text(url: str, max_tokens: int = 40) -> str:
    u = urlparse(url)

    # Only use path (ignore domain)
    path = unquote(u.path or "")

    tokens = re.split(r"[\/\-\_\.\s]+", path)

    tokens = [
        t.lower()
        for t in tokens
        if len(t) > 2 and not t.isdigit()
    ]

    return " ".join(tokens[:max_tokens])

# ---- 1) Run qwen locally via Ollama ----
def run_qwen(prompt: str, model: str = "qwen2.5:72b", max_retries: int = 2) -> str:
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

# ---- 2) Main loop: CSV -> URL -> qwen(URL sentiment) -> write ----
def main():
    df = pd.read_csv(CSV_IN)

    if URL_COL not in df.columns:
        raise ValueError(f"CSV missing expected column '{URL_COL}'. Found: {list(df.columns)}")

    total_rows = len(df)
    urls_present = 0
    qwen_ok = 0
    skipped_no_url = 0
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

        # --- NEW: derive readable URL text ---
        url_text = url_to_clean_text(url)

        #print(f"[{i+1}/{total_rows}] URL sentiment: {url_text}")
        print("unfiltered url is ", url)
        print("URL TEXT IS  ------------------  " ,url_text, "------------------")
        # --- CHANGED: prompt uses URL_TEXT instead of raw URL ---
        URL_prompt = (
            "You are helping with event/news triage.\n"
            "Given ONLY this URL-derived text, infer sentiment if possible.\n"
            "If there isn't enough info, say 'unclear'.\n"
            "Return:\n"
            "- Sentiment: positive|negative|neutral|mixed|unclear\n"
            "- 1 short reason (max 15 words)\n\n"
            f"URL_TEXT:\n{url_text}\n"
        )

        qwen_url_out = run_qwen(URL_prompt)

        if (not qwen_url_out) or qwen_url_out.startswith("[qwen_error]"):
            failed_qwen += 1
            qwen_url_out = ""
        else:
            qwen_ok += 1

        processed_rows.append({
            **row.to_dict(),
            "url_text": url_text,          # optional but useful for debugging
            "qwen_just_URL": qwen_url_out
        })

        row_end = time.perf_counter()
        print(f"   -> done (ok={qwen_ok}, failed={failed_qwen}) row_time={row_end - row_start:.2f}s")
        time.sleep(1.0)

    out_df = pd.DataFrame(processed_rows)
    out_df.to_csv(CSV_OUT, index=False)

    elapsed = time.perf_counter() - run_start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)

    print("\n==== RUN SUMMARY ====")
    print(f"Total rows in input:      {total_rows}")
    print(f"Rows with URL present:    {urls_present}")
    print(f"qwen URL processed OK:    {qwen_ok}")
    print(f"Failed qwen (URL):        {failed_qwen}")
    print(f"Skipped (no URL):         {skipped_no_url}")
    print(f"Output rows written:      {len(out_df)}  (includes URL failures as blank output)")
    print(f"Total runtime:            {int(h):02d}:{int(m):02d}:{s:05.2f}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    print(f"\nTotal runtime: {int(hours):02d}:{int(mins):02d}:{secs:05.2f} (hh:mm:ss)")
