import time
import subprocess
import pandas as pd
import json

CSV_IN = "gdelt_events_basic.csv"
CSV_OUT = "TEST_Qwen_URL_ONLY_7B_then_72B.csv"
URL_COL = "sourceurl"

# -----------------------------
# Ollama runner
# -----------------------------
SIF = "/home/nthindman/ollama_latest.sif"
BIND = "/home/nthindman:/home/nthindman"

def run_qwen(prompt: str, model: str, max_retries: int = 2, timeout_sec: int = 180) -> str:
    """
    Runs ollama through apptainer.
    Returns stdout text (best effort).
    """
    cmd = [
        "apptainer", "exec",
        "--userns",
        "--bind", BIND,
        SIF,
        "ollama", "run", model
    ]
    # If you truly need GPU passthrough add "--nv" right after "--userns".
    # On login nodes, --nv may or may not work.

    for attempt in range(max_retries + 1):
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(prompt, timeout=timeout_sec)

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



# -----------------------------
# Batch sentiment (JSONL in/out)
# -----------------------------
def run_qwen_batch_url_sentiment(url_items, model: str, timeout_sec: int = 180):
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

    out = run_qwen(prompt, model=model, timeout_sec=timeout_sec)


    if out.startswith("[qwen_error]"):
        print(f"   -> {model} error: {out}")
        return []

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


def batch_process(url_items, model: str, batch_size: int, timeout_sec: int = 180, sleep_between_batches: float = 0.5):
    """
    Returns sentiment_map: row_id -> {"sentiment":..., "reason":...}
    """
    sentiment_map = {}

    for start in range(0, len(url_items), batch_size):
        batch = url_items[start:start + batch_size]
        print(f"[{model}] batch {start}..{start + len(batch) - 1} (size={len(batch)})")

        batch_res = run_qwen_batch_url_sentiment(batch, model=model, timeout_sec=timeout_sec)

        # If the whole batch failed, just skip; missing rows will be treated as unclear
        if not batch_res:
            print(f"   -> batch returned 0 results (timeout/error/empty).")
            time.sleep(sleep_between_batches)
            continue

        for r in batch_res:
            rid = r.get("row_id")
            if rid is None:
                continue
            sentiment_map[int(rid)] = {
                "sentiment": (r.get("sentiment") or "unclear").strip().lower(),
                "reason": (r.get("reason") or "").strip()
            }

        time.sleep(sleep_between_batches)

    return sentiment_map


# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(CSV_IN)

    if URL_COL not in df.columns:
        raise ValueError(f"CSV missing expected column '{URL_COL}'. Found: {list(df.columns)}")

    total_rows = len(df)

    # Build url_items for rows that have URLs
    url_items = []
    skipped_no_url = 0
    for i, row in df.iterrows():
        url = str(row.get(URL_COL, "")).strip()
        if not url or url.lower() == "nan":
            skipped_no_url += 1
            continue
        url_items.append({"row_id": i, "url": url})

    print(f"Total rows: {total_rows}")
    print(f"Rows with URL: {len(url_items)}")
    print(f"Skipped (no URL): {skipped_no_url}")

    run_start = time.perf_counter()

    # -----------------------------
    # PASS 1: 7B over ALL URLs
    # -----------------------------
    map_7b = batch_process(
        url_items=url_items,
        model="qwen2.5:7b",
        batch_size=120,        # 7B can usually handle bigger batches
        timeout_sec=180
    )

    # Attach 7B results (default to unclear if missing)
    sentiment_7b = []
    reason_7b = []
    for i in range(total_rows):
        s = map_7b.get(i, {"sentiment": "unclear", "reason": ""})
        sentiment_7b.append(s["sentiment"])
        reason_7b.append(s["reason"])

    df["sentiment_7b"] = sentiment_7b
    df["reason_7b"] = reason_7b

    # Find unclear rows that also have URLs
    unclear_items = []
    for it in url_items:
        rid = int(it["row_id"])
        if df.at[rid, "sentiment_7b"] == "unclear":
            unclear_items.append(it)

    print(f"Unclear after 7B: {len(unclear_items)}")

    # -----------------------------
    # PASS 2: 72B ONLY for unclear
    # -----------------------------
    map_72b = {}
    if unclear_items:
        map_72b = batch_process(
            url_items=unclear_items,
            model="qwen2.5:72b",
            batch_size=40,      # 72B tends to need smaller batches
            timeout_sec=240     # you may need more time for 72B
        )

    # Build final (overwrite unclear if 72B returned something not-unclear)
    sentiment_final = []
    reason_final = []
    sentiment_72b = []
    reason_72b = []

    for i in range(total_rows):
        s7 = df.at[i, "sentiment_7b"] if "sentiment_7b" in df.columns else "unclear"
        r7 = df.at[i, "reason_7b"] if "reason_7b" in df.columns else ""

        s72 = map_72b.get(i, {}).get("sentiment", "")
        r72 = map_72b.get(i, {}).get("reason", "")

        sentiment_72b.append(s72)
        reason_72b.append(r72)

        # overwrite rule:
        # If 7B is unclear AND 72B provided a label (even if it's still unclear),
        # use 72B's output. If you ONLY want overwrite when 72B is NOT unclear,
        # change the condition to: (s72 and s72 != "unclear")
        if s7 == "unclear" and s72:
            sentiment_final.append(s72)
            reason_final.append(r72)
        else:
            sentiment_final.append(s7)
            reason_final.append(r7)

    df["sentiment_72b"] = sentiment_72b
    df["reason_72b"] = reason_72b
    df["sentiment_final"] = sentiment_final
    df["reason_final"] = reason_final

    df.to_csv(CSV_OUT, index=False)

    elapsed = time.perf_counter() - run_start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)

    print("\n==== RUN SUMMARY ====")
    print(f"Total rows in input:      {total_rows}")
    print(f"Rows with URL present:    {len(url_items)}")
    print(f"Unclear after 7B:         {len(unclear_items)}")
    print(f"72B attempted rows:       {len(unclear_items)}")
    print(f"72B returned rows:        {len(map_72b)}")
    print(f"Output written:           {CSV_OUT}")
    print(f"Total runtime:            {int(h):02d}:{int(m):02d}:{s:05.2f}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    print(f"\nTotal runtime: {int(hours):02d}:{int(mins):02d}:{secs:05.2f} (hh:mm:ss)")
