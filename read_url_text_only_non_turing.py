import time
import subprocess
import pandas as pd
import json

CSV_IN = "gdelt_events_basic.csv"
CSV_OUT = "TEST_Qwen_URL_ONLY_7B_then_72B.csv"
URL_COL = "sourceurl"

# -----------------------------
# Ollama runner (LOCAL)
# -----------------------------
def run_qwen(prompt: str, model: str, max_retries: int = 2, timeout_sec: int = 180) -> str:
    """
    Local desktop: calls `ollama run <model>` and feeds prompt via stdin.
    No -p flag, no separate server shell needed.
    """
    cmd = ["ollama", "run", model]

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

            if process.returncode == 0 and stdout and stdout.strip():
                return stdout.strip()

            msg = (stderr or "").strip()
            if msg:
                return f"[qwen_error] {msg}"
            return "[qwen_error] empty_output"

        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass
            return "[qwen_error] timeout"
        except FileNotFoundError:
            return "[qwen_error] ollama_not_found (is Ollama installed / in PATH?)"
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
    url_items: [{"row_id": int, "url": str}, ...]
    Returns:   [{"row_id": int, "sentiment": str, "reason": str}, ...]
    """
    prompt = (
        "You are a URL-only sentiment classifier.\n\n"
        "Task:\n"
        "Classify sentiment using ONLY lexical cues found directly in the URL string.\n"
        "Do NOT assume context outside the URL.\n"
        "Do NOT invent details.\n\n"
        "Labels: positive|negative|neutral|mixed|unclear\n\n"
        "Output Format (STRICT): JSONL ONLY.\n"
        "One JSON object per line, for every input item.\n"
        "Each JSON object must have keys:\n"
        "row_id (int), sentiment (string), reason (string, <= 12 words).\n"
        "No extra text, no markdown, no blank lines.\n\n"
        "ITEMS (tab-separated):\n"
        "<row_id>\\t<url>\n"
    )


    for it in url_items:
        rid = int(it["row_id"])
        url = str(it["url"])
        prompt += f"{rid}\t{url}\n"

    out = run_qwen(prompt, model=model, timeout_sec=timeout_sec)

    if (not out) or out.startswith("[qwen_error]"):
        print(f"   -> {model} raw out: {out[:300] if out else '<EMPTY>'}")

    if out.startswith("[qwen_error]"):
        print(f"   -> {model} error: {out}")
        return []

    results = []
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


def batch_process(url_items, model: str, batch_size: int, timeout_sec: int = 180, sleep_between_batches: float = 0.2):
    """
    Returns sentiment_map: row_id -> {"sentiment":..., "reason":...}
    """
    sentiment_map = {}

    for start in range(0, len(url_items), batch_size):
        batch = url_items[start:start + batch_size]
        print(f"[{model}] batch {start}..{start + len(batch) - 1} (size={len(batch)})")

        batch_res = run_qwen_batch_url_sentiment(batch, model=model, timeout_sec=timeout_sec)

        if not batch_res:
            print("   -> batch returned 0 results (timeout/error/empty).")
            time.sleep(sleep_between_batches)
            continue

        # Optional: quick progress sanity check
        # print(f"   -> received {len(batch_res)} results")

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

    # PASS 1: 7B over ALL URLs
    map_7b = batch_process(
        url_items=url_items,
        model="qwen2.5:7b",
        batch_size=10,      # keep small to avoid context overflow
        timeout_sec=180
    )

    df["sentiment_7b"] = [map_7b.get(i, {"sentiment": "unclear"})["sentiment"] for i in range(total_rows)]
    df["reason_7b"]    = [map_7b.get(i, {"reason": ""})["reason"] for i in range(total_rows)]

    unclear_items = []
    for it in url_items:
        rid = int(it["row_id"])
        if df.at[rid, "sentiment_7b"] == "unclear":
            unclear_items.append(it)

    print(f"Unclear after 7B: {len(unclear_items)}")

    # PASS 2: 72B ONLY for unclear
    map_72b = {}
    if unclear_items:
        map_72b = batch_process(
            url_items=unclear_items,
            model="qwen2.5:72b",
            batch_size=5,     # 72B -> smaller batches
            timeout_sec=300   # give it longer locally
        )

    sentiment_72b = []
    reason_72b = []
    sentiment_final = []
    reason_final = []

    for i in range(total_rows):
        s7 = df.at[i, "sentiment_7b"]
        r7 = df.at[i, "reason_7b"]

        s72 = map_72b.get(i, {}).get("sentiment", "")
        r72 = map_72b.get(i, {}).get("reason", "")

        sentiment_72b.append(s72)
        reason_72b.append(r72)

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

    print(f"\nSaved output to {CSV_OUT}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    print(f"\nTotal runtime: {int(hours):02d}:{int(mins):02d}:{secs:05.2f} (hh:mm:ss)")
