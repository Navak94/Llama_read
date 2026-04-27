import time
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from ddgs import DDGS
from urllib.parse import urlparse

# =========================
# CONFIG
# =========================

def generate_months(start_year, end_year):
    months = []
    for year in range(start_year, end_year + 1):
        for m in range(1, 13):
            months.append(f"{year}-{m:02d}")
    return months

MONTH_RANGE = [
    "2021-11",
    "2021-12",
    "2022-01",
    "2022-02",
    "2022-03",
    "2022-04",
    "2022-05",
    "2022-06",
    "2022-07",
    "2022-08",
    "2022-09",
    "2022-10",
    "2022-11",
]

TOPIC_JOBS = [
    {
        "label": "ukraine",
        "region_terms": ["Ukraine", "Russia"],
        "event_terms": ["war", "conflict", "invasion", "refugees"],
        "origin": "Ukraine",
        "destination": "Poland",
        "output_csv": "ukraine.csv",
        "months": MONTH_RANGE,
    },
    {
        "label": "mexico",
        "region_terms": ["Mexico", "United States", "US border"],
        "event_terms": ["migration", "immigration", "asylum", "migrants"],
        "origin": "Mexico",
        "destination": "United States",
        "output_csv": "mexico.csv",
        "months": MONTH_RANGE,
    },
    {
        "label": "syria",
        "region_terms": ["Syria", "Turkey", "Syrian"],
        "event_terms": ["civil war", "conflict", "refugees", "displacement"],
        "origin": "Syria",
        "destination": "Turkey",
        "output_csv": "syria.csv",
        "months": MONTH_RANGE,
    },
]

SOURCES = {
    "bbc": "bbc.com",
    "npr": "npr.org",
    "pbs": "pbs.org",
    "nbc": "nbcnews.com",
    "cbs": "cbsnews.com",
}

MAX_RESULTS_PER_SOURCE_MONTH = 80
MAX_ARTICLE_CHARS = 12000
SLEEP_BETWEEN_SEARCHES = 1.0
SLEEP_BETWEEN_ARTICLES = 0.75
REQUEST_TIMEOUT = 20
TEXT_ONLY_MODE = False


# =========================
# HELPERS
# =========================

def domain_matches(url, domain):
    try:
        netloc = urlparse(url).netloc.lower()
        return domain.lower() in netloc
    except Exception:
        return False


def clean_text_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def safe_search_ddg(query, max_results=20):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"DDG search failed for query [{query}]: {e}")
        return []


def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        if len(text) > 300:
            return text[:MAX_ARTICLE_CHARS]
    except Exception as e:
        print(f"newspaper3k failed for {url}: {e}")

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text_parts = []

        for p in paragraphs:
            txt = p.get_text(" ", strip=True)
            txt = clean_text_whitespace(txt)
            if len(txt) > 50:
                text_parts.append(txt)

        text = "\n\n".join(text_parts).strip()
        if len(text) > 300:
            return text[:MAX_ARTICLE_CHARS]

        return ""
    except Exception as e:
        print(f"BeautifulSoup fallback failed for {url}: {e}")
        return ""


def build_search_queries(domain, month_str, region_terms, event_terms):
    year, month = month_str.split("-")

    month_names = {
        "01": "January", "02": "February", "03": "March",
        "04": "April",   "05": "May",      "06": "June",
        "07": "July",    "08": "August",   "09": "September",
        "10": "October", "11": "November", "12": "December"
    }

    month_name = month_names[month]

    region_main = region_terms[0] if region_terms else ""
    region_pair = " ".join(region_terms[:2]) if len(region_terms) >= 2 else region_main

    event_main = event_terms[0] if event_terms else ""
    event_pair = " ".join(event_terms[:2]) if len(event_terms) >= 2 else event_main

    queries = [
        f"site:{domain} {region_pair} {event_pair} {month_name} {year}",
        f"site:{domain} {region_pair} {event_main} {year}",
        f"site:{domain} {region_main} {event_main} {month_name} {year}",
        f"site:{domain} {region_main} news {event_main} {year}",
    ]

    return queries


def search_source_for_month(source_name, domain, month_str, label, region_terms, event_terms, origin, destination):
    queries = build_search_queries(domain, month_str, region_terms, event_terms)

    seen_urls = set()
    collected = []

    print(f"\nSearching {source_name} for {label} | {month_str}")

    for query in queries:
        print(f"  Query: {query}")
        results = safe_search_ddg(query, max_results=MAX_RESULTS_PER_SOURCE_MONTH)

        for r in results:
            url = (r.get("href") or r.get("url") or "").strip()
            title = (r.get("title") or "").strip()
            snippet = (r.get("body") or r.get("snippet") or "").strip()

            if not url.startswith("http"):
                continue
            if not domain_matches(url, domain):
                continue
            if url in seen_urls:
                continue

            seen_urls.add(url)

            collected.append({
                "topic_label": label,
                "origin": origin,
                "destination": destination,
                "month": month_str,
                "source_name": source_name,
                "domain": domain,
                "title": title,
                "url": url,
                "snippet": snippet,
                "search_query": query,
            })

        time.sleep(SLEEP_BETWEEN_SEARCHES)

    print(f"  Collected {len(collected)} candidate URLs for {source_name} {month_str}")
    return collected


def run_topic_job(job):
    topic_label = job["label"]
    region_terms = job["region_terms"]
    event_terms = job["event_terms"]
    origin = job["origin"]
    destination = job["destination"]
    output_csv = job["output_csv"]
    months = job["months"]

    all_rows = []

    print("\n" + "#" * 100)
    print(f"STARTING TOPIC: {topic_label}")
    print(f"OUTPUT FILE: {output_csv}")
    print("#" * 100)

    for month in months:
        print("\n" + "=" * 80)
        print(f"TOPIC: {topic_label} | MONTH: {month}")

        for source_name, domain in SOURCES.items():
            candidates = search_source_for_month(
                source_name=source_name,
                domain=domain,
                month_str=month,
                label=topic_label,
                region_terms=region_terms,
                event_terms=event_terms,
                origin=origin,
                destination=destination
            )

            for i, article_row in enumerate(candidates, start=1):
                url = article_row["url"]
                title = article_row["title"]
                snippet = article_row["snippet"]

                print(f"[{topic_label} | {month} | {source_name} | {i}/{len(candidates)}] {title}")

                article_text = ""
                if not TEXT_ONLY_MODE and url.startswith("http"):
                    article_text = get_article_text(url)

                all_rows.append({
                    "topic_label": topic_label,
                    "origin": origin,
                    "destination": destination,
                    "month": month,
                    "source_name": source_name,
                    "domain": domain,
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "search_query": article_row["search_query"],
                    "article_text": article_text,
                    "article_text_length": len(article_text),
                })

                time.sleep(SLEEP_BETWEEN_ARTICLES)

    df = pd.DataFrame(all_rows)

    if df.empty:
        print(f"\nNo articles found for topic: {topic_label}")
        df.to_csv(output_csv, index=False, encoding="utf-8")
        return

    before = len(df)
    df = df.drop_duplicates(subset=["url"]).copy()
    df = df.reset_index(drop=True)
    after = len(df)

    print(f"\n[{topic_label}] Rows before dedupe: {before}")
    print(f"[{topic_label}] Rows after dedupe:  {after}")

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {output_csv}")


def main():
    for job in TOPIC_JOBS:
        run_topic_job(job)


if __name__ == "__main__":
    main()