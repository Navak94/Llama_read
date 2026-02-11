"""
Basic GDELT pull (events) + simple extraction.

Fixes the common gotcha:
- If you set output="json", gdelt.Search returns a STRING (raw json),
  and normcols=True can crash because strings have no ".columns".
- Safest beginner approach: do NOT set output -> returns a pandas DataFrame.

Requirements:
  pip install gdelt pandas pyarrow
"""

import pandas as pd
import gdelt


def main():
    # 1) Initialize client (GDELT v2)
    gd = gdelt.gdelt(version=2)

    # 2) Query a SMALL date range first (GDELT can get huge fast)
    start_date = "2022 Jan 01"
    end_date = "2022 Jan 02"

    results = gd.Search(
        date=[start_date, end_date],
        table="events",       # events / mentions / gkg
        coverage=False,       # True can explode size (15-min intervals)
        translation=True,
        normcols=True         # normalize column names (only works if results is a DataFrame)
        # NOTE: do NOT pass output="json" here for beginners
    )

    # 3) Convert results to a DataFrame (handles either return type safely)
    if isinstance(results, pd.DataFrame):
        df = results
    elif isinstance(results, str):
        # If someone later sets output="json", this still works:
        df = pd.read_json(results)
        # If you want normalized columns when df came from JSON, do it manually:
        df.columns = [c.replace("_", "").lower() for c in df.columns]
    else:
        raise TypeError(f"Unexpected results type from gdelt.Search: {type(results)}")

    print(f"Pulled {len(df):,} rows")
    print("Columns:", df.columns.tolist())

    # 4) Extract a smaller set of useful columns (select safely)
    desired_cols = [
        "sqldate",
        "eventcode",
        "actor1name",
        "actor2name",
        "actiongeo_fullname",
        "actiongeo_countrycode",
        "sourceurl",
    ]
    keep_cols = [c for c in desired_cols if c in df.columns]
    df_small = df[keep_cols].copy()

    print("\nPreview:")
    print(df_small.head(10))

    # 5) Example extraction: top event codes (if present)
    if "eventcode" in df_small.columns:
        print("\nTop EventCodes:")
        print(df_small["eventcode"].value_counts().head(10))

    # 6) Save output
    out_parquet = "gdelt_events_basic.parquet"
    try:
        df_small.to_parquet(out_parquet, index=False)
        print(f"\nSaved: {out_parquet}")
    except Exception as e:
        # If pyarrow isn't installed or parquet fails, fall back to CSV
        out_csv = "gdelt_events_basic.csv"
        df_small.to_csv(out_csv, index=False)
        print(f"\nParquet save failed ({e}). Saved CSV instead: {out_csv}")


if __name__ == "__main__":
    main()