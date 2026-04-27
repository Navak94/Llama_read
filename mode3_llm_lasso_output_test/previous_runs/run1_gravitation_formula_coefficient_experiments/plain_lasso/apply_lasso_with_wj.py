
import argparse
import math
from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# FEATURE NAME MAP
# =========================================================
FEATURE_MAP = {
    "num__gdp_pc_diff": "gdp_pc_diff",
    "num__hdi_diff": "hdi_diff",
    "num__gpi_diff": "gpi_diff",
    "num__push_count": "push_count",
    "num__pull_count": "pull_count",
    "num__unclear_count": "unclear_count",
    "num__mixed_count": "mixed_count",
    "num__n_articles": "n_articles",
    "num__log_origin_gdp_pc": "log_origin_gdp_pc",
    "num__log_dest_gdp_pc": "log_dest_gdp_pc",
    "num__log_distance_km": "log_distance_km",
    "num__dest_hdi": "dest_hdi",
    "num__origin_hdi": "origin_hdi",
    "num__dest_gpi": "dest_gpi",
    "num__origin_gpi": "origin_gpi",
    "num__dest_life_expectancy": "dest_life_expectancy",
    "num__origin_life_expectancy": "origin_life_expectancy",
    "num__dest_expected_schooling": "dest_expected_schooling",
    "num__origin_expected_schooling": "origin_expected_schooling",
    "cat__pair_id_ukraine__poland": "pair_id_ukraine__poland",
}

STATS_COLS = {
    "country": "Country",
    "gpi": "Global_Peace_Index",
    "hdi": "Human Development Index (HDI) ",
    "life": "Life expectancy at birth",
    "school": "Expected years of schooling",
    "gdp_pc": "GDP/Capita",
    "lat": "Latitude",
    "lon": "Longitude",
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def normalize_country(s):
    return str(s).strip().lower()


def load_stats(stats_path):
    stats = pd.read_csv(stats_path).copy()
    stats["_country_key"] = stats[STATS_COLS["country"]].map(normalize_country)
    return stats


def build_feature_table(panel_path, stats_path):
    panel = pd.read_csv(panel_path).copy()
    stats = load_stats(stats_path)

    # Standardize keys
    panel["_origin_key"] = panel["origin"].map(normalize_country)
    panel["_dest_key"] = panel["destination"].map(normalize_country)

    origin_stats = stats.rename(columns={
        STATS_COLS["gpi"]: "origin_gpi",
        STATS_COLS["hdi"]: "origin_hdi",
        STATS_COLS["life"]: "origin_life_expectancy",
        STATS_COLS["school"]: "origin_expected_schooling",
        STATS_COLS["gdp_pc"]: "origin_gdp_pc",
        STATS_COLS["lat"]: "origin_lat",
        STATS_COLS["lon"]: "origin_lon",
    })[["_country_key", "origin_gpi", "origin_hdi", "origin_life_expectancy",
        "origin_expected_schooling", "origin_gdp_pc", "origin_lat", "origin_lon"]]

    dest_stats = stats.rename(columns={
        STATS_COLS["gpi"]: "dest_gpi",
        STATS_COLS["hdi"]: "dest_hdi",
        STATS_COLS["life"]: "dest_life_expectancy",
        STATS_COLS["school"]: "dest_expected_schooling",
        STATS_COLS["gdp_pc"]: "dest_gdp_pc",
        STATS_COLS["lat"]: "dest_lat",
        STATS_COLS["lon"]: "dest_lon",
    })[["_country_key", "dest_gpi", "dest_hdi", "dest_life_expectancy",
        "dest_expected_schooling", "dest_gdp_pc", "dest_lat", "dest_lon"]]

    panel = panel.merge(origin_stats, left_on="_origin_key", right_on="_country_key", how="left")
    panel = panel.drop(columns=["_country_key"])
    panel = panel.merge(dest_stats, left_on="_dest_key", right_on="_country_key", how="left")
    panel = panel.drop(columns=["_country_key"])

    # Build missing count fields from ratios if needed
    if "push_count" not in panel.columns:
        if {"push_ratio", "n_articles"}.issubset(panel.columns):
            panel["push_count"] = panel["push_ratio"] * panel["n_articles"]
        else:
            panel["push_count"] = 0.0

    if "pull_count" not in panel.columns:
        if {"pull_ratio", "n_articles"}.issubset(panel.columns):
            panel["pull_count"] = panel["pull_ratio"] * panel["n_articles"]
        else:
            panel["pull_count"] = 0.0

    for c in ["unclear_count", "mixed_count"]:
        if c not in panel.columns:
            panel[c] = 0.0

    # Derived features
    panel["gdp_pc_diff"] = panel["dest_gdp_pc"] - panel["origin_gdp_pc"]
    panel["hdi_diff"] = panel["dest_hdi"] - panel["origin_hdi"]
    panel["gpi_diff"] = panel["dest_gpi"] - panel["origin_gpi"]

    panel["log_origin_gdp_pc"] = np.log(panel["origin_gdp_pc"].clip(lower=1e-9))
    panel["log_dest_gdp_pc"] = np.log(panel["dest_gdp_pc"].clip(lower=1e-9))

    panel["distance_km"] = panel.apply(
        lambda r: haversine_km(r["origin_lat"], r["origin_lon"], r["dest_lat"], r["dest_lon"]),
        axis=1
    )
    panel["log_distance_km"] = np.log(panel["distance_km"].clip(lower=1e-9))

    # Simple pair dummy for the coefficient file you showed
    panel["pair_id_ukraine__poland"] = (
        (panel["_origin_key"] == "ukraine") & (panel["_dest_key"] == "poland")
    ).astype(float)

    return panel


def read_coefficients(path):
    df = pd.read_csv(path).copy()
    if "feature" not in df.columns or "coefficient" not in df.columns:
        raise ValueError(f"{path} must contain at least 'feature' and 'coefficient' columns.")
    if "final_penalty_weight" not in df.columns:
        df["final_penalty_weight"] = 1.0
    return df


def formula_string(coeff_df, use_wj=False):
    lines = ["migration ="]
    pieces = []
    for _, row in coeff_df.iterrows():
        feat_raw = row["feature"]
        feat = FEATURE_MAP.get(feat_raw, feat_raw)
        beta = row["coefficient"]
        if use_wj:
            wj = row.get("final_penalty_weight", 1.0)
            pieces.append(f"({wj:.6g})*({beta:.6g})*{feat}")
        else:
            pieces.append(f"({beta:.6g})*{feat}")

    for i, piece in enumerate(pieces):
        prefix = "    " if i == 0 else "  + "
        lines.append(prefix + piece)
    return "\n".join(lines)


def apply_formula(df, coeff_df, use_wj=False):
    result = pd.Series(0.0, index=df.index)
    term_columns = []

    for _, row in coeff_df.iterrows():
        feat_raw = row["feature"]
        if feat_raw not in FEATURE_MAP:
            # Skip unknown features, but keep script robust.
            continue

        feat = FEATURE_MAP[feat_raw]
        if feat not in df.columns:
            df[feat] = 0.0

        beta = float(row["coefficient"])
        if use_wj:
            wj = float(row.get("final_penalty_weight", 1.0))
            term = wj * beta * df[feat]
            term_name = f"term_llm_{feat}"
        else:
            term = beta * df[feat]
            term_name = f"term_plain_{feat}"

        df[term_name] = term
        term_columns.append(term_name)
        result = result + term

    return result, term_columns


def main():
    parser = argparse.ArgumentParser(description="Apply plain Lasso and LLM-Lasso coefficients to a migration formula.")
    parser.add_argument("--panel", required=True, help="CSV with origin, destination, year, migration_flow, and article columns.")
    parser.add_argument("--stats", required=True, help="Country stats CSV with GDP/Capita, HDI, GPI, lat/lon.")
    parser.add_argument("--plain-coefs", required=True, help="CSV of plain Lasso coefficients.")
    parser.add_argument("--llm-coefs", required=True, help="CSV of LLM-Lasso coefficients.")
    parser.add_argument("--output", default="migration_formula_output.csv", help="Where to save the output CSV.")
    parser.add_argument("--exp-transform", action="store_true",
                        help="If the formula predicts log(migration), also create exp(...) columns.")
    args = parser.parse_args()

    df = build_feature_table(args.panel, args.stats)
    plain = read_coefficients(args.plain_coefs)
    llm = read_coefficients(args.llm_coefs)

    print("\n================ PLAIN LASSO FORMULA ================\n")
    print(formula_string(plain, use_wj=False))

    print("\n================ LLM LASSO FORMULA ==================\n")
    print(formula_string(llm, use_wj=True))

    df["migration_plain_formula"], plain_terms = apply_formula(df, plain, use_wj=False)
    df["migration_llm_formula"], llm_terms = apply_formula(df, llm, use_wj=True)

    if args.exp_transform:
        df["migration_plain_formula_exp"] = np.exp(df["migration_plain_formula"])
        df["migration_llm_formula_exp"] = np.exp(df["migration_llm_formula"])

    ordered_cols = [
        "origin", "destination", "year", "migration_flow",
        "n_articles", "push_ratio", "pull_ratio",
        "push_count", "pull_count", "unclear_count", "mixed_count",
        "origin_gdp_pc", "dest_gdp_pc", "gdp_pc_diff",
        "origin_hdi", "dest_hdi", "hdi_diff",
        "origin_gpi", "dest_gpi", "gpi_diff",
        "distance_km", "log_distance_km",
        "migration_plain_formula", "migration_llm_formula",
    ]
    if args.exp_transform:
        ordered_cols += ["migration_plain_formula_exp", "migration_llm_formula_exp"]

    existing_cols = [c for c in ordered_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining_cols]

    df.to_csv(args.output, index=False)
    print(f"\nSaved output to: {args.output}")


if __name__ == "__main__":
    main()
