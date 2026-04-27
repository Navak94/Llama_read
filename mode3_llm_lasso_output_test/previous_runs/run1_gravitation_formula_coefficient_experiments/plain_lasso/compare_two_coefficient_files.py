import argparse
import math
from pathlib import Path
import pandas as pd
import numpy as np

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

    panel = panel.merge(origin_stats, left_on="_origin_key", right_on="_country_key", how="left").drop(columns=["_country_key"])
    panel = panel.merge(dest_stats, left_on="_dest_key", right_on="_country_key", how="left").drop(columns=["_country_key"])

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

    panel["gdp_pc_diff"] = panel["dest_gdp_pc"] - panel["origin_gdp_pc"]
    panel["hdi_diff"] = panel["dest_hdi"] - panel["origin_hdi"]
    panel["gpi_diff"] = panel["dest_gpi"] - panel["origin_gpi"]

    panel["log_origin_gdp_pc"] = np.log(panel["origin_gdp_pc"].clip(lower=1e-9))
    panel["log_dest_gdp_pc"] = np.log(panel["dest_gdp_pc"].clip(lower=1e-9))

    panel["distance_km"] = panel.apply(
        lambda r: haversine_km(r["origin_lat"], r["origin_lon"], r["dest_lat"], r["dest_lon"]),
        axis=1,
    )
    panel["log_distance_km"] = np.log(panel["distance_km"].clip(lower=1e-9))

    panel["pair_id_ukraine__poland"] = (
        (panel["_origin_key"] == "ukraine") & (panel["_dest_key"] == "poland")
    ).astype(float)

    return panel


def read_coefficients(path):
    df = pd.read_csv(path).copy()
    if "feature" not in df.columns or "coefficient" not in df.columns:
        raise ValueError(
            f"{path} is not a coefficient file. It needs at least 'feature' and 'coefficient' columns."
        )
    if "final_penalty_weight" not in df.columns:
        df["final_penalty_weight"] = 1.0
    return df


def formula_string(coeff_df, use_wj=False):
    lines = ["migration ="]
    for i, (_, row) in enumerate(coeff_df.iterrows()):
        feat_raw = row["feature"]
        feat = FEATURE_MAP.get(feat_raw, feat_raw)
        beta = float(row["coefficient"])
        if use_wj:
            wj = float(row.get("final_penalty_weight", 1.0))
            piece = f"({wj:.6g})*({beta:.6g})*{feat}"
        else:
            piece = f"({beta:.6g})*{feat}"
        prefix = "    " if i == 0 else "  + "
        lines.append(prefix + piece)
    return "\n".join(lines)


def apply_formula(df, coeff_df, use_wj=False, prefix="model"):
    result = pd.Series(0.0, index=df.index)
    for _, row in coeff_df.iterrows():
        feat_raw = row["feature"]
        if feat_raw not in FEATURE_MAP:
            continue
        feat = FEATURE_MAP[feat_raw]
        if feat not in df.columns:
            df[feat] = 0.0
        beta = float(row["coefficient"])
        if use_wj:
            wj = float(row.get("final_penalty_weight", 1.0))
            term = wj * beta * df[feat]
        else:
            term = beta * df[feat]
        df[f"term_{prefix}_{feat}"] = term
        result = result + term
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare two coefficient files by plugging them into the migration formula.")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--coef-a", required=True)
    parser.add_argument("--coef-b", required=True)
    parser.add_argument("--label-a", default="model_a")
    parser.add_argument("--label-b", default="model_b")
    parser.add_argument("--use-wj-a", action="store_true")
    parser.add_argument("--use-wj-b", action="store_true")
    parser.add_argument("--output", default="migration_compare_output.csv")
    parser.add_argument("--exp-transform", action="store_true")
    args = parser.parse_args()

    df = build_feature_table(args.panel, args.stats)
    summary_rows = []

    for coef_path, label, use_wj in [
        (args.coef_a, args.label_a, args.use_wj_a),
        (args.coef_b, args.label_b, args.use_wj_b),
    ]:
        print(f"\n================ {label.upper()} ================\n")
        try:
            coef_df = read_coefficients(coef_path)
            print(formula_string(coef_df, use_wj=use_wj))
            df[f"migration_{label}"] = apply_formula(df, coef_df, use_wj=use_wj, prefix=label)
            if args.exp_transform:
                df[f"migration_{label}_exp"] = np.exp(df[f"migration_{label}"])
            summary_rows.append({
                "label": label,
                "file": coef_path,
                "status": "ok",
                "used_wj": use_wj,
                "nonzero_coefficients": int((coef_df["coefficient"].abs() > 1e-12).sum()),
            })
        except Exception as e:
            print(f"Could not use {coef_path}: {e}")
            summary_rows.append({
                "label": label,
                "file": coef_path,
                "status": f"error: {e}",
                "used_wj": use_wj,
                "nonzero_coefficients": np.nan,
            })

    df.to_csv(args.output, index=False)
    summary_path = str(Path(args.output).with_name(Path(args.output).stem + "_summary.csv"))
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nSaved comparison output to: {args.output}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
