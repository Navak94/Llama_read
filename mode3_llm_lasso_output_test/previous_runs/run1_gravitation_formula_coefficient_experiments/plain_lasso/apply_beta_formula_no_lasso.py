import argparse
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================================================
# COUNTRY STATS COLUMN MAP
# =========================================================
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

    panel = panel.merge(origin_stats, left_on="_origin_key", right_on="_country_key", how="left")
    panel = panel.drop(columns=["_country_key"])
    panel = panel.merge(dest_stats, left_on="_dest_key", right_on="_country_key", how="left")
    panel = panel.drop(columns=["_country_key"])

    # Build counts from ratios if needed
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

    # Derived numeric features
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

    panel["pair_id_ukraine__poland"] = (
        (panel["_origin_key"] == "ukraine") & (panel["_dest_key"] == "poland")
    ).astype(float)

    return panel


def formula_string(intercept, features, betas, target_name="migration"):
    lines = [f"{target_name} = {intercept:.6g}"]
    for feat, beta in zip(features, betas):
        sign = "+" if beta >= 0 else "-"
        lines.append(f"  {sign} ({abs(beta):.6g})*{feat}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Fit ordinary linear regression to migration_flow and print the learned beta formula."
    )
    parser.add_argument("--panel", required=True, help="CSV with origin, destination, year, migration_flow, and article columns.")
    parser.add_argument("--stats", required=True, help="Country stats CSV with GDP/Capita, HDI, GPI, lat/lon.")
    parser.add_argument("--output", default="migration_beta_output.csv", help="Where to save the output CSV.")
    parser.add_argument("--log-target", action="store_true",
                        help="Fit log(migration_flow) instead of migration_flow. Also outputs exp(prediction).")
    args = parser.parse_args()

    df = build_feature_table(args.panel, args.stats)

    features = [
        "gdp_pc_diff",
        "hdi_diff",
        "gpi_diff",
        "push_count",
        "pull_count",
        "unclear_count",
        "mixed_count",
        "n_articles",
        "log_origin_gdp_pc",
        "log_dest_gdp_pc",
        "log_distance_km",
        "dest_hdi",
        "origin_hdi",
        "dest_gpi",
        "origin_gpi",
        "dest_life_expectancy",
        "origin_life_expectancy",
        "dest_expected_schooling",
        "origin_expected_schooling",
        "pair_id_ukraine__poland",
    ]

    missing = [c for c in ["migration_flow"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required target column(s): {missing}")

    X = df[features].copy().fillna(0.0)

    if args.log_target:
        # Avoid log(0)
        y = np.log(df["migration_flow"].clip(lower=1e-9))
        target_name = "log(migration)"
    else:
        y = df["migration_flow"].astype(float)
        target_name = "migration"

    model = LinearRegression()
    model.fit(X, y)

    intercept = float(model.intercept_)
    betas = model.coef_.astype(float)

    print("\n================ LEARNED BETA FORMULA ================\n")
    print(formula_string(intercept, features, betas, target_name=target_name))

    # Prediction using the learned betas
    pred = model.predict(X)
    if args.log_target:
        df["predicted_log_migration"] = pred
        df["predicted_migration"] = np.exp(pred)
    else:
        df["predicted_migration"] = pred

    # Save coefficients too
    coef_df = pd.DataFrame({
        "feature": ["intercept"] + features,
        "beta": [intercept] + list(betas)
    })
    coef_out = args.output.replace(".csv", "_betas.csv")
    coef_df.to_csv(coef_out, index=False)

    ordered_cols = [
        "origin", "destination", "year", "migration_flow",
        "n_articles", "push_ratio", "pull_ratio",
        "push_count", "pull_count", "unclear_count", "mixed_count",
        "origin_gdp_pc", "dest_gdp_pc", "gdp_pc_diff",
        "origin_hdi", "dest_hdi", "hdi_diff",
        "origin_gpi", "dest_gpi", "gpi_diff",
        "distance_km", "log_distance_km",
    ]
    if args.log_target:
        ordered_cols += ["predicted_log_migration", "predicted_migration"]
    else:
        ordered_cols += ["predicted_migration"]

    existing_cols = [c for c in ordered_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining_cols]

    df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to: {args.output}")
    print(f"Saved betas to: {coef_out}")


if __name__ == "__main__":
    main()
