import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================================================
# FEATURE MAP FOR WJ FILES
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

    panel = panel.merge(origin_stats, left_on="_origin_key", right_on="_country_key", how="left").drop(columns=["_country_key"])
    panel = panel.merge(dest_stats, left_on="_dest_key", right_on="_country_key", how="left").drop(columns=["_country_key"])

    # Build counts if only ratios exist
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

    # Gravity / corridor features
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


def default_feature_list():
    return [
        "log_origin_gdp_pc",
        "log_dest_gdp_pc",
        "log_distance_km",
        "gdp_pc_diff",
        "hdi_diff",
        "gpi_diff",
        "push_count",
        "pull_count",
        "unclear_count",
        "mixed_count",
        "n_articles",
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


def read_wj_file(path):
    df = pd.read_csv(path).copy()
    if "feature" not in df.columns:
        raise ValueError("WJ file must contain a 'feature' column.")
    if "final_penalty_weight" not in df.columns:
        raise ValueError("WJ file must contain a 'final_penalty_weight' column.")

    wj_map = {}
    for _, row in df.iterrows():
        raw_name = row["feature"]
        mapped = FEATURE_MAP.get(raw_name, raw_name)
        wj_map[mapped] = float(row["final_penalty_weight"])
    return wj_map


def build_weight_vector(features, wj_map=None):
    if wj_map is None:
        return np.ones(len(features), dtype=float)
    return np.array([float(wj_map.get(f, 1.0)) for f in features], dtype=float)


def fit_model(df, features, weight_vector, log_target=False):
    X = df[features].copy().fillna(0.0)
    X_weighted = X.mul(weight_vector, axis=1)

    if log_target:
        y = np.log(df["migration_flow"].clip(lower=1e-9)).astype(float)
        target_name = "log(migration)"
    else:
        y = df["migration_flow"].astype(float)
        target_name = "migration"

    model = LinearRegression()
    model.fit(X_weighted, y)

    pred = model.predict(X_weighted)
    return model, pred, target_name, X, X_weighted


def format_formula(intercept, features, betas, weight_vector, target_name):
    lines = [f"{target_name} = {intercept:.6g}"]
    for feat, beta, wj in zip(features, betas, weight_vector):
        sign = "+" if beta >= 0 else "-"
        if abs(wj - 1.0) < 1e-12:
            lines.append(f"  {sign} ({abs(beta):.6g})*{feat}")
        else:
            lines.append(f"  {sign} ({abs(beta):.6g})*({wj:.6g})*{feat}")
    return "\n".join(lines)


def save_betas(path, label, intercept, features, betas, weight_vector):
    rows = [{
        "model": label,
        "feature": "intercept",
        "beta": intercept,
        "wj": 1.0,
        "effective_multiplier": intercept,
    }]
    for feat, beta, wj in zip(features, betas, weight_vector):
        rows.append({
            "model": label,
            "feature": feat,
            "beta": beta,
            "wj": wj,
            "effective_multiplier": beta * wj,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Fit a gravity-style regression twice: once plain, once with wj multipliers."
    )
    parser.add_argument("--panel", required=True, help="Panel CSV with origin, destination, year, migration_flow, and article columns.")
    parser.add_argument("--stats", required=True, help="Country stats CSV.")
    parser.add_argument("--wj-file", required=False, help="Coefficient CSV containing final_penalty_weight for the weighted model.")
    parser.add_argument("--output", default="gravity_regression_output.csv", help="Predictions/output CSV.")
    parser.add_argument("--log-target", action="store_true", help="Fit log(migration_flow) instead of migration_flow.")
    args = parser.parse_args()

    df = build_feature_table(args.panel, args.stats)
    features = default_feature_list()

    if "migration_flow" not in df.columns:
        raise ValueError("Panel file must contain a 'migration_flow' column.")

    # Plain model: all wj = 1
    plain_w = np.ones(len(features), dtype=float)

    # Weighted model: use wj file if given, otherwise all 1s
    if args.wj_file:
        wj_map = read_wj_file(args.wj_file)
        weighted_w = build_weight_vector(features, wj_map)
    else:
        weighted_w = np.ones(len(features), dtype=float)

    plain_model, plain_pred, target_name, _, _ = fit_model(
        df, features, plain_w, log_target=args.log_target
    )
    weighted_model, weighted_pred, _, _, _ = fit_model(
        df, features, weighted_w, log_target=args.log_target
    )

    print("\n================ PLAIN GRAVITY FORMULA ================\n")
    print(format_formula(
        float(plain_model.intercept_),
        features,
        plain_model.coef_.astype(float),
        plain_w,
        target_name,
    ))

    print("\n================ WJ-WEIGHTED GRAVITY FORMULA ================\n")
    print(format_formula(
        float(weighted_model.intercept_),
        features,
        weighted_model.coef_.astype(float),
        weighted_w,
        target_name,
    ))

    if args.log_target:
        df["predicted_log_migration_plain"] = plain_pred
        df["predicted_log_migration_wj"] = weighted_pred
        df["predicted_migration_plain"] = np.exp(plain_pred)
        df["predicted_migration_wj"] = np.exp(weighted_pred)
    else:
        df["predicted_migration_plain"] = plain_pred
        df["predicted_migration_wj"] = weighted_pred

    out_path = Path(args.output)
    df.to_csv(out_path, index=False)

    save_betas(
        out_path.with_name(out_path.stem + "_plain_betas.csv"),
        "plain",
        float(plain_model.intercept_),
        features,
        plain_model.coef_.astype(float),
        plain_w,
    )
    save_betas(
        out_path.with_name(out_path.stem + "_wj_betas.csv"),
        "wj_weighted",
        float(weighted_model.intercept_),
        features,
        weighted_model.coef_.astype(float),
        weighted_w,
    )

    print(f"\nSaved predictions to: {out_path}")
    print(f"Saved plain betas to: {out_path.with_name(out_path.stem + '_plain_betas.csv')}")
    print(f"Saved weighted betas to: {out_path.with_name(out_path.stem + '_wj_betas.csv')}")


if __name__ == "__main__":
    main()
