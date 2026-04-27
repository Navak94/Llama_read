from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EARTH_RADIUS_KM = 6371.0088
Mexico_1_URL = "https://www.nbcnews.com/politics/immigration/migrant-border-crossings-fiscal-year-2022-topped-276-million-breaking-rcna53517"
#Mexico_2_URL =
Ukraine_1_URL ="https://www.nbcnews.com/world/ukraine/ukrainian-refugees-years-after-russia-invasion-fear-return-rcna250438"
Ukraine_2_URL = "https://www.npr.org/2022/11/01/1132167234/russia-ukraine-war-unemployment-displaced-economy"
Ukraine_3_URL = "https://www.npr.org/2022/05/10/1093066817/ukraine-war-gas-prices-refugees"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def normalize_country(s: str) -> str:
    return str(s).strip().lower()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


# -----------------------------------------------------------------------------
# Country feature merge
# -----------------------------------------------------------------------------

def load_country_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["Country_key"] = df["Country"].map(normalize_country)

    rename_map = {
        "Global_Peace_Index": "gpi",
        "Human Development Index (HDI) ": "hdi",
        "Life expectancy at birth": "life_expectancy",
        "Expected years of schooling": "expected_schooling",
        "Mean years of schooling": "mean_schooling",
        "GDP/Capita": "gdp_per_capita",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    keep = ["Country_key"] + list(rename_map.keys())
    df = df[keep].rename(columns=rename_map)
    return df


# -----------------------------------------------------------------------------
# Build a Model-4-like design matrix
# -----------------------------------------------------------------------------

def enrich_panel_with_country_features(panel_df: pd.DataFrame, country_df: pd.DataFrame) -> pd.DataFrame:
    df = panel_df.copy()

    for col in ["origin", "destination"]:
        if col not in df.columns:
            raise ValueError(f"Input panel CSV must contain a '{col}' column.")

    if "migration_flow" not in df.columns:
        raise ValueError("Input panel CSV must contain a 'migration_flow' column.")

    df["origin_key"] = df["origin"].map(normalize_country)
    df["destination_key"] = df["destination"].map(normalize_country)

    origin_features = country_df.add_prefix("origin_")
    dest_features = country_df.add_prefix("dest_")

    df = df.merge(origin_features, left_on="origin_key", right_on="origin_Country_key", how="left")
    df = df.merge(dest_features, left_on="destination_key", right_on="dest_Country_key", how="left")

    # Model-4-ish variables from the paper:
    # p1 log(P_it), p2 log(P_jt), p3 log(I_it), p4 log(I_jt),
    # p7 E_it, p8 E_jt, p9 H_it, p10 H_jt,
    # p11 S_it, p12 S_jt, p13 R_it, p14 R_jt, p15 Y_it, p16 B_jt, + c_ij
    #
    # With your available country file, we can approximate:
    # - I using GDP/Capita
    # - E using expected schooling
    # - H using life expectancy
    # - S/R using Global Peace Index and/or HDI as proxies
    # - c_ij via pair fixed-effect dummies
    # Missing columns such as population, age 20-35, and immigration policy can be
    # supplied in the panel CSV later if you have them.

    # Distance from lat/lon (paper's time-invariant pair feature)
    df["distance_km"] = df.apply(
        lambda r: haversine_km(
            r["origin_latitude"], r["origin_longitude"], r["dest_latitude"], r["dest_longitude"]
        )
        if pd.notna(r["origin_latitude"]) and pd.notna(r["origin_longitude"]) and pd.notna(r["dest_latitude"]) and pd.notna(r["dest_longitude"])
        else np.nan,
        axis=1,
    )

    # Log versions for gravity-style terms
    if "origin_population" in df.columns:
        df["log_origin_population"] = np.log1p(df["origin_population"].clip(lower=0))
    if "dest_population" in df.columns:
        df["log_dest_population"] = np.log1p(df["dest_population"].clip(lower=0))

    df["log_origin_gdp_pc"] = np.log1p(df["origin_gdp_per_capita"].clip(lower=0))
    df["log_dest_gdp_pc"] = np.log1p(df["dest_gdp_per_capita"].clip(lower=0))
    df["log_distance_km"] = np.log1p(df["distance_km"].clip(lower=0))

    # Diffs can help when the data are sparse
    df["gdp_pc_diff"] = df["dest_gdp_per_capita"] - df["origin_gdp_per_capita"]
    df["hdi_diff"] = df["dest_hdi"] - df["origin_hdi"]
    df["gpi_diff"] = df["dest_gpi"] - df["origin_gpi"]

    # Pair fixed-effect key (c_ij analogue)
    df["pair_id"] = df["origin_key"] + "__" + df["destination_key"]

    # Target in log-space, which is usually more stable for flow data
    df["log_migration_flow"] = np.log1p(df["migration_flow"].clip(lower=0))

    return df


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def choose_feature_columns(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    numeric_candidates = [
        "log_origin_population",
        "log_dest_population",
        "log_origin_gdp_pc",
        "log_dest_gdp_pc",
        "origin_expected_schooling",
        "dest_expected_schooling",
        "origin_life_expectancy",
        "dest_life_expectancy",
        "origin_gpi",
        "dest_gpi",
        "origin_hdi",
        "dest_hdi",
        "log_distance_km",
        "gdp_pc_diff",
        "hdi_diff",
        "gpi_diff",
        # Optional news / event features from your panel CSV if present
        "push_ratio",
        "pull_ratio",
        "mixed_ratio",
        "unclear_ratio",
        "n_articles",
        "push_count",
        "pull_count",
        "mixed_count",
        "unclear_count",
    ]
    numeric_features = [c for c in numeric_candidates if c in df.columns]
    categorical_features = [c for c in ["pair_id"] if c in df.columns]
    return numeric_features, categorical_features


def fit_lasso(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_features, categorical_features = choose_feature_columns(df)
    if not numeric_features and not categorical_features:
        raise ValueError("No usable features were found. Add numeric variables to the panel CSV.")

    X = df[numeric_features + categorical_features].copy()
    y = df["log_migration_flow"].astype(float)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    # With small datasets, CV folds must be modest.
    n_rows = len(df)
    cv_folds = max(2, min(5, n_rows // 3 if n_rows >= 6 else 2))

    model = Pipeline([
        ("prep", preprocessor),
        ("lasso", LassoCV(cv=cv_folds, random_state=42, max_iter=50000, n_alphas=200)),
    ])

    model.fit(X, y)
    pred_log = model.predict(X)
    pred_flow = np.expm1(pred_log)

    metrics = {
        "rows": int(len(df)),
        "cv_folds": int(cv_folds),
        "alpha": float(model.named_steps["lasso"].alpha_),
        "r2_train": float(r2_score(y, pred_log)),
        "rmse_train_log": float(mean_squared_error(y, pred_log, squared=False)),
        "mae_train_log": float(mean_absolute_error(y, pred_log)),
    }

    # Recover feature names after preprocessing
    feature_names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["lasso"].coef_
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    pred_df = df.copy()
    pred_df["pred_log_migration_flow"] = pred_log
    pred_df["pred_migration_flow"] = pred_flow
    pred_df["residual_log"] = pred_df["log_migration_flow"] - pred_df["pred_log_migration_flow"]

    pred_df.to_csv(output_dir / "lasso_predictions.csv", index=False)
    coef_df.to_csv(output_dir / "lasso_coefficients.csv", index=False)
    with open(output_dir / "lasso_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(f"  {output_dir / 'lasso_predictions.csv'}")
    print(f"  {output_dir / 'lasso_coefficients.csv'}")
    print(f"  {output_dir / 'lasso_metrics.json'}")
    print("\nTop coefficients:")
    print(coef_df.head(20).to_string(index=False))
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approximate the paper's Model 4 with a Lasso on corridor-time panel data."
    )
    parser.add_argument(
        "--panel_csv",
        required=True,
        help=(
            "CSV with at least: origin, destination, migration_flow. "
            "Optional columns: time, origin_population, dest_population, push_ratio, pull_ratio, etc."
        ),
    )
    parser.add_argument(
        "--country_csv",
        default="HDI_GPI_with_GDP_LATLON.csv",
        help="Path to your country feature file.",
    )
    parser.add_argument(
        "--output_dir",
        default="lasso_model4_output",
        help="Where results should be written.",
    )
    args = parser.parse_args()

    panel_df = pd.read_csv(args.panel_csv)
    country_df = load_country_features(args.country_csv)
    model_df = enrich_panel_with_country_features(panel_df, country_df)

    print("Using columns:")
    print(model_df.columns.tolist())
    fit_lasso(model_df, args.output_dir)


if __name__ == "__main__":
    main()
