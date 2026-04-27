import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# =========================================================
# FEATURE MAP FOR COEFFICIENT / PENALTY FILES
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

    if "push_count" not in panel.columns:
        panel["push_count"] = panel["push_ratio"] * panel["n_articles"] if {"push_ratio", "n_articles"}.issubset(panel.columns) else 0.0
    if "pull_count" not in panel.columns:
        panel["pull_count"] = panel["pull_ratio"] * panel["n_articles"] if {"pull_ratio", "n_articles"}.issubset(panel.columns) else 0.0
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


def read_penalty_file(path):
    df = pd.read_csv(path).copy()
    if "feature" not in df.columns:
        raise ValueError("Penalty file must contain a 'feature' column.")

    penalty_col = None
    for candidate in ["final_penalty_weight", "base_penalty_v", "wj", "penalty", "llm_penalty_score"]:
        if candidate in df.columns:
            penalty_col = candidate
            break
    if penalty_col is None:
        raise ValueError(
            "Penalty file must contain one of: final_penalty_weight, base_penalty_v, wj, penalty, llm_penalty_score"
        )

    penalty_map = {}
    for _, row in df.iterrows():
        raw_name = row["feature"]
        mapped = FEATURE_MAP.get(raw_name, raw_name)
        penalty_map[mapped] = float(row[penalty_col])
    return penalty_map, penalty_col


def build_penalty_vector(features, penalty_map=None):
    if penalty_map is None:
        return np.ones(len(features), dtype=float)
    return np.array([float(penalty_map.get(f, 1.0)) for f in features], dtype=float)


def choose_target(df, log_target=False):
    if log_target:
        return np.log(df["migration_flow"].clip(lower=1e-9)).astype(float).to_numpy(), "log(migration_flow)"
    return df["migration_flow"].astype(float).to_numpy(), "migration_flow"


def standardize_X(df, features):
    X = df[features].copy().fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler


def apply_llm_lasso_transform(X_scaled, penalties, eta=1, eps=1e-8):
    w = np.power(np.maximum(penalties, eps), eta)
    X_tilde = X_scaled / w
    return X_tilde, w


def fit_plain_lasso(X_scaled, y, alpha, max_iter=50000):
    model = Lasso(alpha=alpha, max_iter=max_iter)
    model.fit(X_scaled, y)
    beta = model.coef_.copy()
    intercept = float(model.intercept_)
    pred = model.predict(X_scaled)
    return model, intercept, beta, pred


def fit_weighted_llm_lasso(X_scaled, y, penalties, alpha, eta=1, max_iter=50000):
    X_tilde, w = apply_llm_lasso_transform(X_scaled, penalties, eta=eta)
    model = Lasso(alpha=alpha, max_iter=max_iter)
    model.fit(X_tilde, y)
    theta = model.coef_.copy()
    beta = theta / w
    intercept = float(model.intercept_)
    pred = intercept + X_scaled @ beta
    return model, intercept, beta, theta, w, pred


def tune_alpha_cv(X_scaled, y, penalties=None, eta=1, alpha_grid=None, n_splits=5, max_iter=50000):
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 1, 50)

    n_rows = len(y)
    n_splits = max(2, min(n_splits, n_rows // 2 if n_rows >= 4 else 2))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_alpha = None
    best_mse = float("inf")

    for alpha in alpha_grid:
        fold_mse = []
        for train_idx, val_idx in cv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_val = X_scaled[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            if penalties is None:
                model = Lasso(alpha=float(alpha), max_iter=max_iter)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
            else:
                X_train_tilde, w = apply_llm_lasso_transform(X_train, penalties, eta=eta)
                X_val_tilde = X_val / w
                model = Lasso(alpha=float(alpha), max_iter=max_iter)
                model.fit(X_train_tilde, y_train)
                theta = model.coef_.copy()
                beta = theta / w
                pred = float(model.intercept_) + X_val @ beta

            fold_mse.append(mean_squared_error(y_val, pred))

        avg_mse = float(np.mean(fold_mse))
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_alpha = float(alpha)

    return float(best_alpha), float(best_mse)


def format_estimated_equation(intercept, features, beta, target_name, title):
    lines = [title, f"{target_name} = {intercept:.6g}"]
    for feat, coef in zip(features, beta):
        sign = "+" if coef >= 0 else "-"
        lines.append(f"  {sign} ({abs(float(coef)):.6g})*{feat}")
    return "\n".join(lines)


def format_objective_plain(alpha, target_name):
    return (
        f"Objective solved (plain Lasso on {target_name}):\n"
        f"min_beta  ||y - X beta||^2 + {alpha:.6g} * sum_j |beta_j|"
    )


def format_objective_llm(alpha, features, penalties, eta, target_name):
    terms = [f"({float(p):.6g}^{eta})|{feat}|" for feat, p in zip(features, penalties)]
    joined = " + ".join(terms)
    return (
        f"Objective solved (LLM-weighted Lasso on {target_name}):\n"
        f"min_beta  ||y - X beta||^2 + {alpha:.6g} * [{joined}]"
    )


def metrics_dict(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_coefficients(path, model_name, intercept, features, beta, penalties=None, eta=1):
    rows = [{
        "model": model_name,
        "feature": "intercept",
        "coefficient": float(intercept),
        "penalty_weight": 1.0,
        "eta": eta,
    }]
    for i, (feat, coef) in enumerate(zip(features, beta)):
        rows.append({
            "model": model_name,
            "feature": feat,
            "coefficient": float(coef),
            "penalty_weight": float(penalties[i]) if penalties is not None else 1.0,
            "eta": eta,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare plain Lasso vs proper LLM-weighted Lasso on the same gravity-style migration formula. "
            "The weighted version solves min_beta ||y-Xbeta||^2 + alpha * sum_j w_j |beta_j|."
        )
    )
    parser.add_argument("--panel", required=True, help="Panel CSV with origin, destination, migration_flow, and article columns.")
    parser.add_argument("--stats", required=True, help="Country stats CSV.")
    parser.add_argument("--wj-file", required=False, help="CSV containing feature penalties/weights from the LLM-Lasso run.")
    parser.add_argument("--output", default="gravity_lasso_compare_output.csv", help="Predictions/output CSV.")
    parser.add_argument("--log-target", action="store_true", help="Fit log(migration_flow) instead of migration_flow.")
    parser.add_argument("--eta", type=int, default=1, help="Exponent for penalty weights, matching your main pipeline.")
    parser.add_argument("--alpha", type=float, default=None, help="If provided, use this alpha for both models instead of CV tuning.")
    args = parser.parse_args()

    df = build_feature_table(args.panel, args.stats)
    features = default_feature_list()
    if "migration_flow" not in df.columns:
        raise ValueError("Panel file must contain a 'migration_flow' column.")

    X_raw, X_scaled, scaler = standardize_X(df, features)
    y, target_name = choose_target(df, log_target=args.log_target)

    penalty_map = None
    penalty_col = None
    if args.wj_file:
        penalty_map, penalty_col = read_penalty_file(args.wj_file)
    penalties = build_penalty_vector(features, penalty_map)
    plain_penalties = np.ones(len(features), dtype=float)

    if args.alpha is None:
        alpha_plain, cv_mse_plain = tune_alpha_cv(X_scaled, y, penalties=None)
        alpha_llm, cv_mse_llm = tune_alpha_cv(X_scaled, y, penalties=penalties, eta=args.eta)
    else:
        alpha_plain = float(args.alpha)
        alpha_llm = float(args.alpha)
        cv_mse_plain = None
        cv_mse_llm = None

    plain_model, plain_intercept, plain_beta, plain_pred = fit_plain_lasso(X_scaled, y, alpha=alpha_plain)
    llm_model, llm_intercept, llm_beta, llm_theta, llm_w, llm_pred = fit_weighted_llm_lasso(
        X_scaled, y, penalties=penalties, alpha=alpha_llm, eta=args.eta
    )

    print("\n================ PLAIN LASSO OBJECTIVE ================\n")
    print(format_objective_plain(alpha_plain, target_name))

    print("\n================ LLM-WEIGHTED LASSO OBJECTIVE ================\n")
    print(format_objective_llm(alpha_llm, features, penalties, args.eta, target_name))

    print("\n================ PLAIN LASSO EQUATION ================\n")
    print(format_estimated_equation(plain_intercept, features, plain_beta, target_name, "Estimated equation:"))

    print("\n================ LLM-WEIGHTED LASSO EQUATION ================\n")
    print(format_estimated_equation(llm_intercept, features, llm_beta, target_name, "Estimated equation:"))

    plain_metrics = metrics_dict(y, plain_pred)
    llm_metrics = metrics_dict(y, llm_pred)

    print("\n================ METRICS ================\n")
    print("Plain:", plain_metrics)
    print("LLM-weighted:", llm_metrics)

    out_path = Path(args.output)
    out_df = df.copy()
    out_df["pred_plain"] = plain_pred
    out_df["pred_llm_weighted"] = llm_pred
    if args.log_target:
        out_df["pred_plain_flow"] = np.exp(plain_pred)
        out_df["pred_llm_weighted_flow"] = np.exp(llm_pred)
    out_df.to_csv(out_path, index=False)

    coef_plain_path = out_path.with_name(out_path.stem + "_plain_coefficients.csv")
    coef_llm_path = out_path.with_name(out_path.stem + "_llm_weighted_coefficients.csv")
    save_coefficients(coef_plain_path, "plain_lasso", plain_intercept, features, plain_beta, plain_penalties, eta=1)
    save_coefficients(coef_llm_path, "llm_weighted_lasso", llm_intercept, features, llm_beta, penalties, eta=args.eta)

    metrics_path = out_path.with_name(out_path.stem + "_metrics.json")
    metrics_payload = {
        "target": target_name,
        "penalty_file_used": args.wj_file,
        "penalty_column_used": penalty_col,
        "eta": int(args.eta),
        "plain": {
            "alpha": float(alpha_plain),
            "cv_mse": cv_mse_plain,
            **plain_metrics,
        },
        "llm_weighted": {
            "alpha": float(alpha_llm),
            "cv_mse": cv_mse_llm,
            **llm_metrics,
        },
        "feature_penalties": {feat: float(p) for feat, p in zip(features, penalties)},
    }
    pd.Series(metrics_payload).to_json(metrics_path, indent=2)

    print(f"\nSaved predictions to: {out_path}")
    print(f"Saved plain coefficients to: {coef_plain_path}")
    print(f"Saved LLM-weighted coefficients to: {coef_llm_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
