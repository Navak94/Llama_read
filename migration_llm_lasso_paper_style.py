from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EARTH_RADIUS_KM = 6371.0088


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
    df = pd.read_csv(path).copy()
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

    df["distance_km"] = df.apply(
        lambda r: haversine_km(
            r["origin_latitude"], r["origin_longitude"], r["dest_latitude"], r["dest_longitude"]
        )
        if pd.notna(r["origin_latitude"])
        and pd.notna(r["origin_longitude"])
        and pd.notna(r["dest_latitude"])
        and pd.notna(r["dest_longitude"])
        else np.nan,
        axis=1,
    )

    if "origin_population" in df.columns:
        df["log_origin_population"] = np.log1p(df["origin_population"].clip(lower=0))
    if "dest_population" in df.columns:
        df["log_dest_population"] = np.log1p(df["dest_population"].clip(lower=0))

    df["log_origin_gdp_pc"] = np.log1p(df["origin_gdp_per_capita"].clip(lower=0))
    df["log_dest_gdp_pc"] = np.log1p(df["dest_gdp_per_capita"].clip(lower=0))
    df["log_distance_km"] = np.log1p(df["distance_km"].clip(lower=0))

    df["gdp_pc_diff"] = df["dest_gdp_per_capita"] - df["origin_gdp_per_capita"]
    df["hdi_diff"] = df["dest_hdi"] - df["origin_hdi"]
    df["gpi_diff"] = df["dest_gpi"] - df["origin_gpi"]

    df["pair_id"] = df["origin_key"] + "__" + df["destination_key"]
    df["log_migration_flow"] = np.log1p(df["migration_flow"].clip(lower=0))

    return df


# -----------------------------------------------------------------------------
# Feature selection for the migration formula
# -----------------------------------------------------------------------------


def choose_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
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


# -----------------------------------------------------------------------------
# Paper-style LLM scoring
# -----------------------------------------------------------------------------


def build_feature_descriptions(feature_names: List[str]) -> Dict[str, str]:
    descriptions = {
        "log_origin_population": "Log of origin-country population. Larger origin populations can create more potential migrants.",
        "log_dest_population": "Log of destination-country population. Larger destinations may absorb more migrants.",
        "log_origin_gdp_pc": "Log of origin-country GDP per capita. Lower origin income can act as push pressure.",
        "log_dest_gdp_pc": "Log of destination-country GDP per capita. Higher destination income can act as pull pressure.",
        "origin_expected_schooling": "Expected years of schooling in the origin country, a development and opportunity proxy.",
        "dest_expected_schooling": "Expected years of schooling in the destination country, a development and opportunity proxy.",
        "origin_life_expectancy": "Origin-country life expectancy, a welfare and development proxy.",
        "dest_life_expectancy": "Destination-country life expectancy, a welfare and development proxy.",
        "origin_gpi": "Origin-country Global Peace Index. Worse peace conditions may push migration.",
        "dest_gpi": "Destination-country Global Peace Index. Safer destinations may attract migrants.",
        "origin_hdi": "Origin-country HDI, a broad development proxy.",
        "dest_hdi": "Destination-country HDI, a broad development proxy.",
        "log_distance_km": "Log geographic distance between origin and destination. Distance usually increases migration friction.",
        "gdp_pc_diff": "Destination GDP per capita minus origin GDP per capita, an economic gap feature.",
        "hdi_diff": "Destination HDI minus origin HDI, a development gap feature.",
        "gpi_diff": "Destination GPI minus origin GPI, a safety or instability gap feature.",
        "push_ratio": "Share of migration-related articles labeled as push pressure for the corridor or period.",
        "pull_ratio": "Share of migration-related articles labeled as pull pressure for the corridor or period.",
        "mixed_ratio": "Share of articles labeled mixed, meaning both push and pull cues are present.",
        "unclear_ratio": "Share of articles labeled unclear, meaning weak or ambiguous migration evidence.",
        "n_articles": "Count of migration-related articles used to summarize the corridor or period.",
        "push_count": "Raw count of push-labeled articles.",
        "pull_count": "Raw count of pull-labeled articles.",
        "mixed_count": "Raw count of mixed-labeled articles.",
        "unclear_count": "Raw count of unclear-labeled articles.",
    }
    return {name: descriptions.get(name, f"Feature named {name} in the migration flow model.") for name in feature_names}



def build_llm_prompt(feature_names: List[str]) -> str:
    desc = build_feature_descriptions(feature_names)
    feature_block = "\n".join(f"- {name}: {desc[name]}" for name in feature_names)
    return f"""
We have corridor-level migration-flow data. We want to build a statistical Lasso model to predict migration_flow.

Assign each feature on the following list a penalty score between 0.1 and 1.0 based on its importance for predicting migration flow.
A lower penalty (for example 0.1) means the feature is highly predictive and should be penalized less in Lasso.
A higher penalty (for example 1.0) means the feature is less likely to be strongly predictive and should be penalized more.

Focus on established migration theory or strong domain rationale, not speculation.
Expect most features to receive scores closer to 1.0 than to 0.1.
Do not use pair_id dummies or one-hot encoded corridor identifiers in scoring.
Return ONLY valid JSON in this exact format:
{{
  "scores": [
    {{"feature": "feature_name_1", "penalty": 0.85, "reason": "brief reason"}},
    {{"feature": "feature_name_2", "penalty": 0.30, "reason": "brief reason"}}
  ]
}}

Features:
{feature_block}
""".strip()



def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Could not find a JSON object in the LLM response.")
        return json.loads(match.group(0))



def query_ollama_json(model: str, prompt: str, host: str = "http://127.0.0.1:11434") -> dict:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    response_text = data.get("response", "")
    return extract_json_object(response_text)



def load_or_generate_llm_scores(
    numeric_features: List[str],
    scores_json_in: str | Path | None,
    output_dir: str | Path,
    ollama_model: str,
    ollama_host: str,
) -> Dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if scores_json_in is not None:
        with open(scores_json_in, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "scores" in raw:
            rows = raw["scores"]
            score_map = {str(r["feature"]): float(r["penalty"]) for r in rows}
        else:
            score_map = {str(k): float(v) for k, v in raw.items()}
    else:
        prompt = build_llm_prompt(numeric_features)
        raw = query_ollama_json(model=ollama_model, prompt=prompt, host=ollama_host)
        rows = raw.get("scores", [])
        score_map = {str(r["feature"]): float(r["penalty"]) for r in rows}
        with open(output_dir / "llm_feature_scores_raw.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    clean_scores = {}
    for feat in numeric_features:
        value = float(score_map.get(feat, 1.0))
        value = min(max(value, 0.1), 1.0)
        clean_scores[feat] = value

    pd.DataFrame({
        "feature": list(clean_scores.keys()),
        "llm_penalty_score": list(clean_scores.values()),
    }).to_csv(output_dir / "llm_feature_scores.csv", index=False)

    with open(output_dir / "llm_feature_scores.json", "w", encoding="utf-8") as f:
        json.dump(clean_scores, f, indent=2)

    return clean_scores


# -----------------------------------------------------------------------------
# Paper-style weighted Lasso
# -----------------------------------------------------------------------------


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])



def build_base_penalty_vector(
    transformed_feature_names: List[str],
    llm_numeric_scores: Dict[str, float],
) -> np.ndarray:
    base_penalties = []
    for feat in transformed_feature_names:
        if feat.startswith("num__"):
            raw_name = feat.replace("num__", "", 1)
            penalty = float(llm_numeric_scores.get(raw_name, 1.0))
        else:
            # paper uses feature names/no data; here corridor dummy columns are left neutral
            penalty = 1.0
        base_penalties.append(penalty)
    return np.asarray(base_penalties, dtype=float)



def apply_penalty_transform(X: np.ndarray, penalties_v: np.ndarray, eta: int, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper-style inverse importance family:
        tau(V)_j = v_j ** eta
    with eta = 0 giving plain Lasso because all penalties become 1.

    To fit weighted Lasso with ordinary sklearn LassoCV, rescale columns:
        X_tilde_j = X_j / w_j
    where w_j is the final penalty factor.
    """
    w = np.power(np.maximum(penalties_v, eps), eta)
    X_tilde = X / w
    return X_tilde, w



def fit_paper_style_llm_lasso(
    df: pd.DataFrame,
    output_dir: str | Path,
    llm_scores_json_in: str | Path | None = None,
    ollama_model: str = "qwen2.5:7b",
    ollama_host: str = "http://127.0.0.1:11434",
    eta_max: int = 10,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_features, categorical_features = choose_feature_columns(df)
    if not numeric_features and not categorical_features:
        raise ValueError("No usable features were found. Add numeric variables to the panel CSV.")

    X_df = df[numeric_features + categorical_features].copy()
    y = df["log_migration_flow"].astype(float).to_numpy()

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_proc = preprocessor.fit_transform(X_df)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    transformed_feature_names = list(preprocessor.get_feature_names_out())

    llm_scores = load_or_generate_llm_scores(
        numeric_features=numeric_features,
        scores_json_in=llm_scores_json_in,
        output_dir=output_dir,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
    )
    base_penalties_v = build_base_penalty_vector(transformed_feature_names, llm_scores)

    n_rows = len(df)
    outer_folds = max(2, min(5, n_rows // 3 if n_rows >= 6 else 2))
    inner_folds = outer_folds
    eta_grid = list(range(0, eta_max + 1))

    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    eta_results = []

    for eta in eta_grid:
        fold_mse = []
        fold_mae = []

        for train_idx, val_idx in outer_cv.split(X_proc):
            X_train = X_proc[train_idx]
            X_val = X_proc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            X_train_tilde, w = apply_penalty_transform(X_train, base_penalties_v, eta=eta)
            X_val_tilde = X_val / w

            inner_model = LassoCV(
                cv=inner_folds,
                random_state=42,
                max_iter=50000,
                n_alphas=200,
            )
            inner_model.fit(X_train_tilde, y_train)
            val_pred = inner_model.predict(X_val_tilde)

            fold_mse.append(mean_squared_error(y_val, val_pred))
            fold_mae.append(mean_absolute_error(y_val, val_pred))

        eta_results.append({
            "eta": eta,
            "cv_mse": float(np.mean(fold_mse)),
            "cv_mae": float(np.mean(fold_mae)),
        })

    eta_df = pd.DataFrame(eta_results).sort_values(["cv_mse", "cv_mae"], ascending=True)
    best_eta = int(eta_df.iloc[0]["eta"])

    X_tilde_full, final_w = apply_penalty_transform(X_proc, base_penalties_v, eta=best_eta)
    final_model = LassoCV(
        cv=inner_folds,
        random_state=42,
        max_iter=50000,
        n_alphas=200,
    )
    final_model.fit(X_tilde_full, y)

    pred_log = final_model.predict(X_tilde_full)
    pred_flow = np.expm1(pred_log)

    theta = final_model.coef_
    beta = theta / final_w

    coef_df = pd.DataFrame({
        "feature": transformed_feature_names,
        "base_penalty_v": base_penalties_v,
        "best_eta": best_eta,
        "final_penalty_weight": np.power(base_penalties_v, best_eta),
        "coefficient": beta,
        "abs_coefficient": np.abs(beta),
    }).sort_values("abs_coefficient", ascending=False)

    pred_df = df.copy()
    pred_df["pred_log_migration_flow"] = pred_log
    pred_df["pred_migration_flow"] = pred_flow
    pred_df["residual_log"] = pred_df["log_migration_flow"] - pred_df["pred_log_migration_flow"]

    metrics = {
        "rows": int(len(df)),
        "outer_cv_folds": int(outer_folds),
        "inner_cv_folds": int(inner_folds),
        "best_eta": int(best_eta),
        "alpha": float(final_model.alpha_),
        "r2_train": float(r2_score(y, pred_log)),
        "rmse_train_log": float(np.sqrt(mean_squared_error(y, pred_log))),
        "mae_train_log": float(mean_absolute_error(y, pred_log)),
        "numeric_features_scored_by_llm": numeric_features,
        "categorical_features_neutral_penalty": categorical_features,
    }

    pred_df.to_csv(output_dir / "llm_lasso_predictions.csv", index=False)
    coef_df.to_csv(output_dir / "llm_lasso_coefficients.csv", index=False)
    eta_df.to_csv(output_dir / "llm_lasso_eta_cv.csv", index=False)
    with open(output_dir / "llm_lasso_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(f"  {output_dir / 'llm_feature_scores.csv'}")
    print(f"  {output_dir / 'llm_lasso_predictions.csv'}")
    print(f"  {output_dir / 'llm_lasso_coefficients.csv'}")
    print(f"  {output_dir / 'llm_lasso_eta_cv.csv'}")
    print(f"  {output_dir / 'llm_lasso_metrics.json'}")
    print("\nBest eta:", best_eta)
    print("\nTop coefficients:")
    print(coef_df.head(20).to_string(index=False))
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-style LLM-Lasso for migration flows: builds Model-4-like features, "
            "gets LLM penalty scores, applies the inverse-importance family, and "
            "cross-validates eta like the LLM-Lasso paper."
        )
    )
    parser.add_argument(
        "--panel_csv",
        required=True,
        help=(
            "CSV with at least: origin, destination, migration_flow. Optional columns include "
            "origin_population, dest_population, push_ratio, pull_ratio, mixed_ratio, unclear_ratio, etc."
        ),
    )
    parser.add_argument(
        "--country_csv",
        default="HDI_GPI_with_GDP_LATLON.csv",
        help="Path to country feature CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="migration_llm_lasso_output",
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--llm_scores_json_in",
        default=None,
        help=(
            "Optional JSON file of precomputed LLM penalty scores. If omitted, the script queries Ollama. "
            "Accepted formats: {feature: penalty} or {'scores': [{'feature': ..., 'penalty': ...}, ...]}"
        ),
    )
    parser.add_argument(
        "--ollama_model",
        default="qwen2.5:7b",
        help="Ollama model for feature scoring when --llm_scores_json_in is not provided.",
    )
    parser.add_argument(
        "--ollama_host",
        default="http://127.0.0.1:11434",
        help="Base URL for Ollama API.",
    )
    parser.add_argument(
        "--eta_max",
        type=int,
        default=10,
        help="Maximum eta in the inverse-importance family tau(V)_j = v_j ** eta.",
    )
    args = parser.parse_args()

    panel_df = pd.read_csv(args.panel_csv)
    country_df = load_country_features(args.country_csv)
    model_df = enrich_panel_with_country_features(panel_df, country_df)

    print("Using columns:")
    print(model_df.columns.tolist())

    fit_paper_style_llm_lasso(
        df=model_df,
        output_dir=args.output_dir,
        llm_scores_json_in=args.llm_scores_json_in,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        eta_max=args.eta_max,
    )


if __name__ == "__main__":
    main()
