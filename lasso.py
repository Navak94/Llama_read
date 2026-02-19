#!/usr/bin/env python3
"""
run_llm_lasso.py

One-file pipeline for:
- building log/ln features (gravity-style)
- plain Lasso with CV
- optional LLM-Lasso-style weighted Lasso:
    minimize 0.5||y - Xb||^2 + lam * sum_j w_j |b_j|
  where LLM provides v_j (penalty factors or inverse-importance), and we cross-validate
  a transformation w_j = v_j**eta (eta grid; eta=0 => plain Lasso fallback).

Usage examples:

1) Plain Lasso:
python run_llm_lasso.py \
  --data flows.csv \
  --y_col flow \
  --log_cols pop_i pop_j dist gdp_i gdp_j \
  --keep_cols HDI_i HDI_j GPI_i GPI_j \
  --out_prefix results/plain

2) Weighted (LLM-Lasso-style) using LLM penalty scores file:
python run_llm_lasso.py \
  --data flows.csv \
  --y_col flow \
  --log_cols pop_i pop_j dist gdp_i gdp_j \
  --keep_cols HDI_i HDI_j GPI_i GPI_j \
  --llm_scores llm_penalties.csv \
  --eta_grid 0 1 2 3 4 \
  --out_prefix results/llm

3) Merge your HDI_GPI.csv first:
python run_llm_lasso.py \
  --data flows.csv \
  --merge_hdi_gpi HDI_GPI.csv \
  --merge_key country \
  --y_col flow \
  --log_cols pop_i pop_j dist gdp_i gdp_j \
  --keep_cols HDI_i HDI_j GPI_i GPI_j \
  --out_prefix results/merged
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold


def read_llm_scores(path: str) -> Dict[str, float]:
    """
    Reads LLM feature penalty factors v_j.
    Supported:
      - CSV with columns: feature, score
      - JSON dict: {"feature_name": score, ...}

    Returns dict: feature -> score (float, must be > 0)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LLM scores file not found: {path}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON must be an object mapping feature->score.")
        scores = {str(k): float(v) for k, v in data.items()}
    else:
        df = pd.read_csv(p)
        if not {"feature", "score"}.issubset(df.columns):
            raise ValueError("LLM scores CSV must have columns: feature, score")
        scores = {str(r["feature"]): float(r["score"]) for _, r in df.iterrows()}

    bad = [k for k, v in scores.items() if not np.isfinite(v) or v <= 0]
    if bad:
        raise ValueError(f"All LLM scores must be finite and > 0. Bad keys: {bad[:10]}")
    return scores


def safe_log_series(s: pd.Series, mode: str = "log1p") -> pd.Series:
    """
    mode:
      - log1p: log(1 + x) (works with zeros; requires x >= 0)
      - ln: log(x) (requires x > 0)
    """
    x = pd.to_numeric(s, errors="coerce")
    if mode == "log1p":
        if (x < 0).any():
            # you can change this behavior if you prefer to drop rows instead
            raise ValueError(f"log1p requested but found negative values in column {s.name}")
        return np.log1p(x)
    elif mode == "ln":
        if (x <= 0).any():
            raise ValueError(f"ln requested but found non-positive values in column {s.name}")
        return np.log(x)
    else:
        raise ValueError("mode must be 'log1p' or 'ln'")


def build_design_matrix(
    df: pd.DataFrame,
    y_col: str,
    log_cols: List[str],
    keep_cols: List[str],
    log_mode: str,
    dropna_y: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      X_df: feature matrix (dataframe)
      y: response series (float)
    """
    if y_col not in df.columns:
        raise KeyError(f"y_col '{y_col}' not found in data columns.")

    y = pd.to_numeric(df[y_col], errors="coerce")
    if dropna_y:
        mask = y.notna()
        df = df.loc[mask].copy()
        y = y.loc[mask].copy()

    X_parts = []

    # Logged columns
    for c in log_cols:
        if c not in df.columns:
            raise KeyError(f"log_cols column '{c}' not found.")
        X_parts.append(pd.Series(safe_log_series(df[c], mode=log_mode), name=f"log_{c}"))

    # Keep-as-is columns (e.g., HDI/GPI already in [0,1] or index scale)
    for c in keep_cols:
        if c not in df.columns:
            raise KeyError(f"keep_cols column '{c}' not found.")
        X_parts.append(pd.to_numeric(df[c], errors="coerce").rename(c))

    X_df = pd.concat(X_parts, axis=1)
    return X_df, y


def fit_plain_lasso_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
) -> Tuple[np.ndarray, float]:
    """
    Plain Lasso with standardization + imputation and LassoCV to pick alpha (lambda).
    Returns:
      coef (numpy array aligned to X.columns)
      alpha (float)
    """
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LassoCV(cv=cv_folds, random_state=random_state, n_alphas=100, max_iter=20000)),
        ]
    )
    pipe.fit(X, y)
    model: LassoCV = pipe.named_steps["model"]
    # Coefs are in standardized space, but sklearn reports them in original feature scale post-scaling in pipeline.
    return model.coef_.copy(), float(model.alpha_)


def fit_weighted_lasso_cv(
    X: pd.DataFrame,
    y: pd.Series,
    llm_scores: Dict[str, float],
    eta_grid: List[int],
    cv_folds: int,
    random_state: int,
) -> Dict[str, object]:
    """
    LLM-Lasso-style:
      - LLM provides v_j
      - transform to w_j = v_j**eta
      - do CV over eta, with inner LassoCV to pick alpha for each eta

    Implementation trick:
      Weighted L1 penalty with weights w_j is equivalent to standard lasso on X' = X / w
      then recover beta = beta' / w

    Returns dict with best_eta, best_alpha, best_coef, and per-eta summary.
    """
    # Build v vector aligned to features
    features = list(X.columns)
    v = np.ones(len(features), dtype=float)

    missing = []
    for i, f in enumerate(features):
        if f in llm_scores:
            v[i] = float(llm_scores[f])
        else:
            missing.append(f)

    # If user doesn't provide some features, default v=1 (neutral)
    # (This is safer than crashing; you can change to strict if you want)
    if missing:
        print(f"[warn] LLM scores missing for {len(missing)} features; defaulting those v_j to 1. "
              f"Example missing: {missing[:8]}", file=sys.stderr)

    # CV split
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # We'll evaluate each eta by average validation MSE using a nested approach:
    # For each eta:
    #   - transform weights w = v**eta
    #   - create scaled design X' = X / w
    #   - fit pipeline(impute+scale+LassoCV) on train
    #   - predict on val using recovered beta on original X
    # Note: We could also just use LassoCV on X' and predict in X' space; both are equivalent.
    results = []

    for eta in eta_grid:
        w = np.power(v, eta)  # eta=0 => all ones => plain Lasso fallback
        w = np.clip(w, 1e-12, np.inf)

        # Scale columns by 1/w (so high w => column shrinks => effectively penalized more)
        X_scaled = X.copy()
        X_scaled.loc[:, :] = X_scaled.values / w[None, :]

        fold_mse = []
        fold_alphas = []

        for train_idx, val_idx in kf.split(X_scaled):
            Xtr, Xva = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            ytr, yva = y.iloc[train_idx], y.iloc[val_idx]

            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("model", LassoCV(cv=5, random_state=random_state, n_alphas=100, max_iter=20000)),
                ]
            )
            pipe.fit(Xtr, ytr)
            model: LassoCV = pipe.named_steps["model"]

            # Predict in scaled space (consistent)
            yhat = pipe.predict(Xva)
            mse = float(np.mean((yva.values - yhat) ** 2))

            fold_mse.append(mse)
            fold_alphas.append(float(model.alpha_))

        results.append(
            {
                "eta": int(eta),
                "mean_val_mse": float(np.mean(fold_mse)),
                "std_val_mse": float(np.std(fold_mse)),
                "mean_alpha": float(np.mean(fold_alphas)),
            }
        )

    # Pick best eta
    best = min(results, key=lambda d: d["mean_val_mse"])
    best_eta = int(best["eta"])

    # Refit on full data with best eta to get final coefficients
    w_best = np.clip(np.power(v, best_eta), 1e-12, np.inf)
    X_best = X.copy()
    X_best.loc[:, :] = X_best.values / w_best[None, :]

    final_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LassoCV(cv=cv_folds, random_state=random_state, n_alphas=150, max_iter=30000)),
        ]
    )
    final_pipe.fit(X_best, y)
    final_model: LassoCV = final_pipe.named_steps["model"]

    # Coefs correspond to X_best. Recover coefficients for original X:
    # beta_original = beta_scaled / w
    beta_scaled = final_model.coef_.copy()
    beta_original = beta_scaled / w_best

    return {
        "best_eta": best_eta,
        "best_alpha": float(final_model.alpha_),
        "coef": beta_original,
        "results": results,
        "features": features,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Main CSV with flows and predictors.")
    ap.add_argument("--y_col", required=True, help="Response column name (e.g., flow or log_flow).")

    ap.add_argument("--log_cols", nargs="*", default=[], help="Columns to log-transform (gravity-style).")
    ap.add_argument("--keep_cols", nargs="*", default=[], help="Columns to keep as-is (e.g., HDI, GPI).")
    ap.add_argument("--log_mode", choices=["log1p", "ln"], default="log1p",
                    help="log1p is safer for zeros; ln requires strictly positive values.")

    ap.add_argument("--merge_hdi_gpi", default=None, help="Optional: path to HDI_GPI.csv to merge in.")
    ap.add_argument("--merge_key", default=None, help="Optional: key column to merge on (must exist in both).")
    ap.add_argument("--merge_how", default="left", choices=["left", "inner", "right"], help="Merge type.")

    ap.add_argument("--llm_scores", default=None,
                    help="Optional: CSV(feature,score) or JSON mapping feature->score for weighted Lasso.")
    ap.add_argument("--eta_grid", nargs="*", type=int, default=[0, 1, 2, 3],
                    help="Eta values to try. eta=0 => plain Lasso fallback.")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_prefix", required=True, help="Output prefix (folder/name) for CSV outputs.")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    # Optional merge
    if args.merge_hdi_gpi:
        if not args.merge_key:
            raise ValueError("--merge_key is required when using --merge_hdi_gpi")
        hdi = pd.read_csv(args.merge_hdi_gpi)
        if args.merge_key not in df.columns or args.merge_key not in hdi.columns:
            raise KeyError(f"merge_key '{args.merge_key}' must exist in both dataframes.")
        df = df.merge(hdi, on=args.merge_key, how=args.merge_how)

    X, y = build_design_matrix(
        df=df,
        y_col=args.y_col,
        log_cols=args.log_cols,
        keep_cols=args.keep_cols,
        log_mode=args.log_mode,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # ---- Plain Lasso ----
    plain_coef, plain_alpha = fit_plain_lasso_cv(X, y, cv_folds=args.cv_folds, random_state=args.seed)
    plain_out = pd.DataFrame({"feature": X.columns, "coef": plain_coef})
    plain_out.to_csv(f"{out_prefix}_plain_lasso_coefs.csv", index=False)

    summary = {
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "plain_lasso_alpha": float(plain_alpha),
    }

    # ---- Weighted (LLM-Lasso-style), optional ----
    if args.llm_scores:
        llm = read_llm_scores(args.llm_scores)
        wfit = fit_weighted_lasso_cv(
            X=X, y=y, llm_scores=llm, eta_grid=args.eta_grid, cv_folds=args.cv_folds, random_state=args.seed
        )
        w_out = pd.DataFrame({"feature": wfit["features"], "coef": wfit["coef"]})
        w_out.to_csv(f"{out_prefix}_weighted_lasso_coefs.csv", index=False)

        eta_out = pd.DataFrame(wfit["results"]).sort_values("eta")
        eta_out.to_csv(f"{out_prefix}_eta_cv_results.csv", index=False)

        summary.update(
            {
                "weighted_best_eta": int(wfit["best_eta"]),
                "weighted_best_alpha": float(wfit["best_alpha"]),
            }
        )

    # ---- Write summary ----
    with open(f"{out_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] Wrote:")
    print(f" - {out_prefix}_plain_lasso_coefs.csv")
    if args.llm_scores:
        print(f" - {out_prefix}_weighted_lasso_coefs.csv")
        print(f" - {out_prefix}_eta_cv_results.csv")
    print(f" - {out_prefix}_summary.json")


if __name__ == "__main__":
    main()
