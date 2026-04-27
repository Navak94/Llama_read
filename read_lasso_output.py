import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. READ LASSO COEFFICIENT FILE
# -----------------------------
lasso_df = pd.read_csv("plain_lasso_coefficients(1).csv")

# Keep only nonzero / meaningful features
threshold = 1e-10
selected = lasso_df.loc[lasso_df["coefficient"].abs() > threshold, "feature"].tolist()

print("Selected Lasso features:")
print(selected)

# -----------------------------
# 2. MAP LASSO FEATURE NAMES
# -----------------------------
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

selected_features = [FEATURE_MAP[f] for f in selected if f in FEATURE_MAP]

print("\nMapped selected features:")
print(selected_features)

# -----------------------------
# 3. LOAD YOUR DATASET
# -----------------------------
df = pd.read_csv("migration_beta_output.csv")   # or your feature-built dataset

# If this file already has all feature columns, use it directly.
# Otherwise use your earlier feature-building script first.

# -----------------------------
# 4. BUILD X AND y
# -----------------------------
X = df[selected_features].fillna(0.0)
y = df["migration_flow"].astype(float)

# If you want gravity-style log target instead, use:
# y = np.log(df["migration_flow"].clip(lower=1e-9))

# -----------------------------
# 5. FIT REGULAR REGRESSION
# -----------------------------
model = LinearRegression()
model.fit(X, y)

betas = model.coef_
intercept = model.intercept_

# -----------------------------
# 6. PRINT FINAL FORMULA
# -----------------------------
print("\n===== POST-LASSO OLS FORMULA =====\n")
print(f"migration = {intercept:.6g}")
for feat, beta in zip(selected_features, betas):
    sign = "+" if beta >= 0 else "-"
    print(f"  {sign} ({abs(beta):.6g})*{feat}")

# -----------------------------
# 7. SAVE BETAS
# -----------------------------
beta_df = pd.DataFrame({
    "feature": ["intercept"] + selected_features,
    "beta": [intercept] + list(betas)
})
beta_df.to_csv("post_lasso_ols_betas.csv", index=False)

# -----------------------------
# 8. PREDICTIONS
# -----------------------------
df["predicted_migration_post_lasso_ols"] = model.predict(X)
df.to_csv("post_lasso_ols_predictions.csv", index=False)

print("\nSaved:")
print("  post_lasso_ols_betas.csv")
print("  post_lasso_ols_predictions.csv")