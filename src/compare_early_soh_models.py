
"""Compare early-SOH models using the same chronological split."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "B0005_fixed_time_features.csv"
)

OUT_DIR = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "early_soh_baseline"
    / "model_comparison"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "relative_capacity_retention_pct"
FEATURES = [
    "first_300s_voltage_drop_V",
    "first_600s_voltage_drop_V",
    "first_300s_temperature_rise_C",
    "first_600s_temperature_rise_C",
]

df = pd.read_csv(DATA_FILE).sort_values("discharge_index").reset_index(drop=True)

if df[FEATURES + [TARGET]].isna().any().any():
    raise ValueError("Missing values found in features or target.")

split = int(len(df) * 0.70)
train = df.iloc[:split]
test = df.iloc[split:]

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

models = {
    "linear_regression": make_pipeline(
        StandardScaler(), LinearRegression()
    ),
    "ridge_alpha_1": make_pipeline(
        StandardScaler(), Ridge(alpha=1.0)
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    ),
}

metrics = []
predictions = pd.DataFrame({
    "discharge_index": test["discharge_index"].to_numpy(),
    "actual_retention_pct": y_test.to_numpy(),
})

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics.append({
        "model": name,
        "MAE_pp": mean_absolute_error(y_test, pred),
        "RMSE_pp": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
        "mean_residual_pp": (y_test.to_numpy() - pred).mean(),
    })

    predictions[f"{name}_predicted_pct"] = pred

metrics_df = pd.DataFrame(metrics).sort_values("MAE_pp")
metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
predictions.to_csv(OUT_DIR / "predictions.csv", index=False)

print("Chronological split: first 70% train, last 30% test")
print("Training observations:", len(train))
print("Test observations:", len(test))
print("\nMODEL COMPARISON")
print(metrics_df.round(4).to_string(index=False))
print("\nSaved results to:", OUT_DIR)
