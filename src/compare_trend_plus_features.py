
"""Test whether early-discharge features add value beyond time trend."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
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
    / "trend_plus_features"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "relative_capacity_retention_pct"

FEATURES = [
    "discharge_index",
    "first_300s_voltage_drop_V",
    "first_600s_voltage_drop_V",
    "first_300s_temperature_rise_C",
    "first_600s_temperature_rise_C",
]

df = (
    pd.read_csv(DATA_FILE)
    .sort_values("discharge_index")
    .reset_index(drop=True)
)

if df[FEATURES + [TARGET]].isna().any().any():
    raise ValueError("Missing values found in model inputs or target.")

split = int(len(df) * 0.70)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

y_train = train[TARGET]
y_test = test[TARGET]

# Model A: time trend only.
trend_model = LinearRegression()
trend_model.fit(train[["discharge_index"]], y_train)
trend_pred = trend_model.predict(test[["discharge_index"]])

# Model B: time trend plus early voltage and temperature features.
combined_model = make_pipeline(
    StandardScaler(),
    LinearRegression(),
)
combined_model.fit(train[FEATURES], y_train)
combined_pred = combined_model.predict(test[FEATURES])


def evaluate(name, actual, predicted):
    return {
        "model": name,
        "MAE_pp": mean_absolute_error(actual, predicted),
        "RMSE_pp": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
        "mean_residual_pp": (actual.to_numpy() - predicted).mean(),
    }


metrics = pd.DataFrame([
    evaluate("time_trend_only", y_test, trend_pred),
    evaluate("time_trend_plus_features", y_test, combined_pred),
])

predictions = pd.DataFrame({
    "discharge_index": test["discharge_index"].to_numpy(),
    "actual_retention_pct": y_test.to_numpy(),
    "time_trend_predicted_pct": trend_pred,
    "combined_predicted_pct": combined_pred,
})

metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
predictions.to_csv(OUT_DIR / "predictions.csv", index=False)

trend_mae = metrics.loc[
    metrics["model"] == "time_trend_only", "MAE_pp"
].iloc[0]
combined_mae = metrics.loc[
    metrics["model"] == "time_trend_plus_features", "MAE_pp"
].iloc[0]

improvement_pct = 100 * (trend_mae - combined_mae) / trend_mae

print("Chronological split: first 70% train, last 30% test")
print("Training observations:", len(train))
print("Test observations:", len(test))
print("\nMODEL COMPARISON")
print(metrics.round(4).to_string(index=False))
print(f"\nMAE improvement over time trend: {improvement_pct:.2f}%")
print("\nSaved results to:", OUT_DIR)
