
"""Compare a time-trend baseline with early-discharge features."""
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
    / "trend_baseline"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "relative_capacity_retention_pct"

FEATURES = [
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

if df[[TARGET, "discharge_index"] + FEATURES].isna().any().any():
    raise ValueError("Missing values found in target, index, or features.")

split = int(len(df) * 0.70)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

y_train = train[TARGET]
y_test = test[TARGET]

# Baseline 1: predict the training-set mean for every test observation.
mean_predictions = [y_train.mean()] * len(test)

# Baseline 2: fit a straight-line retention trend against observation index.
trend_model = LinearRegression()
trend_model.fit(train[["discharge_index"]], y_train)
trend_predictions = trend_model.predict(test[["discharge_index"]])

# Reference: early-discharge voltage and temperature model.
feature_model = make_pipeline(
    StandardScaler(),
    LinearRegression(),
)
feature_model.fit(train[FEATURES], y_train)
feature_predictions = feature_model.predict(test[FEATURES])


def calculate_metrics(name, actual, predicted):
    return {
        "model": name,
        "MAE_pp": mean_absolute_error(actual, predicted),
        "RMSE_pp": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
        "mean_residual_pp": (actual.to_numpy() - predicted).mean(),
    }


metrics = pd.DataFrame([
    calculate_metrics("training_mean_baseline", y_test, mean_predictions),
    calculate_metrics("linear_time_trend", y_test, trend_predictions),
    calculate_metrics("voltage_temperature_features", y_test, feature_predictions),
]).sort_values("MAE_pp")

predictions = pd.DataFrame({
    "discharge_index": test["discharge_index"].to_numpy(),
    "actual_retention_pct": y_test.to_numpy(),
    "training_mean_predicted_pct": mean_predictions,
    "time_trend_predicted_pct": trend_predictions,
    "voltage_temperature_predicted_pct": feature_predictions,
})

metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
predictions.to_csv(OUT_DIR / "predictions.csv", index=False)

print("Chronological split: first 70% train, last 30% test")
print("Training observations:", len(train))
print("Test observations:", len(test))
print("\nMODEL COMPARISON")
print(metrics.round(4).to_string(index=False))
print("\nTime-trend slope (percentage points per observation):",
      round(float(trend_model.coef_[0]), 6))
print("\nSaved results to:", OUT_DIR)
