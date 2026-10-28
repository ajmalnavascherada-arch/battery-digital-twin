
"""Chronological baseline evaluation for early SOH estimation."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "B0005_fixed_time_features.csv"
)
OUT_DIR = DATA.parent / "early_soh_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA).sort_values("discharge_index").reset_index(drop=True)

if len(df) != 168 or df["discharge_index"].nunique() != 168:
    raise ValueError("Expected 168 unique B0005 discharge records.")

target = "relative_capacity_retention_pct"

feature_sets = {
    "voltage_only": [
        "first_300s_voltage_drop_V",
        "first_600s_voltage_drop_V",
    ],
    "voltage_plus_temperature": [
        "first_300s_voltage_drop_V",
        "first_600s_voltage_drop_V",
        "first_300s_temperature_rise_C",
        "first_600s_temperature_rise_C",
    ],
}

# First 70% for training; latest 30% for testing.
split = int(len(df) * 0.70)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

y_train = train[target].to_numpy()
y_test = test[target].to_numpy()

results = [{
    "model": "persistence_100pct",
    "features": "none",
    "train_records": len(train),
    "test_records": len(test),
    "MAE_pp": mean_absolute_error(
        y_test, np.full(len(y_test), 100.0)
    ),
    "RMSE_pp": np.sqrt(mean_squared_error(
        y_test, np.full(len(y_test), 100.0)
    )),
    "R2": r2_score(y_test, np.full(len(y_test), 100.0)),
}]

prediction_frames = []

for name, columns in feature_sets.items():
    X_train = train[columns]
    X_test = test[columns]

    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "model": name,
        "features": "; ".join(columns),
        "train_records": len(train),
        "test_records": len(test),
        "MAE_pp": mean_absolute_error(y_test, predictions),
        "RMSE_pp": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    })

    prediction_frames.append(pd.DataFrame({
        "discharge_index": test["discharge_index"].to_numpy(),
        "actual_retention_pct": y_test,
        "predicted_retention_pct": predictions,
        "model": name,
    }))

results_df = pd.DataFrame(results)
results_path = OUT_DIR / "chronological_metrics.csv"
results_df.to_csv(results_path, index=False)

predictions_path = OUT_DIR / "chronological_predictions.csv"
pd.concat(prediction_frames, ignore_index=True).to_csv(
    predictions_path, index=False
)

print("Chronological evaluation completed.")
print("Training records:", len(train))
print("Test records:", len(test))
print("Training discharge indices:",
      int(train["discharge_index"].min()), "to",
      int(train["discharge_index"].max()))
print("Test discharge indices:",
      int(test["discharge_index"].min()), "to",
      int(test["discharge_index"].max()))

print("\nMetrics (errors in percentage points):")
print(
    results_df[
        ["model", "train_records", "test_records", "MAE_pp", "RMSE_pp", "R2"]
    ].round(4).to_string(index=False)
)

print("\nSaved metrics:", results_path)
print("Saved predictions:", predictions_path)
