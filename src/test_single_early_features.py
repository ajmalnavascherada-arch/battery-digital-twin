
"""Test individual early-discharge features beyond the time trend."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    / "single_feature_tests"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "relative_capacity_retention_pct"
TIME_FEATURE = "discharge_index"

EARLY_FEATURES = [
    "first_300s_voltage_drop_V",
    "first_600s_voltage_drop_V",
    "first_300s_temperature_rise_C",
    "first_600s_temperature_rise_C",
]

df = pd.read_csv(DATA_FILE).sort_values(TIME_FEATURE).reset_index(drop=True)

required = [TARGET, TIME_FEATURE] + EARLY_FEATURES
if df[required].isna().any().any():
    raise ValueError("Missing values found in required columns.")

split = int(len(df) * 0.70)
train = df.iloc[:split]
test = df.iloc[split:]

y_train = train[TARGET]
y_test = test[TARGET]

results = []
prediction_data = pd.DataFrame({
    TIME_FEATURE: test[TIME_FEATURE].to_numpy(),
    "actual_retention_pct": y_test.to_numpy(),
})

# Reference: time trend only.
candidate_models = {
    "time_trend_only": [TIME_FEATURE],
}

# Add one early feature at a time.
for feature in EARLY_FEATURES:
    candidate_models[f"time_trend_plus_{feature}"] = [
        TIME_FEATURE, feature
    ]

for name, columns in candidate_models.items():
    model = make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )

    model.fit(train[columns], y_train)
    predicted = model.predict(test[columns])

    results.append({
        "model": name,
        "features": ", ".join(columns),
        "MAE_pp": mean_absolute_error(y_test, predicted),
        "RMSE_pp": mean_squared_error(y_test, predicted) ** 0.5,
        "R2": r2_score(y_test, predicted),
        "mean_residual_pp": (y_test.to_numpy() - predicted).mean(),
    })

    prediction_data[f"{name}_predicted_pct"] = predicted

results_df = pd.DataFrame(results).sort_values("MAE_pp")
results_df.to_csv(OUT_DIR / "metrics.csv", index=False)
prediction_data.to_csv(OUT_DIR / "predictions.csv", index=False)

baseline_mae = results_df.loc[
    results_df["model"] == "time_trend_only", "MAE_pp"
].iloc[0]

results_df["MAE_change_vs_trend_pct"] = (
    100 * (results_df["MAE_pp"] - baseline_mae) / baseline_mae
)

print("Chronological split: first 70% train, last 30% test")
print("Training observations:", len(train))
print("Test observations:", len(test))
print("\nSINGLE-FEATURE MODEL COMPARISON")
print(
    results_df[
        ["model", "MAE_pp", "RMSE_pp", "R2",
         "mean_residual_pp", "MAE_change_vs_trend_pct"]
    ].round(4).to_string(index=False)
)
print("\nNegative MAE change means improvement over the time-trend baseline.")
print("Saved results to:", OUT_DIR)
