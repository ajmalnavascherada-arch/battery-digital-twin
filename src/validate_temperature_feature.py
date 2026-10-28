
"""Test whether the 300-second temperature feature is robust."""
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
    / "temperature_feature_validation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "relative_capacity_retention_pct"
TIME = "discharge_index"
TEMP = "first_300s_temperature_rise_C"

df = (
    pd.read_csv(DATA_FILE)
    .sort_values(TIME)
    .reset_index(drop=True)
)

if df[[TARGET, TIME, TEMP]].isna().any().any():
    raise ValueError("Missing values found in required columns.")

# Each split trains on earlier observations and tests on later ones.
# Test periods are different; no future rows are used for training.
splits = [
    (0.50, 0.70),
    (0.60, 0.80),
    (0.70, 1.00),
]

results = []
predictions = []

for split_number, (train_fraction, test_fraction) in enumerate(splits, start=1):
    train_end = int(len(df) * train_fraction)
    test_end = int(len(df) * test_fraction)

    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end:test_end].copy()

    if train.empty or test.empty:
        continue

    y_train = train[TARGET]
    y_test = test[TARGET]

    candidates = {
        "trend_only": [TIME],
        "trend_plus_300s_temperature": [TIME, TEMP],
    }

    for model_name, columns in candidates.items():
        model = make_pipeline(
            StandardScaler(),
            LinearRegression(),
        )
        model.fit(train[columns], y_train)
        predicted = model.predict(test[columns])

        results.append({
            "split": split_number,
            "train_fraction": train_fraction,
            "test_fraction_end": test_fraction,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_start_index": test[TIME].iloc[0],
            "test_end_index": test[TIME].iloc[-1],
            "model": model_name,
            "MAE_pp": mean_absolute_error(y_test, predicted),
            "RMSE_pp": mean_squared_error(y_test, predicted) ** 0.5,
            "R2": r2_score(y_test, predicted),
        })

        for idx, actual, estimate in zip(
            test[TIME], y_test, predicted
        ):
            predictions.append({
                "split": split_number,
                "model": model_name,
                "discharge_index": idx,
                "actual_retention_pct": actual,
                "predicted_retention_pct": estimate,
            })

results_df = pd.DataFrame(results)
predictions_df = pd.DataFrame(predictions)

results_df.to_csv(OUT_DIR / "split_metrics.csv", index=False)
predictions_df.to_csv(OUT_DIR / "predictions.csv", index=False)

print("EXPANDING CHRONOLOGICAL VALIDATION")
print(results_df.round(4).to_string(index=False))

print("\nMAE BY SPLIT")
print(
    results_df.pivot(
        index="split", columns="model", values="MAE_pp"
    ).round(4).to_string()
)

print("\nSaved results to:", OUT_DIR)
