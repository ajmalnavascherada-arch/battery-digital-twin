
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "B0005_fixed_time_features.csv"
)

OUTPUT_DIR = (
    ROOT / "results_capacity_comparison"
    / "early_feature_ridge_comparison"
)

FEATURES = [
    "first_300s_voltage_drop_V",
    "first_300s_temperature_rise_C",
]
TARGET = "relative_capacity_retention_pct"

INITIAL_TRAIN_FRACTION = 0.70
HORIZON = 5
STRIDE = 5


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values("discharge_index").reset_index(drop=True)

    required = ["discharge_index", TARGET, *FEATURES]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    df = df.reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    n = len(df)
    initial_train_size = int(np.floor(n * INITIAL_TRAIN_FRACTION))

    if initial_train_size + HORIZON > n:
        raise ValueError("Not enough observations for the chosen split.")

    prediction_rows = []

    for origin in range(
        initial_train_size, n - HORIZON + 1, STRIDE
    ):
        actual = y[origin:origin + HORIZON]

        model_predictions = {
            "persistence": np.full(HORIZON, y[origin - 1]),
        }

        for model_name, estimator in [
            ("linear_regression", LinearRegression()),
            ("ridge_alpha_1", Ridge(alpha=1.0)),
            ("ridge_alpha_10", Ridge(alpha=10.0)),
        ]:
            model = make_pipeline(
                StandardScaler(),
                estimator,
            )
            model.fit(X[:origin], y[:origin])
            model_predictions[model_name] = model.predict(
                X[origin:origin + HORIZON]
            )

        for model_name, predicted in model_predictions.items():
            for step, (actual_value, predicted_value) in enumerate(
                zip(actual, predicted), start=1
            ):
                prediction_rows.append({
                    "origin_index": origin,
                    "forecast_step": step,
                    "target_discharge_index": df.iloc[
                        origin + step - 1
                    ]["discharge_index"],
                    "model": model_name,
                    "actual_retention_pct": actual_value,
                    "predicted_retention_pct": predicted_value,
                    "residual_pp": actual_value - predicted_value,
                })

    predictions = pd.DataFrame(prediction_rows)
    summary_rows = []

    for model_name, group in predictions.groupby("model"):
        summary_rows.append({
            "model": model_name,
            "forecast_windows": group["origin_index"].nunique(),
            "forecast_points": len(group),
            "MAE_pp": mean_absolute_error(
                group["actual_retention_pct"],
                group["predicted_retention_pct"],
            ),
            "RMSE_pp": np.sqrt(mean_squared_error(
                group["actual_retention_pct"],
                group["predicted_retention_pct"],
            )),
            "mean_residual_pp": group["residual_pp"].mean(),
        })

    summary = pd.DataFrame(summary_rows).sort_values("MAE_pp")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)

    print("EARLY-FEATURE MODEL COMPARISON")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
