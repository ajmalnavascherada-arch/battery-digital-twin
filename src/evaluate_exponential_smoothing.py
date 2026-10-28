
"""Evaluate calibrated exponential smoothing for next-step retention."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "B0005": ROOT / "data" / "processed_nasa_original" / "B0005_cycle_summary.csv",
    "B0006": ROOT / "data" / "processed_B0006" / "B0006_cycle_summary.csv",
    "B0007": ROOT / "data" / "processed_B0007" / "B0007_cycle_summary.csv",
}

INITIAL_FRACTION = 0.20
ALPHAS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

OUT_DIR = (
    ROOT / "results_capacity_comparison"
    / "exponential_smoothing"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cell(cell_id, path):
    df = pd.read_csv(path)

    df = df.loc[
        df["cycle_type"].astype(str).str.lower().eq("discharge")
    ].copy()

    df["cycle_number"] = pd.to_numeric(
        df["cycle_number"], errors="coerce"
    )
    df["capacity_Ah"] = pd.to_numeric(
        df["capacity_Ah"], errors="coerce"
    )

    df = df.dropna(subset=["cycle_number", "capacity_Ah"])
    df = df.loc[df["capacity_Ah"] > 0]
    df = df.sort_values("cycle_number").reset_index(drop=True)

    initial_capacity = df["capacity_Ah"].iloc[0]
    df["retention_pct"] = 100 * df["capacity_Ah"] / initial_capacity
    df["observation_index"] = range(1, len(df) + 1)
    df["cell_id"] = cell_id

    return df[
        ["cell_id", "cycle_number", "observation_index", "retention_pct"]
    ]


def metric_row(cell_id, model_name, actual, predicted, alpha=None):
    return {
        "cell_id": cell_id,
        "model": model_name,
        "alpha": alpha,
        "test_observations": len(actual),
        "MAE_pp": mean_absolute_error(actual, predicted),
        "RMSE_pp": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
        "mean_residual_pp": np.mean(
            np.asarray(actual) - np.asarray(predicted)
        ),
    }


metrics = []
predictions_output = []

for cell_id, path in FILES.items():
    df = load_cell(cell_id, path)

    calibration_count = max(3, int(len(df) * INITIAL_FRACTION))
    calibration = df.iloc[:calibration_count]
    test = df.iloc[calibration_count:]

    # Choose alpha using one-step-ahead validation within calibration only.
    alpha_scores = {}

    for alpha in ALPHAS:
        calibration_predictions = []
        calibration_actual = []

        for position in range(1, len(calibration)):
            history = calibration["retention_pct"].iloc[:position]
            prediction = float(history.iloc[-1])

            # One-step exponential smoothing update:
            # forecast = alpha * latest actual + (1-alpha) * previous forecast.
            forecast = float(history.iloc[0])
            for value in history.iloc[1:]:
                forecast = (
                    alpha * float(value)
                    + (1 - alpha) * forecast
                )

            calibration_predictions.append(forecast)
            calibration_actual.append(
                float(calibration["retention_pct"].iloc[position])
            )

        alpha_scores[alpha] = mean_absolute_error(
            calibration_actual, calibration_predictions
        )

    best_alpha = min(alpha_scores, key=alpha_scores.get)

    # Initialize the forecast state using only calibration observations.
    forecast_state = float(calibration["retention_pct"].iloc[0])
    for value in calibration["retention_pct"].iloc[1:]:
        forecast_state = (
            best_alpha * float(value)
            + (1 - best_alpha) * forecast_state
        )

    smooth_predictions = []
    persistence_predictions = []
    actual_test = []

    # One-step ahead: predict first, then update with the newly observed value.
    for _, row in test.iterrows():
        smooth_predictions.append(forecast_state)
        persistence_predictions.append(
            float(df.loc[row.name - 1, "retention_pct"])
        )
        actual_test.append(float(row["retention_pct"]))

        forecast_state = (
            best_alpha * float(row["retention_pct"])
            + (1 - best_alpha) * forecast_state
        )

    metrics.append(
        metric_row(
            cell_id,
            "exponential_smoothing",
            actual_test,
            smooth_predictions,
            best_alpha,
        )
    )
    metrics.append(
        metric_row(
            cell_id,
            "persistence",
            actual_test,
            persistence_predictions,
        )
    )

    for index, actual, smooth, persist in zip(
        test["observation_index"],
        actual_test,
        smooth_predictions,
        persistence_predictions,
    ):
        predictions_output.append({
            "cell_id": cell_id,
            "observation_index": int(index),
            "actual_retention_pct": actual,
            "exponential_smoothing_predicted_pct": smooth,
            "persistence_predicted_pct": persist,
            "residual_smoothing_pp": actual - smooth,
            "residual_persistence_pp": actual - persist,
        })

    print(
        f"{cell_id}: calibration={len(calibration)}, test={len(test)}, "
        f"selected alpha={best_alpha}, "
        f"calibration MAE by alpha={alpha_scores}"
    )

metrics_df = pd.DataFrame(metrics)
predictions_df = pd.DataFrame(predictions_output)

metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
predictions_df.to_csv(OUT_DIR / "predictions.csv", index=False)

print("\nEXPONENTIAL SMOOTHING RESULTS")
print(metrics_df.round(4).to_string(index=False))
print("\nSaved results to:", OUT_DIR)
