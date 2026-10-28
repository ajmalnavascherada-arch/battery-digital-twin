
"""Evaluate 5- and 10-observation-ahead capacity forecasts.

At each forecast origin, models use only observations available up to
that point. Forecast horizons are evaluated separately for each cell.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "B0005": ROOT / "data" / "processed_nasa_original" / "B0005_cycle_summary.csv",
    "B0006": ROOT / "data" / "processed_B0006" / "B0006_cycle_summary.csv",
    "B0007": ROOT / "data" / "processed_B0007" / "B0007_cycle_summary.csv",
}

HORIZONS = [5, 10]
INITIAL_FRACTION = 0.20

OUT_DIR = (
    ROOT / "results_capacity_comparison"
    / "multistep_forecasting"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cell(cell_id, path):
    df = pd.read_csv(path)

    required = {"cycle_type", "cycle_number", "capacity_Ah"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

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

    if len(df) < 20:
        raise ValueError(
            f"Too few valid discharge observations for {cell_id}: {len(df)}"
        )

    initial_capacity = float(df["capacity_Ah"].iloc[0])
    df["retention_pct"] = (
        100.0 * df["capacity_Ah"] / initial_capacity
    )
    df["observation_index"] = np.arange(1, len(df) + 1)
    df["cell_id"] = cell_id

    return df[
        ["cell_id", "cycle_number", "observation_index", "retention_pct"]
    ]


all_predictions = []
all_metrics = []

for cell_id, path in FILES.items():
    df = load_cell(cell_id, path)
    n = len(df)

    # Require an initial history before the first forecast.
    initial_history = max(5, int(n * INITIAL_FRACTION))

    for horizon in HORIZONS:
        records = []

        # origin_position is the last observed row available to the model.
        # The target is exactly `horizon` observations after that origin.
        for origin_position in range(
            initial_history - 1,
            n - horizon,
        ):
            history = df.iloc[:origin_position + 1]
            target_position = origin_position + horizon

            origin_index = int(
                df.iloc[origin_position]["observation_index"]
            )
            target_row = df.iloc[target_position]

            y_history = history["retention_pct"].to_numpy()
            x_history = history[["observation_index"]]
            x_target = pd.DataFrame({
                "observation_index": [
                    int(target_row["observation_index"])
                ]
            })

            # Baseline: retention remains at the latest observed value.
            persistence_prediction = float(y_history[-1])

            # Fit only on history available at this forecast origin.
            trend_model = LinearRegression()
            trend_model.fit(x_history, y_history)
            trend_prediction = float(
                trend_model.predict(x_target)[0]
            )

            records.append({
                "cell_id": cell_id,
                "horizon": horizon,
                "origin_index": origin_index,
                "target_index": int(
                    target_row["observation_index"]
                ),
                "target_cycle_number": float(
                    target_row["cycle_number"]
                ),
                "actual_retention_pct": float(
                    target_row["retention_pct"]
                ),
                "persistence_prediction_pct": persistence_prediction,
                "linear_trend_prediction_pct": trend_prediction,
            })

        predictions = pd.DataFrame(records)

        if predictions.empty:
            raise ValueError(
                f"No forecasts generated for {cell_id}, horizon {horizon}"
            )

        all_predictions.append(predictions)

        for model_name, prediction_column in [
            ("persistence", "persistence_prediction_pct"),
            ("linear_trend", "linear_trend_prediction_pct"),
        ]:
            actual = predictions["actual_retention_pct"].to_numpy()
            predicted = predictions[prediction_column].to_numpy()

            all_metrics.append({
                "cell_id": cell_id,
                "horizon": horizon,
                "model": model_name,
                "forecast_count": len(predictions),
                "MAE_pp": mean_absolute_error(actual, predicted),
                "RMSE_pp": mean_squared_error(actual, predicted) ** 0.5,
                "R2": r2_score(actual, predicted),
                "mean_residual_pp": float(
                    np.mean(actual - predicted)
                ),
            })

    print(
        f"{cell_id}: observations={n}, "
        f"initial history={initial_history}"
    )

metrics_df = pd.DataFrame(all_metrics)
predictions_df = pd.concat(all_predictions, ignore_index=True)

metrics_path = OUT_DIR / "metrics.csv"
predictions_path = OUT_DIR / "predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

print("\nMULTI-STEP FORECAST RESULTS")
print(
    metrics_df[
        [
            "cell_id",
            "horizon",
            "model",
            "forecast_count",
            "MAE_pp",
            "RMSE_pp",
            "R2",
            "mean_residual_pp",
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved metrics to:", metrics_path)
print("Saved predictions to:", predictions_path)
