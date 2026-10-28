
"""Evaluate fixed-origin forecasts for NASA battery capacity retention."""
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

INITIAL_FRACTION = 0.20
HORIZONS = [5, 10]

OUT_DIR = (
    ROOT / "results_capacity_comparison"
    / "fixed_origin_forecasting"
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

    if len(df) < 20:
        raise ValueError(f"Too few observations for {cell_id}")

    initial_capacity = float(df["capacity_Ah"].iloc[0])
    df["retention_pct"] = (
        100.0 * df["capacity_Ah"] / initial_capacity
    )
    df["observation_index"] = np.arange(1, len(df) + 1)
    df["cell_id"] = cell_id

    return df[
        ["cell_id", "cycle_number", "observation_index", "retention_pct"]
    ]


metrics = []
prediction_rows = []

for cell_id, path in FILES.items():
    df = load_cell(cell_id, path)
    calibration_count = max(5, int(len(df) * INITIAL_FRACTION))

    history = df.iloc[:calibration_count]
    future = df.iloc[calibration_count:]

    if len(future) < max(HORIZONS):
        raise ValueError(f"Insufficient future observations for {cell_id}")

    # Fit once at the forecast origin. No future data enter model fitting.
    model = LinearRegression()
    model.fit(
        history[["observation_index"]],
        history["retention_pct"],
    )

    for horizon in HORIZONS:
        target = future.iloc[:horizon].copy()

        trend_predictions = model.predict(
            target[["observation_index"]]
        )

        # Persistence keeps the last observed calibration retention constant.
        persistence_predictions = np.repeat(
            float(history["retention_pct"].iloc[-1]),
            horizon,
        )

        actual = target["retention_pct"].to_numpy()

        for model_name, predicted in [
            ("persistence", persistence_predictions),
            ("linear_trend", trend_predictions),
        ]:
            metrics.append({
                "cell_id": cell_id,
                "horizon": horizon,
                "model": model_name,
                "calibration_observations": len(history),
                "forecast_observations": horizon,
                "forecast_start_index": int(
                    target["observation_index"].iloc[0]
                ),
                "forecast_end_index": int(
                    target["observation_index"].iloc[-1]
                ),
                "MAE_pp": mean_absolute_error(actual, predicted),
                "RMSE_pp": mean_squared_error(actual, predicted) ** 0.5,
                "R2": (
                    r2_score(actual, predicted)
                    if len(actual) > 1
                    else np.nan
                ),
                "mean_residual_pp": float(
                    np.mean(actual - predicted)
                ),
            })

            for (_, row), estimate in zip(
                target.iterrows(), predicted
            ):
                prediction_rows.append({
                    "cell_id": cell_id,
                    "horizon": horizon,
                    "model": model_name,
                    "observation_index": int(row["observation_index"]),
                    "actual_retention_pct": float(row["retention_pct"]),
                    "predicted_retention_pct": float(estimate),
                    "residual_pp": float(row["retention_pct"] - estimate),
                })

    print(
        f"{cell_id}: calibration={len(history)}, "
        f"remaining observations={len(future)}"
    )

metrics_df = pd.DataFrame(metrics)
predictions_df = pd.DataFrame(prediction_rows)

metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
predictions_df.to_csv(OUT_DIR / "predictions.csv", index=False)

print("\nFIXED-ORIGIN FORECAST RESULTS")
print(
    metrics_df[
        [
            "cell_id",
            "horizon",
            "model",
            "MAE_pp",
            "RMSE_pp",
            "R2",
            "mean_residual_pp",
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved metrics to:", OUT_DIR / "metrics.csv")
print("Saved predictions to:", OUT_DIR / "predictions.csv")
