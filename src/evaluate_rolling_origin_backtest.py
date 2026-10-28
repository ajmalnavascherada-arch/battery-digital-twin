
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CELL_FILES = {
    "B0005": PROJECT_ROOT / "data/processed_nasa_original/B0005_cycle_summary.csv",
    "B0006": PROJECT_ROOT / "data/processed_B0006/B0006_cycle_summary.csv",
    "B0007": PROJECT_ROOT / "data/processed_B0007/B0007_cycle_summary.csv",
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_capacity_comparison"
    / "rolling_origin_backtest"
)

# Use the first 20% of each cell's observations as the minimum history.
INITIAL_TRAIN_FRACTION = 0.20

# Forecast horizons, measured in subsequent discharge observations.
HORIZONS = (5, 10)

# Advance the forecast origin by this many observations.
# A stride of 5 reduces overlap between adjacent forecast windows.
ORIGIN_STRIDE = 5


def load_capacity_data(cell_id, file_path):
    df = pd.read_csv(file_path)

    required = {"cycle_type", "capacity_Ah"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{cell_id}: missing required columns: {sorted(missing)}"
        )

    discharge = df.loc[
        df["cycle_type"].astype(str).str.lower().eq("discharge")
    ].copy()

    discharge["capacity_Ah"] = pd.to_numeric(
        discharge["capacity_Ah"], errors="coerce"
    )

    discharge = discharge.loc[
        np.isfinite(discharge["capacity_Ah"])
        & (discharge["capacity_Ah"] > 0)
    ].reset_index(drop=True)

    if len(discharge) < 20:
        raise ValueError(
            f"{cell_id}: only {len(discharge)} valid discharge observations."
        )

    # Normalize to the first valid discharge capacity for this cell.
    initial_capacity = discharge["capacity_Ah"].iloc[0]
    retention = 100.0 * discharge["capacity_Ah"].to_numpy() / initial_capacity

    return retention


def calculate_metrics(actual, predicted):
    residual = actual - predicted

    return {
        "MAE_pp": mean_absolute_error(actual, predicted),
        "RMSE_pp": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) >= 2
            else np.nan
        ),
        "mean_residual_pp": np.mean(residual),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_metrics = []

    for cell_id, file_path in CELL_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"{cell_id} summary not found: {file_path}"
            )

        retention = load_capacity_data(cell_id, file_path)
        n = len(retention)

        initial_train_size = max(
            10, int(np.floor(n * INITIAL_TRAIN_FRACTION))
        )

        print(
            f"{cell_id}: {n} observations; "
            f"minimum history={initial_train_size}"
        )

        for horizon in HORIZONS:
            # Every origin forecasts the next complete horizon.
            # The history expands as the origin moves forward.
            origins = range(
                initial_train_size,
                n - horizon + 1,
                ORIGIN_STRIDE,
            )

            for origin in origins:
                train_y = retention[:origin]
                actual = retention[origin:origin + horizon]

                # Persistence: carry the latest observed retention forward.
                persistence_prediction = np.full(
                    horizon, train_y[-1], dtype=float
                )

                # Linear trend: fit on all observations available at origin.
                x_train = np.arange(origin).reshape(-1, 1)
                x_future = np.arange(
                    origin, origin + horizon
                ).reshape(-1, 1)

                model = LinearRegression()
                model.fit(x_train, train_y)
                trend_prediction = model.predict(x_future)

                model_predictions = {
                    "persistence": persistence_prediction,
                    "linear_trend": trend_prediction,
                }

                for model_name, predicted in model_predictions.items():
                    metrics = calculate_metrics(actual, predicted)

                    all_metrics.append({
                        "cell_id": cell_id,
                        "horizon": horizon,
                        "origin_index": origin,
                        "history_size": origin,
                        "model": model_name,
                        **metrics,
                    })

                    for step, (actual_value, predicted_value) in enumerate(
                        zip(actual, predicted), start=1
                    ):
                        all_predictions.append({
                            "cell_id": cell_id,
                            "horizon": horizon,
                            "origin_index": origin,
                            "history_size": origin,
                            "forecast_step": step,
                            "model": model_name,
                            "actual_retention_pct": actual_value,
                            "predicted_retention_pct": predicted_value,
                            "residual_pp": actual_value - predicted_value,
                        })

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)

    # Summarize across rolling origins. Each row here is one origin/horizon
    # window; errors are averaged equally across windows.
    summary_rows = []

    for (cell_id, horizon, model_name), group in metrics_df.groupby(
        ["cell_id", "horizon", "model"]
    ):
        summary_rows.append({
            "cell_id": cell_id,
            "horizon": horizon,
            "model": model_name,
            "forecast_windows": len(group),
            "mean_window_MAE_pp": group["MAE_pp"].mean(),
            "mean_window_RMSE_pp": group["RMSE_pp"].mean(),
            "mean_window_R2": group["R2"].mean(),
            "mean_bias_pp": group["mean_residual_pp"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Also calculate pooled errors across all forecasted points.
    pooled_rows = []

    for (cell_id, horizon, model_name), group in predictions_df.groupby(
        ["cell_id", "horizon", "model"]
    ):
        pooled_rows.append({
            "cell_id": cell_id,
            "horizon": horizon,
            "model": model_name,
            "forecast_windows": group["origin_index"].nunique(),
            "pooled_MAE_pp": mean_absolute_error(
                group["actual_retention_pct"],
                group["predicted_retention_pct"],
            ),
            "pooled_RMSE_pp": np.sqrt(mean_squared_error(
                group["actual_retention_pct"],
                group["predicted_retention_pct"],
            )),
            "pooled_bias_pp": group["residual_pp"].mean(),
        })

    pooled_df = pd.DataFrame(pooled_rows)

    metrics_path = OUTPUT_DIR / "window_metrics.csv"
    summary_path = OUTPUT_DIR / "summary_metrics.csv"
    pooled_path = OUTPUT_DIR / "pooled_metrics.csv"
    predictions_path = OUTPUT_DIR / "predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pooled_df.to_csv(pooled_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nROLLING-ORIGIN BACKTEST: MEAN METRICS PER FORECAST WINDOW")
    print(summary_df.round(4).to_string(index=False))

    print("\nROLLING-ORIGIN BACKTEST: POOLED FORECAST-POINT METRICS")
    print(pooled_df.round(4).to_string(index=False))

    print(f"\nSaved window metrics: {metrics_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(f"Saved pooled metrics: {pooled_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
