
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "capacity_retention_trajectories.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "recent_trend_vs_persistence"
)

MIN_HISTORY = 20
RECENT_WINDOW = 20
ORIGIN_STRIDE = 5
HORIZONS = (5, 10)


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not identify the {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    return {
        "MAE_pp": float(np.mean(np.abs(errors))),
        "RMSE_pp": float(np.sqrt(np.mean(errors ** 2))),
        # Positive bias means overprediction of capacity retention.
        "bias_predicted_minus_actual_pp": float(np.mean(errors)),
        "n_forecasts": int(len(errors)),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    cell_col = find_column(
        df, ["cell_id", "cell", "battery_id"], "cell identifier"
    )
    index_col = find_column(
        df,
        ["discharge_index", "observation_index", "cycle_index",
         "discharge_observation", "cycle_number"],
        "discharge observation index",
    )
    target_col = find_column(
        df,
        ["relative_capacity_retention_pct",
         "capacity_retention_pct",
         "retention_pct",
         "capacity_retention_percent"],
        "capacity retention",
    )

    df = df[[cell_col, index_col, target_col]].copy()
    df.columns = ["cell_id", "observation_index", "retention_pct"]

    df["observation_index"] = pd.to_numeric(
        df["observation_index"], errors="coerce"
    )
    df["retention_pct"] = pd.to_numeric(
        df["retention_pct"], errors="coerce"
    )
    df = df.dropna()
    df = df.sort_values(["cell_id", "observation_index"])

    all_predictions = []

    for cell_id, cell in df.groupby("cell_id"):
        cell = cell.reset_index(drop=True)
        y = cell["retention_pct"].to_numpy(dtype=float)
        observation_indices = cell["observation_index"].to_numpy()

        if len(y) <= MIN_HISTORY:
            print(f"Skipping {cell_id}: insufficient observations.")
            continue

        # Origins are the number of observations already available.
        origins = range(
            MIN_HISTORY,
            len(y),
            ORIGIN_STRIDE,
        )

        for origin in origins:
            history_start = max(0, origin - RECENT_WINDOW)
            recent_y = y[history_start:origin]

            # x represents sequential observations within the
            # recent history, avoiding assumptions about cycle IDs.
            x_history = np.arange(len(recent_y), dtype=float)
            x_future = np.arange(
                len(recent_y),
                len(recent_y) + max(HORIZONS),
                dtype=float,
            )

            if len(recent_y) < 2:
                continue

            slope, intercept = np.polyfit(
                x_history, recent_y, deg=1
            )

            for horizon in HORIZONS:
                end = min(origin + horizon, len(y))

                # Only evaluate a complete forecast horizon.
                if end - origin != horizon:
                    continue

                actual_values = y[origin:end]
                target_indices = observation_indices[origin:end]

                trend_values = (
                    intercept + slope * x_future[:horizon]
                )

                persistence_values = np.full(
                    horizon, y[origin - 1], dtype=float
                )

                for method, predictions in [
                    ("persistence", persistence_values),
                    ("recent_20_trend", trend_values),
                ]:
                    for step in range(horizon):
                        all_predictions.append({
                            "cell_id": cell_id,
                            "forecast_origin": observation_indices[
                                origin - 1
                            ],
                            "horizon": horizon,
                            "step_ahead": step + 1,
                            "target_observation_index":
                                target_indices[step],
                            "method": method,
                            "actual_retention_pct":
                                actual_values[step],
                            "predicted_retention_pct":
                                predictions[step],
                            "error_predicted_minus_actual_pp":
                                predictions[step] - actual_values[step],
                        })

    if not all_predictions:
        raise ValueError(
            "No forecasts were generated. Check the input data "
            "and the identified column names."
        )

    predictions_df = pd.DataFrame(all_predictions)

    summary_rows = []

    for (cell_id, horizon, method), group in (
        predictions_df.groupby(["cell_id", "horizon", "method"])
    ):
        metrics = calculate_metrics(
            group["actual_retention_pct"],
            group["predicted_retention_pct"],
        )

        metrics.update({
            "cell_id": cell_id,
            "horizon": horizon,
            "method": method,
            "unique_target_observations":
                group["target_observation_index"].nunique(),
            "unique_forecast_origins":
                group["forecast_origin"].nunique(),
        })

        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "forecast_predictions.csv"
    summary_path = OUTPUT_DIR / "forecast_summary.csv"

    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nRECENT-TREND VS PERSISTENCE")
    print("=" * 65)
    print(f"Input: {INPUT_FILE}")
    print(f"Minimum history: {MIN_HISTORY} observations")
    print(f"Recent trend window: {RECENT_WINDOW} observations")
    print(f"Forecast horizons: {HORIZONS}")
    print("\nSummary (capacity-retention errors in percentage points):")

    print(
        summary_df[
            [
                "cell_id",
                "horizon",
                "method",
                "MAE_pp",
                "RMSE_pp",
                "bias_predicted_minus_actual_pp",
                "unique_forecast_origins",
                "unique_target_observations",
            ]
        ]
        .sort_values(["cell_id", "horizon", "method"])
        .round(3)
        .to_string(index=False)
    )

    print(f"\nPredictions saved: {predictions_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
