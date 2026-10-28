
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
    / "nonoverlapping_validation"
)

MIN_HISTORY = 20
RECENT_WINDOW = 20
HORIZONS = (5, 10)


def find_column(df, candidates, description):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Cannot identify {description}.\n"
        f"Available columns: {list(df.columns)}"
    )


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual

    return {
        "MAE_pp": np.mean(np.abs(errors)),
        "RMSE_pp": np.sqrt(np.mean(errors ** 2)),
        "bias_pp": np.mean(errors),
        "n_predictions": len(errors),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    cell_col = find_column(
        df, ["cell_id", "cell", "battery_id"], "cell identifier"
    )
    index_col = find_column(
        df,
        [
            "discharge_index",
            "observation_index",
            "cycle_index",
            "discharge_observation",
            "cycle_number",
        ],
        "observation index",
    )
    target_col = find_column(
        df,
        [
            "relative_capacity_retention_pct",
            "capacity_retention_pct",
            "retention_pct",
            "capacity_retention_percent",
        ],
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

    prediction_rows = []

    for cell_id, cell in df.groupby("cell_id"):
        cell = cell.reset_index(drop=True)
        y = cell["retention_pct"].to_numpy(dtype=float)
        obs_indices = cell["observation_index"].to_numpy()

        for horizon in HORIZONS:
            # Start with MIN_HISTORY observations available.
            # Advance by the entire horizon, so test blocks
            # do not overlap within this evaluation.
            for origin in range(
                MIN_HISTORY, len(y) - horizon + 1, horizon
            ):
                history = y[
                    max(0, origin - RECENT_WINDOW):origin
                ]

                x_history = np.arange(len(history), dtype=float)
                x_future = np.arange(
                    len(history),
                    len(history) + horizon,
                    dtype=float,
                )

                slope, intercept = np.polyfit(
                    x_history, history, deg=1
                )

                trend_prediction = intercept + slope * x_future
                persistence_prediction = np.full(
                    horizon, y[origin - 1], dtype=float
                )

                actual = y[origin:origin + horizon]

                for method, predicted in [
                    ("persistence", persistence_prediction),
                    ("recent_20_trend", trend_prediction),
                ]:
                    for step in range(horizon):
                        prediction_rows.append({
                            "cell_id": cell_id,
                            "horizon": horizon,
                            "forecast_origin": obs_indices[origin - 1],
                            "target_observation_index":
                                obs_indices[origin + step],
                            "step_ahead": step + 1,
                            "method": method,
                            "actual_retention_pct": actual[step],
                            "predicted_retention_pct": predicted[step],
                            "error_pp": predicted[step] - actual[step],
                        })

    if not prediction_rows:
        raise ValueError(
            "No predictions were generated. Check input columns "
            "and the number of observations per cell."
        )

    predictions = pd.DataFrame(prediction_rows)

    summary_rows = []

    for (cell_id, horizon, method), group in predictions.groupby(
        ["cell_id", "horizon", "method"]
    ):
        metrics = calculate_metrics(
            group["actual_retention_pct"],
            group["predicted_retention_pct"],
        )
        metrics.update({
            "cell_id": cell_id,
            "horizon": horizon,
            "method": method,
            "forecast_origins": group["forecast_origin"].nunique(),
            "unique_targets":
                group["target_observation_index"].nunique(),
        })
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "nonoverlapping_predictions.csv"
    summary_path = OUTPUT_DIR / "nonoverlapping_summary.csv"

    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nNON-OVERLAPPING FORECAST EVALUATION")
    print("=" * 72)
    print(f"Input: {INPUT_FILE}")
    print(f"Minimum history: {MIN_HISTORY}")
    print(f"Recent trend window: {RECENT_WINDOW}")
    print(f"Horizons: {HORIZONS}")
    print("\nMetrics (capacity-retention percentage points):")
    print(
        summary[
            [
                "cell_id",
                "horizon",
                "method",
                "MAE_pp",
                "RMSE_pp",
                "bias_pp",
                "forecast_origins",
                "unique_targets",
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
