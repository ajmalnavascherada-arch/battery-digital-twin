
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
    / "nonoverlapping_rul_validation"
)

EOL_THRESHOLD_PCT = 80.0
MIN_HISTORY = 20
RECENT_WINDOW = 20
HORIZONS = (5, 10)


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Cannot identify {description}.\n"
        f"Available columns: {list(df.columns)}"
    )


def estimate_eol(history, origin, method):
    """Return predicted EOL observation index, or NaN if unavailable."""
    y = np.asarray(history, dtype=float)

    if len(y) < 2 or not np.all(np.isfinite(y)):
        return np.nan

    if method == "persistence":
        # Constant retention has no downward crossing.
        return np.nan

    if method == "full_history":
        fit_y = y
    elif method == "recent_20":
        fit_y = y[-RECENT_WINDOW:]
    else:
        raise ValueError(f"Unknown method: {method}")

    x = np.arange(len(fit_y), dtype=float)
    slope, intercept = np.polyfit(x, fit_y, deg=1)

    # A non-negative slope does not imply a future EOL crossing.
    if not np.isfinite(slope) or slope >= 0:
        return np.nan

    # Fit x=0..n-1 for the history. The next observation is x=n.
    predicted_x = (EOL_THRESHOLD_PCT - intercept) / slope

    if not np.isfinite(predicted_x):
        return np.nan

    # Convert the fitted relative position to a 1-based
    # discharge-observation index.
    predicted_eol = origin - len(fit_y) + 1 + predicted_x

    # Do not report an EOL prediction that is already in the past.
    if predicted_eol <= origin:
        return np.nan

    return float(predicted_eol)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    cell_col = find_column(
        df, ["cell_id", "cell", "battery_id"], "cell identifier"
    )
    target_col = find_column(
        df,
        [
            "relative_capacity_retention_pct",
            "capacity_retention_pct",
            "retention_pct",
            "capacity_retention_percent",
        ],
        "capacity-retention percentage",
    )

    df = df[[cell_col, target_col]].copy()
    df.columns = ["cell_id", "retention_pct"]
    df["retention_pct"] = pd.to_numeric(
        df["retention_pct"], errors="coerce"
    )
    df = df.dropna(subset=["retention_pct"])

    prediction_rows = []

    for cell_id, group in df.groupby("cell_id", sort=True):
        # Each row represents one discharge observation. Use its
        # sequential position as the index to avoid gaps in labels.
        y = group["retention_pct"].to_numpy(dtype=float)
        n = len(y)

        crossed = np.flatnonzero(y <= EOL_THRESHOLD_PCT)

        if len(crossed) == 0:
            print(
                f"Skipping {cell_id}: no observed "
                f"{EOL_THRESHOLD_PCT:.0f}% EOL crossing."
            )
            continue

        actual_eol = int(crossed[0] + 1)

        # Use non-overlapping forecast origins separately for each
        # horizon. Origins are measured in discharge observations.
        for horizon in HORIZONS:
            origins = range(
                MIN_HISTORY,
                min(actual_eol, n + 1),
                horizon,
            )

            for origin in origins:
                # origin observations are available; the next
                # observation has sequential index origin + 1.
                history = y[:origin]

                methods = {
                    "full_history": estimate_eol(
                        history, origin, "full_history"
                    ),
                    "recent_20": estimate_eol(
                        history, origin, "recent_20"
                    ),
                    "persistence": estimate_eol(
                        history, origin, "persistence"
                    ),
                }

                for method, predicted_eol in methods.items():
                    available = np.isfinite(predicted_eol)

                    prediction_rows.append({
                        "cell_id": cell_id,
                        "horizon": horizon,
                        "forecast_origin": origin,
                        "actual_eol_index": actual_eol,
                        "actual_rul_observations": actual_eol - origin,
                        "method": method,
                        "predicted_eol_index": (
                            predicted_eol if available else np.nan
                        ),
                        "predicted_rul_observations": (
                            predicted_eol - origin
                            if available else np.nan
                        ),
                        "eol_error_observations": (
                            predicted_eol - actual_eol
                            if available else np.nan
                        ),
                        "absolute_eol_error": (
                            abs(predicted_eol - actual_eol)
                            if available else np.nan
                        ),
                        "crossing_predicted": bool(available),
                    })

    if not prediction_rows:
        raise ValueError(
            "No forecasts were generated. Check the retention "
            "column and available observations."
        )

    predictions = pd.DataFrame(prediction_rows)

    summary_rows = []

    for (cell_id, horizon, method), group in predictions.groupby(
        ["cell_id", "horizon", "method"]
    ):
        valid = group[group["crossing_predicted"]]
        total = len(group)

        if len(valid):
            mae = valid["absolute_eol_error"].mean()
            rmse = np.sqrt(
                np.mean(valid["eol_error_observations"] ** 2)
            )
            bias = valid["eol_error_observations"].mean()
        else:
            mae = np.nan
            rmse = np.nan
            bias = np.nan

        summary_rows.append({
            "cell_id": cell_id,
            "horizon": horizon,
            "method": method,
            "forecast_origins": total,
            "crossings_predicted": len(valid),
            "crossing_coverage_pct": 100 * len(valid) / total,
            "MAE_eol_observations": mae,
            "RMSE_eol_observations": rmse,
            "bias_eol_observations": bias,
        })

    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "rul_predictions.csv"
    summary_path = OUTPUT_DIR / "rul_summary.csv"

    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nNON-OVERLAPPING RUL EVALUATION")
    print("=" * 76)
    print(f"EOL threshold: {EOL_THRESHOLD_PCT:.1f}% retention")
    print(f"Minimum history: {MIN_HISTORY} observations")
    print(f"Recent trend window: {RECENT_WINDOW} observations")
    print(f"Horizons: {HORIZONS}")
    print("EOL errors are measured in discharge observations.")
    print("\nSummary:")

    print(
        summary[
            [
                "cell_id",
                "horizon",
                "method",
                "forecast_origins",
                "crossings_predicted",
                "crossing_coverage_pct",
                "MAE_eol_observations",
                "RMSE_eol_observations",
                "bias_eol_observations",
            ]
        ]
        .sort_values(["cell_id", "horizon", "method"])
        .round(2)
        .to_string(index=False)
    )

    print(f"\nPredictions saved: {predictions_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
