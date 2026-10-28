
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    / "constrained_rul_validation"
)

EOL_THRESHOLD_PCT = 80.0
MIN_HISTORY = 30
ORIGIN_STRIDE = 5
WINDOWS = (10, 20, 30)

# Exploratory screening threshold, not a validated physical limit.
# Units: percentage points of retention per discharge observation.
MIN_DECLINE_RATE_PP_PER_OBS = 0.05


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Cannot identify {description}.\n"
        f"Available columns: {list(df.columns)}"
    )


def fit_eol(history, origin, window_size=None):
    """
    Estimate EOL from sequential observations.

    origin is the number of observations already available.
    Returns the raw estimate, fitted slope and a validity reason.
    """
    history = np.asarray(history, dtype=float)

    if window_size is not None:
        history = history[-window_size:]

    if len(history) < 2 or not np.isfinite(history).all():
        return np.nan, np.nan, "insufficient_or_invalid_data"

    x = np.arange(len(history), dtype=float)
    slope, intercept = np.polyfit(x, history, 1)

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return np.nan, slope, "invalid_fit"

    crossing_x = (EOL_THRESHOLD_PCT - intercept) / slope

    # Map fitted x back to the sequential, 1-based observation index.
    first_history_index = origin - len(history) + 1
    raw_eol = first_history_index + crossing_x

    if slope >= 0:
        return raw_eol, slope, "non_declining_slope"

    if abs(slope) < MIN_DECLINE_RATE_PP_PER_OBS:
        return raw_eol, slope, "slope_too_shallow"

    if not np.isfinite(raw_eol) or raw_eol <= origin:
        return raw_eol, slope, "no_future_crossing"

    return float(raw_eol), float(slope), "valid"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    cell_col = find_column(
        df, ["cell_id", "cell", "battery_id"], "cell identifier"
    )
    retention_col = find_column(
        df,
        [
            "relative_capacity_retention_pct",
            "capacity_retention_pct",
            "retention_pct",
            "capacity_retention_percent",
        ],
        "capacity-retention percentage",
    )

    df = df[[cell_col, retention_col]].copy()
    df.columns = ["cell_id", "retention_pct"]
    df["retention_pct"] = pd.to_numeric(
        df["retention_pct"], errors="coerce"
    )
    df = df.dropna(subset=["retention_pct"])

    prediction_rows = []

    for cell_id, group in df.groupby("cell_id", sort=True):
        y = group["retention_pct"].to_numpy(dtype=float)

        crossings = np.flatnonzero(y <= EOL_THRESHOLD_PCT)
        if len(crossings) == 0:
            print(f"Skipping {cell_id}: no observed EOL crossing.")
            continue

        actual_eol = int(crossings[0] + 1)

        # All methods are evaluated at identical origins.
        for origin in range(MIN_HISTORY, actual_eol, ORIGIN_STRIDE):
            history = y[:origin]

            models = [("full_history", None)] + [
                (f"recent_{window}", window) for window in WINDOWS
            ]

            for method, window in models:
                raw_eol, slope, status = fit_eol(
                    history, origin, window
                )

                valid = status == "valid"
                error = raw_eol - actual_eol if valid else np.nan

                prediction_rows.append({
                    "cell_id": cell_id,
                    "forecast_origin": origin,
                    "actual_eol_index": actual_eol,
                    "actual_rul_observations": actual_eol - origin,
                    "method": method,
                    "fitted_slope_pp_per_observation": slope,
                    "raw_predicted_eol_index": raw_eol,
                    "predicted_eol_index": raw_eol if valid else np.nan,
                    "eol_error_observations": error,
                    "absolute_eol_error": (
                        abs(error) if valid else np.nan
                    ),
                    "valid_prediction": valid,
                    "status": status,
                })

    if not prediction_rows:
        raise ValueError(
            "No forecasts generated. Check the retention dataset."
        )

    predictions = pd.DataFrame(prediction_rows)
    summary_rows = []

    for (cell_id, method), group in predictions.groupby(
        ["cell_id", "method"]
    ):
        valid = group[group["valid_prediction"]]
        errors = valid["eol_error_observations"].to_numpy(dtype=float)

        if len(errors):
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            bias = np.mean(errors)
        else:
            mae = rmse = bias = np.nan

        summary_rows.append({
            "cell_id": cell_id,
            "method": method,
            "forecast_origins": len(group),
            "valid_predictions": len(valid),
            "coverage_pct": 100 * len(valid) / len(group),
            "MAE_eol_observations": mae,
            "RMSE_eol_observations": rmse,
            "bias_eol_observations": bias,
            "shallow_slope_rejections": int(
                (group["status"] == "slope_too_shallow").sum()
            ),
            "other_rejections": int(
                (~group["status"].isin(["valid", "slope_too_shallow"])).sum()
            ),
        })

    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "constrained_predictions.csv"
    summary_path = OUTPUT_DIR / "constrained_summary.csv"
    plot_path = OUTPUT_DIR / "constrained_rul_diagnostics.png"

    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    cells = sorted(predictions["cell_id"].unique())
    fig, axes = plt.subplots(
        len(cells), 1,
        figsize=(10, 4 * len(cells)),
        squeeze=False,
    )

    for ax, cell_id in zip(axes[:, 0], cells):
        cell_df = predictions[predictions["cell_id"] == cell_id]
        actual_eol = int(cell_df["actual_eol_index"].iloc[0])

        ax.axhline(
            actual_eol,
            linestyle="--",
            label=f"Observed EOL = {actual_eol}",
        )

        for method in [
            "full_history",
            "recent_10",
            "recent_20",
            "recent_30",
        ]:
            subset = cell_df[cell_df["method"] == method]
            ax.plot(
                subset["forecast_origin"],
                subset["predicted_eol_index"],
                marker="o",
                label=method,
            )

        ax.set_title(cell_id)
        ax.set_xlabel("Forecast origin (observations available)")
        ax.set_ylabel("Predicted EOL observation index")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Constrained RUL forecasts: common origins and slope screening",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nCONSTRAINED RUL VALIDATION")
    print("=" * 82)
    print(f"EOL threshold: {EOL_THRESHOLD_PCT:.1f}%")
    print(f"Common minimum history: {MIN_HISTORY}")
    print(f"Forecast-origin stride: {ORIGIN_STRIDE}")
    print(f"Recent windows: {WINDOWS}")
    print(
        "Minimum decline-rate screen: "
        f"{MIN_DECLINE_RATE_PP_PER_OBS:.3f} pp/observation"
    )
    print("The slope threshold is exploratory, not physically validated.")
    print("\nSummary:")

    print(
        summary[
            [
                "cell_id",
                "method",
                "forecast_origins",
                "valid_predictions",
                "coverage_pct",
                "MAE_eol_observations",
                "RMSE_eol_observations",
                "bias_eol_observations",
                "shallow_slope_rejections",
                "other_rejections",
            ]
        ]
        .sort_values(["cell_id", "method"])
        .round(2)
        .to_string(index=False)
    )

    print("\nStatus counts:")
    print(
        predictions.groupby(["method", "status"])
        .size()
        .rename("count")
        .to_string()
    )

    print(f"\nPredictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
