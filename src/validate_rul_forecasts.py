
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
    / "rul_validation_v2"
)

EOL_THRESHOLD_PCT = 80.0
MIN_HISTORY = 20
RECENT_WINDOW = 20
ORIGIN_STRIDE = 5


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Cannot identify {description}.\n"
        f"Available columns: {list(df.columns)}"
    )


def predict_eol(history, origin, mode):
    """
    origin is the number of observations available (1-based endpoint).
    Returns the predicted 1-based EOL observation index, or NaN.
    """
    history = np.asarray(history, dtype=float)

    if mode == "recent_20":
        history = history[-RECENT_WINDOW:]

    if len(history) < 2 or not np.isfinite(history).all():
        return np.nan

    x = np.arange(len(history), dtype=float)
    slope, intercept = np.polyfit(x, history, 1)

    # A flat or rising trend does not support a future downward crossing.
    if not np.isfinite(slope) or slope >= 0:
        return np.nan

    crossing_x = (EOL_THRESHOLD_PCT - intercept) / slope

    if not np.isfinite(crossing_x):
        return np.nan

    # Map the fitted x coordinate back to the sequential observation index.
    first_history_index = origin - len(history) + 1
    predicted_eol = first_history_index + crossing_x

    # Do not count a crossing that the model places at or before the origin.
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
        # Use sequential discharge-observation positions consistently.
        y = group["retention_pct"].to_numpy(dtype=float)

        crossings = np.flatnonzero(y <= EOL_THRESHOLD_PCT)

        if len(crossings) == 0:
            print(f"Skipping {cell_id}: no observed EOL crossing.")
            continue

        # 1-based observation index of the first measured threshold crossing.
        actual_eol = int(crossings[0] + 1)

        # Identical forecast origins for both methods.
        origins = range(
            MIN_HISTORY,
            actual_eol,
            ORIGIN_STRIDE,
        )

        for origin in origins:
            history = y[:origin]

            for method in ("full_history", "recent_20"):
                predicted_eol = predict_eol(history, origin, method)
                valid = np.isfinite(predicted_eol)

                error = (
                    predicted_eol - actual_eol if valid else np.nan
                )

                prediction_rows.append({
                    "cell_id": cell_id,
                    "forecast_origin": origin,
                    "actual_eol_index": actual_eol,
                    "actual_rul_observations": actual_eol - origin,
                    "method": method,
                    "predicted_eol_index": (
                        predicted_eol if valid else np.nan
                    ),
                    "predicted_rul_observations": (
                        predicted_eol - origin if valid else np.nan
                    ),
                    "eol_error_observations": error,
                    "absolute_eol_error": (
                        abs(error) if valid else np.nan
                    ),
                    "crossing_predicted": bool(valid),
                })

    if not prediction_rows:
        raise ValueError(
            "No forecasts generated. Check the input retention data."
        )

    predictions = pd.DataFrame(prediction_rows)

    summary_rows = []

    for (cell_id, method), group in predictions.groupby(
        ["cell_id", "method"]
    ):
        valid = group[group["crossing_predicted"]]
        total = len(group)

        if len(valid):
            errors = valid["eol_error_observations"].to_numpy()

            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            bias = np.mean(errors)
        else:
            mae = rmse = bias = np.nan

        summary_rows.append({
            "cell_id": cell_id,
            "method": method,
            "forecast_origins": total,
            "valid_crossings": len(valid),
            "coverage_pct": 100 * len(valid) / total,
            "MAE_eol_observations": mae,
            "RMSE_eol_observations": rmse,
            "bias_eol_observations": bias,
        })

    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "predictions.csv"
    summary_path = OUTPUT_DIR / "summary.csv"
    plot_path = OUTPUT_DIR / "eol_forecast_diagnostics.png"

    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Diagnostic plot: predicted EOL versus forecast origin.
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

        for method, label in [
            ("full_history", "Full-history trend"),
            ("recent_20", "Recent-20 trend"),
        ]:
            method_df = (
                cell_df[cell_df["method"] == method]
                .sort_values("forecast_origin")
            )

            ax.plot(
                method_df["forecast_origin"],
                method_df["predicted_eol_index"],
                marker="o",
                label=label,
            )

        ax.set_title(cell_id)
        ax.set_xlabel("Forecast origin (observations available)")
        ax.set_ylabel("Predicted EOL observation index")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("RUL validation: predicted versus observed EOL", y=1.01)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nRUL FORECAST VALIDATION V2")
    print("=" * 78)
    print(f"EOL threshold: {EOL_THRESHOLD_PCT}%")
    print(f"Minimum history: {MIN_HISTORY}")
    print(f"Recent window: {RECENT_WINDOW}")
    print(f"Forecast-origin stride: {ORIGIN_STRIDE}")
    print("EOL errors are measured in discharge observations.")
    print("\nSummary:")

    print(
        summary[
            [
                "cell_id",
                "method",
                "forecast_origins",
                "valid_crossings",
                "coverage_pct",
                "MAE_eol_observations",
                "RMSE_eol_observations",
                "bias_eol_observations",
            ]
        ]
        .sort_values(["cell_id", "method"])
        .round(2)
        .to_string(index=False)
    )

    print(f"\nPredictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
