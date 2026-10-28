
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
    / "leave_one_cell_out"
)

EOL_THRESHOLD = 80.0
MIN_HISTORY = 30
ORIGIN_STRIDE = 5
WINDOWS = (10, 20, 30)
MIN_TRAIN_COVERAGE = 0.80


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"Cannot identify {description}. "
        f"Available columns: {list(df.columns)}"
    )


def estimate_eol(history, origin, window=None):
    """Return predicted 1-based EOL index and a validity flag."""
    y = np.asarray(history, dtype=float)

    if window is not None:
        y = y[-window:]

    if len(y) < 2 or not np.isfinite(y).all():
        return np.nan, False

    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return np.nan, False

    # No future downward crossing if the trend is flat or increasing.
    if slope >= 0:
        return np.nan, False

    # Avoid extrapolation from an almost-flat degradation trend.
    if abs(slope) < 0.05:
        return np.nan, False

    crossing_x = (EOL_THRESHOLD - intercept) / slope
    first_index = origin - len(y) + 1
    predicted_eol = first_index + crossing_x

    if not np.isfinite(predicted_eol) or predicted_eol <= origin:
        return np.nan, False

    return float(predicted_eol), True


def evaluate_cell(cell_id, values):
    """Evaluate all methods at the same pre-EOL forecast origins."""
    y = np.asarray(values, dtype=float)
    crossings = np.flatnonzero(y <= EOL_THRESHOLD)

    if len(crossings) == 0:
        return []

    actual_eol = int(crossings[0] + 1)
    rows = []

    for origin in range(MIN_HISTORY, actual_eol, ORIGIN_STRIDE):
        history = y[:origin]

        methods = [("full_history", None)] + [
            (f"recent_{window}", window) for window in WINDOWS
        ]

        for method, window in methods:
            predicted, valid = estimate_eol(history, origin, window)

            rows.append({
                "cell_id": cell_id,
                "forecast_origin": origin,
                "actual_eol_index": actual_eol,
                "method": method,
                "predicted_eol_index": predicted if valid else np.nan,
                "valid_prediction": valid,
                "eol_error": (
                    predicted - actual_eol if valid else np.nan
                ),
                "absolute_eol_error": (
                    abs(predicted - actual_eol) if valid else np.nan
                ),
            })

    return rows


def score_method(rows):
    """Return coverage and error metrics for a candidate window."""
    total = len(rows)
    valid_rows = [r for r in rows if r["valid_prediction"]]
    coverage = len(valid_rows) / total if total else 0.0

    errors = np.array(
        [r["eol_error"] for r in valid_rows], dtype=float
    )

    mae = float(np.mean(np.abs(errors))) if len(errors) else np.nan
    rmse = (
        float(np.sqrt(np.mean(errors ** 2)))
        if len(errors) else np.nan
    )
    bias = float(np.mean(errors)) if len(errors) else np.nan

    return {
        "forecast_origins": total,
        "valid_predictions": len(valid_rows),
        "coverage_pct": 100 * coverage,
        "MAE_eol_observations": mae,
        "RMSE_eol_observations": rmse,
        "bias_eol_observations": bias,
    }


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
        "capacity retention",
    )

    df = df[[cell_col, retention_col]].copy()
    df.columns = ["cell_id", "retention_pct"]
    df["retention_pct"] = pd.to_numeric(
        df["retention_pct"], errors="coerce"
    )
    df = df.dropna(subset=["retention_pct"])

    cell_values = {
        cell_id: group["retention_pct"].to_numpy(dtype=float)
        for cell_id, group in df.groupby("cell_id", sort=True)
    }

    if len(cell_values) < 3:
        raise ValueError(
            f"Expected at least 3 cells; found {len(cell_values)}."
        )

    # Precompute each cell's forecast results. Each result uses only
    # that cell's history available at each forecast origin.
    rows_by_cell = {
        cell_id: evaluate_cell(cell_id, values)
        for cell_id, values in cell_values.items()
    }

    fold_rows = []
    selection_rows = []
    prediction_rows = []

    for held_out in sorted(cell_values):
        training_cells = [
            cell for cell in sorted(cell_values) if cell != held_out
        ]

        # Choose a window using training cells only.
        training_scores = []

        for window in WINDOWS:
            method = f"recent_{window}"
            train_rows = [
                row
                for cell in training_cells
                for row in rows_by_cell[cell]
                if row["method"] == method
            ]

            score = score_method(train_rows)
            eligible = (
                score["coverage_pct"] / 100 >= MIN_TRAIN_COVERAGE
                and np.isfinite(score["MAE_eol_observations"])
            )

            training_scores.append({
                "held_out_cell": held_out,
                "training_cells": ";".join(training_cells),
                "candidate_window": window,
                **score,
                "eligible_by_coverage": eligible,
            })

        eligible_scores = [
            item for item in training_scores
            if item["eligible_by_coverage"]
        ]

        if eligible_scores:
            selected = min(
                eligible_scores,
                key=lambda item: item["MAE_eol_observations"],
            )
            selection_rule = "lowest_MAE_among_eligible"
        else:
            # If no window meets the coverage requirement, prioritize
            # coverage and use MAE only to break ties.
            usable = [
                item for item in training_scores
                if np.isfinite(item["MAE_eol_observations"])
            ]
            if not usable:
                raise ValueError(
                    f"No usable training-window scores for {held_out}."
                )
            selected = sorted(
                usable,
                key=lambda item: (
                    -item["coverage_pct"],
                    item["MAE_eol_observations"],
                ),
            )[0]
            selection_rule = "highest_coverage_fallback"

        chosen_window = int(selected["candidate_window"])

        for item in training_scores:
            item["selected_window"] = chosen_window
            item["selection_rule"] = selection_rule
            selection_rows.append(item)

        heldout_rows = rows_by_cell[held_out]
        for method in ("full_history", f"recent_{chosen_window}"):
            subset = [
                row for row in heldout_rows
                if row["method"] == method
            ]

            score = score_method(subset)
            fold_rows.append({
                "held_out_cell": held_out,
                "training_cells": ";".join(training_cells),
                "selected_window": chosen_window,
                "method": method,
                **score,
            })

            for row in subset:
                prediction_rows.append({
                    "held_out_cell": held_out,
                    "training_cells": ";".join(training_cells),
                    "selected_window": chosen_window,
                    **row,
                })

    fold_summary = pd.DataFrame(fold_rows)
    selection_details = pd.DataFrame(selection_rows)
    predictions = pd.DataFrame(prediction_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_path = OUTPUT_DIR / "loco_summary.csv"
    selection_path = OUTPUT_DIR / "training_window_selection.csv"
    predictions_path = OUTPUT_DIR / "loco_predictions.csv"

    fold_summary.to_csv(fold_path, index=False)
    selection_details.to_csv(selection_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\nLEAVE-ONE-CELL-OUT RUL VALIDATION")
    print("=" * 78)
    print(f"EOL threshold: {EOL_THRESHOLD}%")
    print(f"Minimum history: {MIN_HISTORY}")
    print(f"Origin stride: {ORIGIN_STRIDE}")
    print(f"Candidate windows: {WINDOWS}")
    print(f"Training coverage requirement: {100 * MIN_TRAIN_COVERAGE:.0f}%")

    print("\n1. Window selection on training cells only")
    print(
        selection_details[
            [
                "held_out_cell",
                "training_cells",
                "candidate_window",
                "coverage_pct",
                "MAE_eol_observations",
                "eligible_by_coverage",
                "selected_window",
                "selection_rule",
            ]
        ].round(2).to_string(index=False)
    )

    print("\n2. Held-out-cell results")
    print(
        fold_summary[
            [
                "held_out_cell",
                "training_cells",
                "selected_window",
                "method",
                "forecast_origins",
                "valid_predictions",
                "coverage_pct",
                "MAE_eol_observations",
                "RMSE_eol_observations",
                "bias_eol_observations",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nSaved files:")
    print(fold_path)
    print(selection_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
