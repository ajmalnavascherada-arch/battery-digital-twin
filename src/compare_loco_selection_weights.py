
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RUL_DIR = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "leave_one_cell_out"
)

PREDICTIONS_FILE = RUL_DIR / "loco_predictions.csv"
TRAJECTORY_FILE = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "capacity_retention_trajectories.csv"
)

OUTPUT_DIR = RUL_DIR / "selection_weight_comparison"

WINDOWS = (10, 20, 30)
MIN_TRAIN_COVERAGE = 80.0
EOL_THRESHOLD = 80.0
MIN_HISTORY = 30
ORIGIN_STRIDE = 5


def estimate_eol(history, origin, window=None):
    """Reproduce the baseline's linear-trend EOL calculation."""
    y = np.asarray(history, dtype=float)

    if window is not None:
        y = y[-window:]

    if len(y) < 2 or not np.isfinite(y).all():
        return np.nan

    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return np.nan

    if slope >= 0 or abs(slope) < 0.05:
        return np.nan

    crossing_x = (EOL_THRESHOLD - intercept) / slope
    first_index = origin - len(y) + 1
    predicted_eol = first_index + crossing_x

    if not np.isfinite(predicted_eol) or predicted_eol <= origin:
        return np.nan

    return float(predicted_eol)


def score_errors(errors):
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        return {
            "valid_predictions": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "bias": np.nan,
        }

    return {
        "valid_predictions": len(errors),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(np.mean(errors)),
    }


def main():
    trajectories = pd.read_csv(TRAJECTORY_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    required = {
        "cell_id",
        "discharge_index",
        "capacity_retention_pct",
    }
    if not required.issubset(trajectories.columns):
        raise ValueError(
            f"Trajectory CSV needs columns: {sorted(required)}"
        )

    # Use the same ordered trajectories and first observed EOL definition.
    cell_data = {}

    for cell, group in trajectories.groupby("cell_id", sort=True):
        group = group.sort_values("discharge_index")
        y = group["capacity_retention_pct"].to_numpy(dtype=float)

        crossings = np.flatnonzero(y <= EOL_THRESHOLD)
        if len(crossings) == 0:
            continue

        actual_eol = int(
            group["discharge_index"].to_numpy()[crossings[0]]
        )
        cell_data[cell] = (y, actual_eol)

    # Independently regenerate each cell's forecasts so the three
    # candidate windows can be compared on identical origins.
    rows = []

    for cell, (y, actual_eol) in cell_data.items():
        for origin in range(MIN_HISTORY, actual_eol, ORIGIN_STRIDE):
            history = y[:origin]

            for window in WINDOWS:
                predicted = estimate_eol(history, origin, window)
                valid = np.isfinite(predicted)

                rows.append({
                    "cell_id": cell,
                    "forecast_origin": origin,
                    "actual_eol_index": actual_eol,
                    "window": window,
                    "valid_prediction": bool(valid),
                    "eol_error": (
                        predicted - actual_eol if valid else np.nan
                    ),
                })

    forecasts = pd.DataFrame(rows)

    if forecasts.empty:
        raise ValueError("No forecast rows were generated.")

    # Confirm regenerated errors agree with saved recent-window errors.
    saved = predictions[
        predictions["method"].isin(
            [f"recent_{window}" for window in WINDOWS]
        )
    ].copy()

    saved["window"] = (
        saved["method"].str.replace("recent_", "", regex=False).astype(int)
    )

    merged = forecasts.merge(
        saved,
        left_on=["cell_id", "forecast_origin", "window"],
        right_on=["held_out_cell", "forecast_origin", "window"],
        how="left",
        suffixes=("_new", "_saved"),
        validate="one_to_one",
    )

    comparable = (
        merged["predicted_eol_index"].notna()
        & merged["predicted_eol_index"].notna()
    ) if "predicted_eol_index" in merged.columns else pd.Series(
        False, index=merged.index
    )

    # Directly compare regenerated errors with saved errors where present.
    saved_error_col = "eol_error_saved"
    new_error_col = "eol_error_new"

    if saved_error_col in merged.columns:
        both_finite = (
            np.isfinite(merged[new_error_col])
            & np.isfinite(merged[saved_error_col])
        )
        if both_finite.any():
            max_difference = (
                merged.loc[both_finite, new_error_col]
                - merged.loc[both_finite, saved_error_col]
            ).abs().max()
        else:
            max_difference = np.nan
    else:
        max_difference = np.nan

    print("\nSELECTION WEIGHT COMPARISON")
    print("=" * 75)
    print(f"Regenerated forecast rows: {len(forecasts)}")
    print(
        "Maximum absolute difference in matching saved errors:",
        max_difference,
    )

    pooled_rows = []
    equal_cell_rows = []

    for held_out in sorted(cell_data):
        train_cells = [
            cell for cell in sorted(cell_data) if cell != held_out
        ]

        for window in WINDOWS:
            train = forecasts[
                forecasts["cell_id"].isin(train_cells)
                & (forecasts["window"] == window)
            ]

            total = len(train)
            valid = train[train["valid_prediction"]]
            pooled_coverage = 100.0 * len(valid) / total if total else 0.0
            pooled = score_errors(valid["eol_error"])

            per_cell = []
            for cell in train_cells:
                subset = train[train["cell_id"] == cell]
                valid_cell = subset[subset["valid_prediction"]]
                coverage = (
                    100.0 * len(valid_cell) / len(subset)
                    if len(subset) else 0.0
                )
                cell_score = score_errors(valid_cell["eol_error"])
                per_cell.append({
                    "cell_id": cell,
                    "coverage_pct": coverage,
                    **cell_score,
                })

            eligible_cells = [
                item for item in per_cell
                if item["coverage_pct"] >= MIN_TRAIN_COVERAGE
                and np.isfinite(item["MAE"])
            ]

            equal_coverage = (
                len(eligible_cells) == len(train_cells)
            )

            equal_mae = (
                float(np.mean([item["MAE"] for item in per_cell]))
                if all(np.isfinite(item["MAE"]) for item in per_cell)
                else np.nan
            )

            pooled_rows.append({
                "held_out_cell": held_out,
                "training_cells": ";".join(train_cells),
                "window": window,
                "pooled_coverage_pct": pooled_coverage,
                "pooled_MAE": pooled["MAE"],
                "eligible_pooled": (
                    pooled_coverage >= MIN_TRAIN_COVERAGE
                    and np.isfinite(pooled["MAE"])
                ),
                "equal_cell_MAE": equal_mae,
                "all_cells_meet_coverage": equal_coverage,
                "cell_details": "; ".join(
                    f"{item['cell_id']}: coverage="
                    f"{item['coverage_pct']:.2f}%, "
                    f"MAE={item['MAE']:.2f}"
                    for item in per_cell
                ),
            })

    results = pd.DataFrame(pooled_rows)

    # Selection A: reproduce baseline pooled-MAE selection.
    # Selection B: require each training cell to meet coverage,
    # then minimize the equally weighted average of per-cell MAEs.
    selection_rows = []

    for held_out, group in results.groupby("held_out_cell", sort=True):
        pooled_eligible = group[group["eligible_pooled"]]
        if pooled_eligible.empty:
            pooled_choice = group.sort_values(
                ["pooled_coverage_pct", "pooled_MAE"],
                ascending=[False, True],
            ).iloc[0]
        else:
            pooled_choice = pooled_eligible.loc[
                pooled_eligible["pooled_MAE"].idxmin()
            ]

        equal_eligible = group[
            group["all_cells_meet_coverage"]
            & np.isfinite(group["equal_cell_MAE"])
        ]

        if equal_eligible.empty:
            equal_choice = None
        else:
            equal_choice = equal_eligible.loc[
                equal_eligible["equal_cell_MAE"].idxmin()
            ]

        selection_rows.append({
            "held_out_cell": held_out,
            "baseline_pooled_choice": int(pooled_choice["window"]),
            "baseline_pooled_training_MAE": pooled_choice["pooled_MAE"],
            "equal_weight_choice": (
                int(equal_choice["window"])
                if equal_choice is not None else np.nan
            ),
            "equal_weight_training_MAE": (
                equal_choice["equal_cell_MAE"]
                if equal_choice is not None else np.nan
            ),
            "selection_changed": (
                int(pooled_choice["window"])
                != int(equal_choice["window"])
                if equal_choice is not None else np.nan
            ),
        })

    selection = pd.DataFrame(selection_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "candidate_window_scores.csv", index=False)
    selection.to_csv(OUTPUT_DIR / "selection_comparison.csv", index=False)

    print("\nCandidate window scores:")
    print(
        results[
            [
                "held_out_cell",
                "window",
                "pooled_coverage_pct",
                "pooled_MAE",
                "eligible_pooled",
                "equal_cell_MAE",
                "all_cells_meet_coverage",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nWindow selection comparison:")
    print(selection.round(2).to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_DIR / "candidate_window_scores.csv")
    print(OUTPUT_DIR / "selection_comparison.csv")

    print(
        "\nInterpretation note: this is a sensitivity experiment. "
        "It does not replace the validated baseline."
    )


if __name__ == "__main__":
    main()
