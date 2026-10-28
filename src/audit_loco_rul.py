
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

TRAJECTORY_FILE = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "capacity_retention_trajectories.csv"
)

SUMMARY_FILE = RUL_DIR / "loco_summary.csv"
PREDICTIONS_FILE = RUL_DIR / "loco_predictions.csv"

EOL_THRESHOLD = 80.0
TOLERANCE = 1e-6


def require_columns(df, columns, filename):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}"
        )


def main():
    for path in (
        TRAJECTORY_FILE,
        SUMMARY_FILE,
        PREDICTIONS_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    trajectories = pd.read_csv(TRAJECTORY_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    require_columns(
        trajectories,
        ["cell_id", "discharge_index", "capacity_retention_pct"],
        TRAJECTORY_FILE.name,
    )
    require_columns(
        summary,
        [
            "held_out_cell",
            "method",
            "forecast_origins",
            "valid_predictions",
            "coverage_pct",
            "MAE_eol_observations",
            "RMSE_eol_observations",
            "bias_eol_observations",
        ],
        SUMMARY_FILE.name,
    )
    require_columns(
        predictions,
        [
            "held_out_cell",
            "forecast_origin",
            "actual_eol_index",
            "method",
            "predicted_eol_index",
            "valid_prediction",
            "eol_error",
        ],
        PREDICTIONS_FILE.name,
    )

    trajectories["discharge_index"] = pd.to_numeric(
        trajectories["discharge_index"], errors="coerce"
    )
    trajectories["capacity_retention_pct"] = pd.to_numeric(
        trajectories["capacity_retention_pct"], errors="coerce"
    )

    predictions["forecast_origin"] = pd.to_numeric(
        predictions["forecast_origin"], errors="coerce"
    )
    predictions["actual_eol_index"] = pd.to_numeric(
        predictions["actual_eol_index"], errors="coerce"
    )
    predictions["predicted_eol_index"] = pd.to_numeric(
        predictions["predicted_eol_index"], errors="coerce"
    )
    predictions["eol_error"] = pd.to_numeric(
        predictions["eol_error"], errors="coerce"
    )

    # Reconstruct actual EOL from the first observed <=80% record.
    actual_eol_by_cell = {}

    print("\n1. ACTUAL EOL REFERENCE CHECK")
    print("=" * 65)

    for cell, group in trajectories.groupby("cell_id", sort=True):
        group = group.sort_values("discharge_index")

        indices = group["discharge_index"].to_numpy(dtype=float)
        retention = group["capacity_retention_pct"].to_numpy(dtype=float)

        if (
            not np.isfinite(indices).all()
            or not np.isfinite(retention).all()
        ):
            raise ValueError(
                f"{cell}: missing/non-finite index or retention values."
            )

        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{cell}: duplicate discharge indices.")

        if not np.all(np.diff(indices) > 0):
            raise ValueError(f"{cell}: discharge indices are not increasing.")

        crossings = np.flatnonzero(retention <= EOL_THRESHOLD)

        if len(crossings) == 0:
            print(f"{cell}: no observed EOL crossing.")
            continue

        actual_eol = int(indices[crossings[0]])
        actual_eol_by_cell[cell] = actual_eol

        print(
            f"{cell}: first observed crossing at discharge index "
            f"{actual_eol}, retention={retention[crossings[0]]:.4f}%"
        )

    print("\n2. PREDICTION ROW CHECKS")
    print("=" * 65)

    valid_values = (
        predictions["valid_prediction"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    recognized = valid_values.isin(["true", "false"])
    if not recognized.all():
        raise ValueError(
            "Unrecognized valid_prediction values: "
            f"{valid_values[~recognized].unique().tolist()}"
        )

    predictions["valid_prediction"] = valid_values.eq("true")

    total_checks = 0
    failures = []

    for row_number, row in predictions.iterrows():
        cell = row["held_out_cell"]
        origin = row["forecast_origin"]
        actual = row["actual_eol_index"]
        predicted = row["predicted_eol_index"]
        error = row["eol_error"]
        valid = row["valid_prediction"]
        method = row["method"]

        def fail(message):
            failures.append(
                f"row {row_number + 1}, {cell}, {method}: {message}"
            )

        total_checks += 1

        if cell not in actual_eol_by_cell:
            fail("no observed EOL reference in trajectory data")
            continue

        expected_actual = actual_eol_by_cell[cell]

        if not np.isfinite(actual) or actual != expected_actual:
            fail(
                f"actual EOL {actual} does not match "
                f"trajectory EOL {expected_actual}"
            )

        if not np.isfinite(origin) or origin >= expected_actual:
            fail("forecast origin is not before actual EOL")

        if valid:
            if not np.isfinite(predicted):
                fail("marked valid but predicted EOL is not finite")
            else:
                if predicted <= origin:
                    fail("valid predicted EOL is not after forecast origin")

                if not np.isfinite(error):
                    fail("valid prediction has a missing error")
                elif abs(error - (predicted - actual)) > TOLERANCE:
                    fail("signed error does not equal predicted minus actual EOL")
        else:
            if np.isfinite(predicted):
                fail("marked invalid but predicted EOL is finite")
            if np.isfinite(error):
                fail("marked invalid but error is finite")

    print(f"Prediction rows checked: {total_checks}")
    print(f"Failed row checks: {len(failures)}")

    if failures:
        for item in failures[:30]:
            print("FAIL:", item)
        if len(failures) > 30:
            print(f"... plus {len(failures) - 30} additional failures")

    print("\n3. SUMMARY METRIC REPRODUCTION")
    print("=" * 65)

    metric_failures = []

    for _, row in summary.iterrows():
        cell = row["held_out_cell"]
        method = row["method"]

        subset = predictions[
            (predictions["held_out_cell"] == cell)
            & (predictions["method"] == method)
        ]

        if subset.empty:
            metric_failures.append(f"{cell}/{method}: no prediction rows")
            continue

        valid = subset["valid_prediction"]
        valid_rows = subset.loc[valid]
        errors = valid_rows["eol_error"].to_numpy(dtype=float)

        calculated = {
            "forecast_origins": len(subset),
            "valid_predictions": len(valid_rows),
            "coverage_pct": 100.0 * len(valid_rows) / len(subset),
            "MAE_eol_observations": (
                float(np.mean(np.abs(errors))) if len(errors) else np.nan
            ),
            "RMSE_eol_observations": (
                float(np.sqrt(np.mean(errors ** 2)))
                if len(errors) else np.nan
            ),
            "bias_eol_observations": (
                float(np.mean(errors)) if len(errors) else np.nan
            ),
        }

        print(f"\n{cell} / {method}")

        for metric, value in calculated.items():
            recorded = row[metric]

            if pd.isna(value) and pd.isna(recorded):
                matches = True
            else:
                matches = (
                    pd.notna(value)
                    and pd.notna(recorded)
                    and np.isclose(
                        float(value),
                        float(recorded),
                        atol=TOLERANCE,
                        rtol=TOLERANCE,
                    )
                )

            print(
                f"  {metric}: calculated={value}, "
                f"saved={recorded}, match={matches}"
            )

            if not matches:
                metric_failures.append(
                    f"{cell}/{method}: {metric} does not match saved summary"
                )

    print("\n4. FINAL AUDIT STATUS")
    print("=" * 65)

    if failures or metric_failures:
        print("AUDIT FAILED: inspect the issues listed above.")
        for item in metric_failures:
            print("METRIC FAIL:", item)
    else:
        print("PASS: prediction checks and summary metrics are consistent.")

    print(
        "\nNote: this checks internal consistency, not whether the "
        "forecasting method generalizes to unseen cells."
    )


if __name__ == "__main__":
    main()
