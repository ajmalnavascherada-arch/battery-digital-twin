
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "constrained_rul_validation"
    / "constrained_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "equal_coverage_comparison"
)

METHODS = [
    "full_history",
    "recent_10",
    "recent_20",
    "recent_30",
]


def metrics(errors):
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        return {
            "n_shared": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "bias": np.nan,
        }

    return {
        "n_shared": len(errors),
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "bias": np.mean(errors),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = {
        "cell_id",
        "forecast_origin",
        "actual_eol_index",
        "actual_rul_observations",
        "method",
        "eol_error_observations",
        "valid_prediction",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Convert CSV booleans safely.
    if df["valid_prediction"].dtype != bool:
        df["valid_prediction"] = (
            df["valid_prediction"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )

    df["eol_error_observations"] = pd.to_numeric(
        df["eol_error_observations"], errors="coerce"
    )

    # One row per cell, origin and method is expected.
    key = ["cell_id", "forecast_origin", "method"]
    if df.duplicated(key).any():
        raise ValueError(
            "Duplicate cell/origin/method rows found. "
            "Inspect the input before comparing."
        )

    valid = df[
        df["valid_prediction"]
        & df["eol_error_observations"].notna()
    ].copy()

    # Wide table makes it possible to compare methods on identical
    # origins, rather than comparing different subsets of predictions.
    wide = valid.pivot(
        index=["cell_id", "forecast_origin"],
        columns="method",
        values="eol_error_observations",
    ).reset_index()

    for method in METHODS:
        if method not in wide.columns:
            wide[method] = np.nan

    metadata = (
        df[[
            "cell_id",
            "forecast_origin",
            "actual_eol_index",
            "actual_rul_observations",
        ]]
        .drop_duplicates(["cell_id", "forecast_origin"])
    )

    wide = wide.merge(
        metadata,
        on=["cell_id", "forecast_origin"],
        how="left",
        validate="one_to_one",
    )

    # Pairwise comparison: compare each recent window against
    # full-history on exactly the origins where both are valid.
    pair_rows = []

    for cell_id in sorted(df["cell_id"].unique()):
        cell = wide[wide["cell_id"] == cell_id]

        for method in METHODS[1:]:
            paired = cell[
                cell["full_history"].notna()
                & cell[method].notna()
            ]

            full_metrics = metrics(paired["full_history"])
            window_metrics = metrics(paired[method])

            pair_rows.append({
                "cell_id": cell_id,
                "comparison": f"{method}_vs_full_history",
                "shared_origins": len(paired),
                "full_history_MAE": full_metrics["MAE"],
                "window_MAE": window_metrics["MAE"],
                "MAE_change_pct": (
                    100 * (
                        window_metrics["MAE"]
                        - full_metrics["MAE"]
                    ) / full_metrics["MAE"]
                    if full_metrics["MAE"] not in (0, np.nan)
                    and pd.notna(full_metrics["MAE"])
                    else np.nan
                ),
                "full_history_RMSE": full_metrics["RMSE"],
                "window_RMSE": window_metrics["RMSE"],
                "full_history_bias": full_metrics["bias"],
                "window_bias": window_metrics["bias"],
            })

    pairwise = pd.DataFrame(pair_rows)

    # Strict equal-coverage comparison: all four methods must have
    # a valid prediction at an origin.
    common = wide.dropna(subset=METHODS).copy()

    all_rows = []
    for cell_id in sorted(df["cell_id"].unique()):
        cell = common[common["cell_id"] == cell_id]

        for method in METHODS:
            result = metrics(cell[method])
            all_rows.append({
                "cell_id": cell_id,
                "method": method,
                **result,
            })

    all_method_summary = pd.DataFrame(all_rows)

    # Coverage counts include all attempted origins, including
    # rejected or unavailable predictions.
    coverage = (
        df.groupby(["cell_id", "method"])
        .agg(
            forecast_origins=("forecast_origin", "nunique"),
            valid_predictions=("valid_prediction", "sum"),
        )
        .reset_index()
    )
    coverage["coverage_pct"] = (
        100 * coverage["valid_predictions"]
        / coverage["forecast_origins"]
    )

    # Early vs near-EOL diagnostics on the common valid origins.
    # "Early" means more than half the observed EOL index remains;
    # "near EOL" means at most half remains. This is a diagnostic
    # split, not a universal battery-aging definition.
    common["forecast_stage"] = np.where(
        common["actual_rul_observations"]
        > 0.5 * common["actual_eol_index"],
        "early",
        "near_eol",
    )

    stage_rows = []
    for (cell_id, stage), cell in common.groupby(
        ["cell_id", "forecast_stage"]
    ):
        for method in METHODS:
            result = metrics(cell[method])
            stage_rows.append({
                "cell_id": cell_id,
                "forecast_stage": stage,
                "method": method,
                **result,
            })

    stage_summary = pd.DataFrame(stage_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairwise_path = OUTPUT_DIR / "pairwise_shared_origin_comparison.csv"
    common_path = OUTPUT_DIR / "all_methods_equal_coverage.csv"
    coverage_path = OUTPUT_DIR / "forecast_coverage.csv"
    stage_path = OUTPUT_DIR / "early_vs_near_eol.csv"

    pairwise.to_csv(pairwise_path, index=False)
    all_method_summary.to_csv(common_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    stage_summary.to_csv(stage_path, index=False)

    print("\nRUL EQUAL-COVERAGE COMPARISON")
    print("=" * 78)

    print("\n1. Forecast coverage across all attempted origins")
    print(
        coverage.sort_values(["cell_id", "method"])
        .round(2)
        .to_string(index=False)
    )

    print("\n2. Pairwise comparison against full-history")
    print(
        pairwise.sort_values(["cell_id", "comparison"])
        .round(2)
        .to_string(index=False)
    )

    print("\n3. Strict equal-coverage results: all four methods valid")
    print(f"Shared origins across all methods: {len(common)}")
    print(
        all_method_summary.sort_values(["cell_id", "method"])
        .round(2)
        .to_string(index=False)
    )

    print("\n4. Early versus near-EOL results on shared origins")
    print(
        stage_summary.sort_values(
            ["cell_id", "forecast_stage", "method"]
        )
        .round(2)
        .to_string(index=False)
    )

    print("\nSaved files:")
    print(pairwise_path)
    print(common_path)
    print(coverage_path)
    print(stage_path)


if __name__ == "__main__":
    main()
