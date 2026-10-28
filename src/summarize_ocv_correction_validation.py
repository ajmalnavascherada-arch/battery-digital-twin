
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "ocv_shape_correction_validation"
    / "ocv_shape_correction_metrics.csv"
)

COEFFICIENT_FILE = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "ocv_shape_correction_validation"
    / "frozen_correction_coefficients.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "ocv_shape_correction_validation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing metrics file:\n{INPUT_FILE}")

    metrics = pd.read_csv(INPUT_FILE)
    required = {
        "cycle_number",
        "evaluation",
        "region",
        "samples",
        "baseline_rmse_v",
        "corrected_rmse_v",
        "baseline_mae_v",
        "corrected_mae_v",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Only independent cycles, using all active-discharge samples.
    test = metrics[
        (metrics["evaluation"] == "independent_cycle")
        & (metrics["region"] == "all_active")
    ].copy()

    if test.empty:
        raise ValueError("No independent-cycle all-active metrics found.")

    test["rmse_reduction_percent"] = (
        100.0
        * (test["baseline_rmse_v"] - test["corrected_rmse_v"])
        / test["baseline_rmse_v"]
    )
    test["mae_reduction_percent"] = (
        100.0
        * (test["baseline_mae_v"] - test["corrected_mae_v"])
        / test["baseline_mae_v"]
    )

    # Equal weighting: average the cycle-specific metrics.
    mean_original_rmse = test["baseline_rmse_v"].mean()
    mean_corrected_rmse = test["corrected_rmse_v"].mean()
    mean_original_mae = test["baseline_mae_v"].mean()
    mean_corrected_mae = test["corrected_mae_v"].mean()

    mean_rmse_reduction = (
        100.0
        * (mean_original_rmse - mean_corrected_rmse)
        / mean_original_rmse
    )
    mean_mae_reduction = (
        100.0
        * (mean_original_mae - mean_corrected_mae)
        / mean_original_mae
    )

    detailed_file = OUTPUT_DIR / "independent_cycle_validation_summary.csv"
    test.to_csv(detailed_file, index=False)

    if COEFFICIENT_FILE.exists():
        coefficients = pd.read_csv(COEFFICIENT_FILE).iloc[0]
        coefficient_text = (
            f"Training SOC interval: "
            f"{coefficients['training_soc_min']:.0%}–"
            f"{coefficients['training_soc_max']:.0%}\n"
            f"Polynomial degree: {int(coefficients['polynomial_degree'])}\n"
            f"SOC center: {coefficients['soc_center']:.3f}\n"
            f"Quadratic coefficient: {coefficients['coefficient_x2']:.8f}\n"
            f"Linear coefficient: {coefficients['coefficient_x1']:.8f}\n"
            f"Constant coefficient: "
            f"{coefficients['coefficient_constant']:.8f}\n"
        )
    else:
        coefficient_text = "Coefficient file not found.\n"

    report_file = OUTPUT_DIR / "independent_cycle_validation_report.txt"

    with report_file.open("w", encoding="utf-8") as f:
        f.write("Frozen SOC-Dependent Voltage Correction\n")
        f.write("Independent-Cycle Validation Summary\n")
        f.write("=" * 48 + "\n\n")
        f.write("Training cycle: NASA B0005 cycle 2 only\n")
        f.write("Evaluation cycles: 4, 6, 10, 20\n")
        f.write("Correction fitted on estimated SOC from 15% to 85%\n")
        f.write("No correction refitting was performed on test cycles.\n\n")

        f.write(coefficient_text + "\n")

        f.write("Independent-cycle results:\n")
        f.write(test.to_string(index=False) + "\n\n")

        f.write(
            f"Mean original RMSE: {mean_original_rmse * 1000:.3f} mV\n"
        )
        f.write(
            f"Mean corrected RMSE: {mean_corrected_rmse * 1000:.3f} mV\n"
        )
        f.write(f"Mean RMSE reduction: {mean_rmse_reduction:.2f}%\n")
        f.write(
            f"Mean original MAE: {mean_original_mae * 1000:.3f} mV\n"
        )
        f.write(
            f"Mean corrected MAE: {mean_corrected_mae * 1000:.3f} mV\n"
        )
        f.write(f"Mean MAE reduction: {mean_mae_reduction:.2f}%\n\n")

        f.write("Limitations:\n")
        f.write(
            "- The polynomial is a diagnostic voltage correction, not a "
            "measured or calibrated OCV-SOC curve.\n"
        )
        f.write(
            "- The SOC estimate assumes initial SOC and recorded capacity.\n"
        )
        f.write(
            "- Test cycles come from the same B0005 cell, not independent cells.\n"
        )
        f.write(
            "- End-of-discharge voltage-knee error remains significant.\n"
        )
        f.write(
            "- These results do not establish safety-critical accuracy.\n"
        )

    print("\nINDEPENDENT-CYCLE VALIDATION SUMMARY")
    print("===================================")
    print(f"Test cycles: {test['cycle_number'].tolist()}")
    print(f"Mean original RMSE: {mean_original_rmse * 1000:.3f} mV")
    print(f"Mean corrected RMSE: {mean_corrected_rmse * 1000:.3f} mV")
    print(f"Mean RMSE reduction: {mean_rmse_reduction:.2f}%")
    print(f"Mean original MAE: {mean_original_mae * 1000:.3f} mV")
    print(f"Mean corrected MAE: {mean_corrected_mae * 1000:.3f} mV")
    print(f"Mean MAE reduction: {mean_mae_reduction:.2f}%")
    print(f"\nDetailed CSV: {detailed_file}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
