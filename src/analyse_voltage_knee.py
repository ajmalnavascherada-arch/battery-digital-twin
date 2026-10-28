
"""Diagnose the end-of-discharge voltage knee for NASA B0005 cycle 2.

Reads existing comparison data and writes a diagnostic CSV and report.
Does not modify any existing model or result files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(
    "results_nasa_original/ecm_comparison/"
    "B0005_cycle_2_aligned_predictions.csv"
)
OUT_DIR = Path("results_nasa_original/voltage_knee_diagnostic")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing comparison file: {INPUT}")

    df = pd.read_csv(INPUT).sort_values("time_s").reset_index(drop=True)

    required = [
        "time_s",
        "current_positive_discharge_A",
        "voltage_V",
        "predicted_voltage_V",
        "predicted_voltage_v",
        "soc_raw",
        "soc_bounded",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["voltage_slope_V_per_s"] = (
        df["voltage_V"].diff() / df["time_s"].diff()
    )
    df["baseline_error_V"] = (
        df["predicted_voltage_V"] - df["voltage_V"]
    )
    df["soc_model_error_V"] = (
        df["predicted_voltage_v"] - df["voltage_V"]
    )

    # Diagnostic threshold, not a universal physical definition:
    # flag measured samples below 3.2 V.
    below_threshold = df["voltage_V"] < 3.2
    first_low = df.index[below_threshold][0] if below_threshold.any() else None

    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "B0005_cycle_2_voltage_knee.csv"
    report_path = out / "B0005_cycle_2_voltage_knee_report.txt"
    df.to_csv(csv_path, index=False)

    lines = [
        "NASA B0005 cycle 2: voltage-knee diagnostic",
        "============================================",
        f"Samples inspected: {len(df)}",
        f"Time range: {df['time_s'].min():.3f} to "
        f"{df['time_s'].max():.3f} s",
        f"Minimum measured voltage: {df['voltage_V'].min():.6f} V",
        f"Maximum discharge current: "
        f"{df['current_positive_discharge_A'].max():.6f} A",
        f"Final raw SOC: {df['soc_raw'].iloc[-1]:.6f}",
        f"Final bounded SOC: {df['soc_bounded'].iloc[-1]:.6f}",
    ]

    if first_low is not None:
        row = df.loc[first_low]
        lines += [
            "",
            "First measured sample below 3.2 V (diagnostic threshold):",
            f"Time: {row['time_s']:.3f} s",
            f"Voltage: {row['voltage_V']:.6f} V",
            f"Discharge current: "
            f"{row['current_positive_discharge_A']:.6f} A",
            f"Voltage slope: {row['voltage_slope_V_per_s']:.8f} V/s",
            f"Baseline error: {row['baseline_error_V']:.6f} V",
            f"SOC model error: {row['soc_model_error_V']:.6f} V",
        ]

    lines += [
        "",
        "Interpretation caution:",
        "- The 3.2 V threshold is diagnostic, not a validated cutoff.",
        "- Coulomb-counted SOC depends on initial SOC and capacity assumptions.",
        "- The illustrative OCV-SOC curve is not experimentally calibrated.",
        "- This diagnostic alone does not identify the physical cause of the knee.",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nDiagnostic CSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
