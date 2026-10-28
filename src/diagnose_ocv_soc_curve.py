
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "ecm_soc_voltage"
    / "B0005_cycle_2_soc_ecm.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "ocv_soc_diagnostic"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "time_s",
        "current_discharge_positive_a",
        "measured_voltage_v",
        "soc_bounded",
        "illustrative_ocv_v",
        "predicted_voltage_v",
        "voltage_error_v",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[required].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Focus the residual analysis on active discharge.
    active = df[
        df["current_discharge_positive_a"] >= 0.5
    ].copy()

    if active.empty:
        raise ValueError("No active-discharge samples found.")

    active["soc_percent"] = 100.0 * active["soc_bounded"]

    # Residual sign: positive means measured voltage exceeds prediction.
    active["residual_v"] = (
        active["measured_voltage_v"] - active["predicted_voltage_v"]
    )

    # Use 10%-wide SOC bins for a descriptive residual summary.
    bin_edges = np.arange(0, 110, 10)
    active["soc_bin"] = pd.cut(
        active["soc_percent"],
        bins=bin_edges,
        include_lowest=True,
    )

    summary = (
        active.groupby("soc_bin", observed=False)
        .agg(
            samples=("residual_v", "size"),
            mean_soc_percent=("soc_percent", "mean"),
            mean_current_a=("current_discharge_positive_a", "mean"),
            mean_measured_voltage_v=("measured_voltage_v", "mean"),
            mean_illustrative_ocv_v=("illustrative_ocv_v", "mean"),
            mean_predicted_voltage_v=("predicted_voltage_v", "mean"),
            mean_residual_v=("residual_v", "mean"),
            residual_rmse_v=(
                "residual_v",
                lambda x: float(np.sqrt(np.mean(x**2))) if len(x) else np.nan,
            ),
        )
        .reset_index()
    )

    summary_file = OUTPUT_DIR / "B0005_cycle_2_ocv_soc_residual_bins.csv"
    summary.to_csv(summary_file, index=False)

    # A descriptive scatter plot: not a direct OCV measurement.
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].scatter(
        active["soc_percent"],
        active["measured_voltage_v"],
        s=18,
        label="Measured terminal voltage",
    )
    axes[0].scatter(
        active["soc_percent"],
        active["illustrative_ocv_v"],
        s=18,
        label="Illustrative OCV estimate",
    )
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("B0005 cycle 2: voltage and illustrative OCV versus SOC")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(
        active["soc_percent"],
        1000.0 * active["residual_v"],
        s=20,
    )
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_xlabel("Estimated SOC (%)")
    axes[1].set_ylabel("Measured − predicted (mV)")
    axes[1].set_title("Active-discharge residual versus estimated SOC")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_file = OUTPUT_DIR / "B0005_cycle_2_ocv_soc_diagnostic.png"
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)

    residual = active["residual_v"].to_numpy()

    print("\nOCV-SOC RESIDUAL DIAGNOSTIC")
    print("===========================")
    print(f"Active-discharge samples: {len(active)}")
    print(f"Estimated SOC range: {active['soc_percent'].min():.2f}% "
          f"to {active['soc_percent'].max():.2f}%")
    print(f"Residual mean: {np.mean(residual):.6f} V")
    print(f"Residual RMSE: {np.sqrt(np.mean(residual**2)):.6f} V")
    print(f"Residual MAE: {np.mean(np.abs(residual)):.6f} V")
    print("\nResidual summary by SOC bin:")
    print(summary.to_string(index=False))
    print(f"\nSummary CSV: {summary_file}")
    print(f"Diagnostic plot: {plot_file}")

    print(
        "\nCAUTION: Terminal voltage under load is not OCV. "
        "This plot diagnoses model mismatch; it does not calibrate "
        "a true OCV-SOC curve."
    )


if __name__ == "__main__":
    main()
