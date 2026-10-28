
"""Plot active-discharge ECM errors across fitted and test cycles."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results_nasa_original" / "ecm_cross_cycle_comparison"

# Cycle 2 metrics are from the comparison script's shared active samples.
# Cycle 4 and cycles 6, 10, 20 are fixed-parameter evaluations.
ROWS = [
    {"cycle": 2, "role": "Fit cycle (in-sample)",
     "RMSE_V": 0.056991, "MAE_V": 0.045319, "samples": 178},
    {"cycle": 4, "role": "Independent test",
     "RMSE_V": 0.061159134, "MAE_V": 0.047151883, "samples": 177},
    {"cycle": 6, "role": "Independent test",
     "RMSE_V": 0.063047, "MAE_V": 0.048148, "samples": 176},
    {"cycle": 10, "role": "Independent test",
     "RMSE_V": 0.063182, "MAE_V": 0.047728, "samples": 176},
    {"cycle": 20, "role": "Independent test",
     "RMSE_V": 0.065144, "MAE_V": 0.048263, "samples": 175},
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ROWS)
    csv_path = OUT_DIR / "B0005_cross_cycle_metrics.csv"
    plot_path = OUT_DIR / "B0005_cross_cycle_error_comparison.png"

    df.to_csv(csv_path, index=False)

    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, df["RMSE_V"] * 1000, width, label="RMSE")
    ax.bar(x + width / 2, df["MAE_V"] * 1000, width, label="MAE")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Cycle {c}" for c in df["cycle"]],
        rotation=0,
    )
    ax.set_ylabel("Voltage error (mV)")
    ax.set_title("NASA B0005: SOC-dependent ECM cross-cycle comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for i, row in df.iterrows():
        ax.text(
            i,
            max(row["RMSE_V"], row["MAE_V"]) * 1000 + 1,
            "Fit" if row["cycle"] == 2 else "Test",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print("Cross-cycle comparison")
    print("======================")
    print(df.to_string(index=False))
    print("\nInterpretation:")
    print("- Cycle 2 is an in-sample reference, not an independent test.")
    print("- Cycles 4, 6, 10, and 20 use fixed cycle-2 parameters.")
    print("- OCV-SOC curve is illustrative and not calibrated.")
    print(f"\nMetrics CSV: {csv_path}")
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()

