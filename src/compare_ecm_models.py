
"""Compare baseline and SOC-dependent ECM on identical active samples.

Existing model scripts and result files are read-only inputs.
Metrics are in-sample and are not independent validation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_FILE = Path(
    "results_nasa_original/ecm/B0005_cycle_2_ecm_predictions.csv"
)
SOC_FILE = Path(
    "results_nasa_original/ecm_soc_voltage/B0005_cycle_2_soc_ecm.csv"
)
OUT_DIR = Path("results_nasa_original/ecm_comparison")
CYCLE = 2
ACTIVE_CURRENT_THRESHOLD_A = 0.5


def load_and_validate() -> pd.DataFrame:
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(f"Missing baseline file: {BASELINE_FILE}")
    if not SOC_FILE.exists():
        raise FileNotFoundError(f"Missing SOC model file: {SOC_FILE}")

    baseline = pd.read_csv(BASELINE_FILE)
    soc_model = pd.read_csv(SOC_FILE)

    baseline_required = {
        "time_s",
        "voltage_V",
        "current_positive_discharge_A",
        "predicted_voltage_V",
    }
    soc_required = {
        "sample_index",
        "time_s",
        "current_discharge_positive_a",
        "measured_voltage_v",
        "predicted_voltage_v",
    }

    missing_base = baseline_required - set(baseline.columns)
    missing_soc = soc_required - set(soc_model.columns)

    if missing_base:
        raise ValueError(
            f"Baseline CSV missing columns: {sorted(missing_base)}"
        )
    if missing_soc:
        raise ValueError(
            f"SOC model CSV missing columns: {sorted(missing_soc)}"
        )

    # Baseline is the 178-sample active segment; the SOC model contains
    # the full cycle. Align each baseline sample to the nearest exact
    # timestamp in the full-cycle data.
    baseline = baseline.copy()
    soc_model = soc_model.copy()
    baseline["time_s"] = pd.to_numeric(baseline["time_s"], errors="raise")
    soc_model["time_s"] = pd.to_numeric(soc_model["time_s"], errors="raise")

    merged = pd.merge(
        baseline,
        soc_model,
        on="time_s",
        how="inner",
        suffixes=("_base", "_soc"),
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No identical timestamps found. Do not compare the models "
            "until their sample alignment is resolved."
        )

    # Ensure this is a genuinely active discharge interval.
    current = merged["current_positive_discharge_A"].to_numpy(float)
    active = current >= ACTIVE_CURRENT_THRESHOLD_A
    merged = merged.loc[active].copy()

    if len(merged) < 10:
        raise ValueError(
            f"Only {len(merged)} shared active samples found; "
            "inspect timestamp and current alignment."
        )

    merged["baseline_error_V"] = (
        merged["predicted_voltage_V"] - merged["voltage_V"]
    )
    merged["soc_error_V"] = (
        merged["predicted_voltage_v"] - merged["measured_voltage_v"]
    )

    # Both files should refer to the same measured voltage.
    voltage_difference = np.max(
        np.abs(
            merged["voltage_V"].to_numpy(float)
            - merged["measured_voltage_v"].to_numpy(float)
        )
    )
    if voltage_difference > 1e-6:
        raise ValueError(
            "Measured voltages differ between files; alignment needs review. "
            f"Maximum difference: {voltage_difference:.8g} V"
        )

    return merged.sort_values("time_s").reset_index(drop=True)


def metrics(error: np.ndarray) -> dict[str, float]:
    return {
        "RMSE_V": float(np.sqrt(np.mean(error**2))),
        "MAE_V": float(np.mean(np.abs(error))),
        "Mean_error_V": float(np.mean(error)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_validate()

    baseline_error = data["baseline_error_V"].to_numpy(float)
    soc_error = data["soc_error_V"].to_numpy(float)

    rows = []
    for name, error in [
        ("Baseline constant-OCV ECM", baseline_error),
        ("SOC-dependent ECM", soc_error),
    ]:
        row = {"model": name, "samples": len(data)}
        row.update(metrics(error))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "B0005_cycle_2_model_comparison.csv"
    aligned_path = OUT_DIR / "B0005_cycle_2_aligned_predictions.csv"
    plot_path = OUT_DIR / "B0005_cycle_2_residual_comparison.png"
    report_path = OUT_DIR / "B0005_cycle_2_comparison_report.txt"

    summary.to_csv(summary_path, index=False)
    data.to_csv(aligned_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        data["time_s"],
        baseline_error,
        label="Baseline ECM residual",
    )
    ax.plot(
        data["time_s"],
        soc_error,
        label="SOC-dependent ECM residual",
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Predicted − measured voltage (V)")
    ax.set_title("Cycle 2: residual comparison on shared samples")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    report_lines = [
        "NASA B0005 cycle 2: ECM comparison",
        "=================================",
        f"Shared active samples: {len(data)}",
        f"Time range: {data['time_s'].min():.3f} to "
        f"{data['time_s'].max():.3f} s",
        f"Active current threshold: {ACTIVE_CURRENT_THRESHOLD_A:.2f} A",
        "",
        summary.to_string(index=False),
        "",
        "Limitations:",
        "- Both models were fitted on this same cycle; this is in-sample.",
        "- The SOC-dependent OCV-SOC curve is illustrative, not calibrated.",
        "- This comparison uses the shared active interval only.",
        "- Lower error does not establish physical validity or generalization.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nSummary CSV: {summary_path}")
    print(f"Aligned predictions: {aligned_path}")
    print(f"Residual plot: {plot_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
