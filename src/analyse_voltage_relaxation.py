
"""Analyse near-zero-current voltage relaxation in NASA B0005 cycle 2.

This is a diagnostic, not an OCV calibration or a validated ECM fit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT
    / "results_nasa_original"
    / "ecm_soc_voltage"
    / "B0005_cycle_2_soc_ecm.csv"
)

OUT_DIR = ROOT / "results_nasa_original" / "voltage_relaxation"

CURRENT_THRESHOLD_A = 0.05
MIN_RELAXATION_TIME_S = 300.0


def exponential_relaxation(t, v_inf, amplitude, tau):
    """Simple one-exponential descriptive curve."""
    return v_inf - amplitude * np.exp(-t / tau)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    data = pd.read_csv(INPUT)

    current = data["current_discharge_positive_a"].to_numpy(float)
    time_s = data["time_s"].to_numpy(float)
    voltage = data["measured_voltage_v"].to_numpy(float)

    # Identify contiguous near-zero-current segments.
    rest = np.abs(current) < CURRENT_THRESHOLD_A
    indices = np.flatnonzero(rest)

    if len(indices) == 0:
        raise ValueError("No near-zero-current samples found.")

    # Split segments wherever sample indices are not consecutive.
    segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    segment_frames = []

    for segment_id, idx in enumerate(segments, start=1):
        if len(idx) < 3:
            continue

        t = time_s[idx]
        v = voltage[idx]
        elapsed = t - t[0]

        duration_s = float(t[-1] - t[0])
        delta_v = float(v[-1] - v[0])

        row = {
            "segment_id": segment_id,
            "samples": len(idx),
            "start_time_s": float(t[0]),
            "end_time_s": float(t[-1]),
            "duration_s": duration_s,
            "initial_voltage_v": float(v[0]),
            "final_voltage_v": float(v[-1]),
            "voltage_change_v": delta_v,
            "exponential_tau_s": np.nan,
            "exponential_rmse_v": np.nan,
        }

        segment = pd.DataFrame({
            "segment_id": segment_id,
            "time_s": t,
            "elapsed_since_segment_start_s": elapsed,
            "current_discharge_positive_a": current[idx],
            "measured_voltage_v": v,
        })

        # Fit only the long end-of-discharge segment for a descriptive
        # relaxation time constant. This does not establish equilibrium OCV.
        if duration_s >= MIN_RELAXATION_TIME_S and delta_v > 0:
            initial = [float(v[-1] + 0.01), float(v[-1] + 0.01 - v[0]), 100.0]

            try:
                params, _ = curve_fit(
                    exponential_relaxation,
                    elapsed,
                    v,
                    p0=initial,
                    bounds=(
                        [min(v) - 1.0, 0.0, 1.0],
                        [max(v) + 2.0, 5.0, 10000.0],
                    ),
                    maxfev=20000,
                )
                fitted = exponential_relaxation(elapsed, *params)
                row["exponential_tau_s"] = float(params[2])
                row["exponential_rmse_v"] = float(
                    np.sqrt(np.mean((fitted - v) ** 2))
                )
                segment["fitted_exponential_voltage_v"] = fitted
            except (RuntimeError, ValueError):
                segment["fitted_exponential_voltage_v"] = np.nan

        summary_rows.append(row)
        segment_frames.append(segment)

    if not summary_rows:
        raise ValueError("No usable near-zero-current segments found.")

    summary = pd.DataFrame(summary_rows)
    observations = pd.concat(segment_frames, ignore_index=True)

    summary_path = OUT_DIR / "B0005_cycle_2_relaxation_summary.csv"
    observations_path = OUT_DIR / "B0005_cycle_2_relaxation_samples.csv"
    plot_path = OUT_DIR / "B0005_cycle_2_voltage_relaxation.png"

    summary.to_csv(summary_path, index=False)
    observations.to_csv(observations_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    for segment_id, group in observations.groupby("segment_id"):
        ax.plot(
            group["elapsed_since_segment_start_s"],
            group["measured_voltage_v"],
            marker="o",
            label=f"Segment {segment_id}: measured",
        )
        if (
            "fitted_exponential_voltage_v" in group.columns
            and group["fitted_exponential_voltage_v"].notna().any()
        ):
            ax.plot(
                group["elapsed_since_segment_start_s"],
                group["fitted_exponential_voltage_v"],
                linestyle="--",
                label=f"Segment {segment_id}: exponential fit",
            )

    ax.set_xlabel("Time since rest segment began (s)")
    ax.set_ylabel("Measured voltage (V)")
    ax.set_title("NASA B0005 cycle 2: near-zero-current voltage relaxation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("Voltage relaxation diagnostic")
    print("=============================")
    print(summary.to_string(index=False))
    print("\nOutputs:")
    print(summary_path)
    print(observations_path)
    print(plot_path)
    print(
        "\nCaution: fitted exponential time constants are descriptive "
        "and are not automatically equivalent to the ECM tau."
    )


if __name__ == "__main__":
    main()
