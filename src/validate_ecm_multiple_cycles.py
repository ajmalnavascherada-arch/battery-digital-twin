
"""Multi-cycle validation of the SOC-dependent Thevenin ECM.

Uses fixed parameters from the cycle-2 fit without refitting test cycles.
The OCV-SOC curve is illustrative, not experimentally calibrated.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SIGNALS = (
    PROJECT_ROOT
    / "data"
    / "processed_nasa_original"
    / "B0005_signals_long.csv"
)
SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed_nasa_original"
    / "B0005_cycle_summary.csv"
)
OUT_DIR = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "ecm_multi_validation"
)

TEST_CYCLES = [6, 10, 20]
INITIAL_SOC = 1.0
ACTIVE_CURRENT_A = 0.5

# Fixed parameters from the cycle-2 fit. Do not refit on test cycles.
R0_OHM = 0.106110
R1_OHM = 0.000100
TAU_S = 1.000002

SOC_KNOTS = np.array(
    [0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00]
)
OCV_KNOTS_V = np.array(
    [3.00, 3.30, 3.45, 3.58, 3.70, 3.82, 3.98, 4.10, 4.20]
)


def validate_cycle(signals, summary, cycle_number):
    """Validate one cycle without refitting model parameters."""

    capacity_rows = summary.loc[
        (summary["cycle_number"] == cycle_number)
        & (summary["cycle_type"] == "discharge"),
        "capacity_Ah",
    ]

    if capacity_rows.empty:
        raise ValueError(f"No discharge capacity for cycle {cycle_number}.")

    capacity_ah = float(capacity_rows.iloc[0])
    if not np.isfinite(capacity_ah) or capacity_ah <= 0:
        raise ValueError(f"Invalid capacity for cycle {cycle_number}.")

    df = signals.loc[
        (signals["cycle_number"] == cycle_number)
        & (signals["cycle_type"] == "discharge")
        & signals["signal"].isin(
            ["Voltage_measured", "Current_measured"]
        )
    ].copy()

    if df.empty:
        raise ValueError(f"No current/voltage signals for cycle {cycle_number}.")

    wide = df.pivot(
        index="sample_index",
        columns="signal",
        values="value",
    ).sort_index()

    time_rows = (
        df.loc[
            df["signal"] == "Voltage_measured",
            ["sample_index", "time_s", "time_aligned"],
        ]
        .drop_duplicates("sample_index")
        .set_index("sample_index")
    )

    wide = wide.join(time_rows, how="inner")
    wide = wide.dropna(
        subset=["Voltage_measured", "Current_measured", "time_s"]
    ).sort_values("time_s")

    aligned = wide["time_aligned"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    if not aligned.all():
        raise ValueError(
            f"Cycle {cycle_number} contains unaligned timestamps."
        )

    time_s = wide["time_s"].to_numpy(dtype=float)
    measured_v = wide["Voltage_measured"].to_numpy(dtype=float)

    # NASA's measured discharge current is negative.
    current_a = -wide["Current_measured"].to_numpy(dtype=float)

    if len(time_s) < 3:
        raise ValueError(f"Too few samples in cycle {cycle_number}.")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        raise ValueError(
            f"Cycle {cycle_number} timestamps are not strictly increasing."
        )
    if not np.all(np.isfinite(current_a)):
        raise ValueError(f"Non-finite current in cycle {cycle_number}.")

    # Coulomb counting, assuming 100% initial SOC.
    dt_s = np.diff(time_s)
    delta_ah = (
        0.5 * (current_a[:-1] + current_a[1:]) * dt_s / 3600.0
    )
    removed_ah = np.r_[0.0, np.cumsum(delta_ah)]
    soc_raw = INITIAL_SOC - removed_ah / capacity_ah
    soc = np.clip(soc_raw, 0.0, 1.0)

    ocv_v = np.interp(soc, SOC_KNOTS, OCV_KNOTS_V)

    # Simulate polarization over the full recorded cycle.
    vp_v = np.zeros_like(current_a)
    for k in range(1, len(current_a)):
        step_s = time_s[k] - time_s[k - 1]
        decay = np.exp(-step_s / TAU_S)
        vp_v[k] = (
            decay * vp_v[k - 1]
            + R1_OHM * (1.0 - decay) * current_a[k - 1]
        )

    predicted_v = ocv_v - current_a * R0_OHM - vp_v
    error_v = predicted_v - measured_v
    active = current_a >= ACTIVE_CURRENT_A

    if active.sum() < 10:
        raise ValueError(
            f"Too few active-discharge samples in cycle {cycle_number}."
        )

    active_error = error_v[active]
    rmse_v = float(np.sqrt(np.mean(active_error**2)))
    mae_v = float(np.mean(np.abs(active_error)))
    bias_v = float(np.mean(active_error))

    result = pd.DataFrame(
        {
            "cycle_number": cycle_number,
            "sample_index": wide.index.to_numpy(),
            "time_s": time_s,
            "current_discharge_positive_a": current_a,
            "measured_voltage_v": measured_v,
            "predicted_voltage_v": predicted_v,
            "voltage_error_v": error_v,
            "cumulative_removed_ah": removed_ah,
            "soc_raw": soc_raw,
            "soc_bounded": soc,
            "illustrative_ocv_v": ocv_v,
            "polarization_voltage_v": vp_v,
            "active_discharge": active,
        }
    )

    report_row = {
        "cycle_number": cycle_number,
        "capacity_Ah": capacity_ah,
        "total_samples": len(result),
        "active_samples": int(active.sum()),
        "integrated_discharge_Ah": float(removed_ah[-1]),
        "raw_final_SOC": float(soc_raw[-1]),
        "RMSE_V": rmse_v,
        "MAE_V": mae_v,
        "mean_error_V": bias_v,
    }

    return result, report_row


def main():
    if not SIGNALS.exists():
        raise FileNotFoundError(f"Missing signals file: {SIGNALS}")
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY}")

    signals = pd.read_csv(SIGNALS)
    summary = pd.read_csv(SUMMARY)

    all_results = []
    report_rows = []

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cycle_number in TEST_CYCLES:
        result, row = validate_cycle(signals, summary, cycle_number)
        all_results.append(result)
        report_rows.append(row)

        csv_path = OUT_DIR / f"B0005_cycle_{cycle_number}_validation.csv"
        plot_path = OUT_DIR / f"B0005_cycle_{cycle_number}_validation.png"

        result.to_csv(csv_path, index=False)

        active = result["active_discharge"].to_numpy(dtype=bool)
        t = result["time_s"].to_numpy(dtype=float)
        measured = result["measured_voltage_v"].to_numpy(dtype=float)
        predicted = result["predicted_voltage_v"].to_numpy(dtype=float)
        error = result["voltage_error_v"].to_numpy(dtype=float)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t[active], measured[active], label="Measured voltage")
        axes[0].plot(t[active], predicted[active], label="Fixed model")
        axes[0].set_ylabel("Voltage (V)")
        axes[0].set_title(f"B0005 recorded cycle {cycle_number}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t[active], error[active], label="Predicted - measured")
        axes[1].axhline(0.0, linewidth=1)
        axes[1].set_xlabel("Elapsed time (s)")
        axes[1].set_ylabel("Voltage error (V)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

    metrics = pd.DataFrame(report_rows)
    metrics_path = OUT_DIR / "B0005_multi_cycle_metrics.csv"
    report_path = OUT_DIR / "B0005_multi_cycle_report.txt"

    metrics.to_csv(metrics_path, index=False)

    lines = [
        "Multi-cycle SOC-dependent Thevenin ECM validation",
        "=" * 52,
        "Fixed parameters from cycle-2 fit; no test-cycle refitting.",
        f"Initial SOC assumption: {INITIAL_SOC:.3f}",
        f"Active-current threshold: {ACTIVE_CURRENT_A:.3f} A",
        f"R0: {R0_OHM:.6f} ohm",
        f"R1: {R1_OHM:.6f} ohm",
        f"Tau: {TAU_S:.6f} s",
        "",
        metrics.to_string(index=False),
        "",
        "Limitations:",
        "- OCV-SOC curve is illustrative, not experimentally calibrated.",
        "- Initial SOC is assumed to be 100%, not independently measured.",
        "- Parameters were fitted on cycle 2 and not refitted here.",
        "- Metrics cover active-discharge samples only.",
        "- Results are preliminary and not suitable for safety-critical control.",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nMetrics CSV: {metrics_path}")
    print(f"Report: {report_path}")
    print(f"Detailed cycle CSVs and plots: {OUT_DIR}")


if __name__ == "__main__":
    main()
