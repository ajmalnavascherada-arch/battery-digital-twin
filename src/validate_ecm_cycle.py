
"""Independent-cycle validation of the SOC-dependent Thevenin ECM.

Parameters are taken from the cycle-2 fit and are NOT refitted on cycle 4.
The OCV-SOC curve remains illustrative and is not experimentally calibrated.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIGNALS = Path(
    "data/processed_nasa_original/B0005_signals_long.csv"
)
SUMMARY = Path(
    "data/processed_nasa_original/B0005_cycle_summary.csv"
)
OUT_DIR = Path("results_nasa_original/ecm_validation")

TEST_CYCLE = 4
INITIAL_SOC = 1.0
ACTIVE_CURRENT_A = 0.5

# Fixed parameters from the cycle-2 SOC-dependent ECM fit.
# They are intentionally not refitted on cycle 4.
R0_OHM = 0.106110
R1_OHM = 0.000100
TAU_S = 1.000002

# Same illustrative OCV-SOC curve used in the cycle-2 model.
SOC_KNOTS = np.array(
    [0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00]
)
OCV_KNOTS_V = np.array(
    [3.00, 3.30, 3.45, 3.58, 3.70, 3.82, 3.98, 4.10, 4.20]
)


def main():
    if not SIGNALS.exists():
        raise FileNotFoundError(f"Missing signals file: {SIGNALS}")
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing cycle summary: {SUMMARY}")

    summary = pd.read_csv(SUMMARY)
    capacity_rows = summary.loc[
        (summary["cycle_number"] == TEST_CYCLE)
        & (summary["cycle_type"] == "discharge"),
        "capacity_Ah",
    ]
    if capacity_rows.empty or not np.isfinite(float(capacity_rows.iloc[0])):
        raise ValueError(
            f"No valid discharge capacity found for cycle {TEST_CYCLE}."
        )
    capacity_ah = float(capacity_rows.iloc[0])
    if capacity_ah <= 0:
        raise ValueError("Test-cycle capacity must be positive.")

    df = pd.read_csv(SIGNALS)
    df = df.loc[
        (df["cycle_number"] == TEST_CYCLE)
        & (df["cycle_type"] == "discharge")
        & (
            df["signal"].isin(
                ["Voltage_measured", "Current_measured"]
            )
        )
    ].copy()

    if df.empty:
        raise ValueError(f"No signals found for cycle {TEST_CYCLE}.")

    wide = df.pivot(
        index="sample_index",
        columns="signal",
        values="value",
    ).sort_index()

    required = ["Voltage_measured", "Current_measured"]
    if any(column not in wide.columns for column in required):
        raise ValueError("Voltage or current measurements are missing.")

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

    aligned = (
        wide["time_aligned"].astype(str).str.lower().isin(["true", "1"])
    )
    if not aligned.all():
        raise ValueError("Some test-cycle samples lack aligned timestamps.")

    time_s = wide["time_s"].to_numpy(float)
    measured_v = wide["Voltage_measured"].to_numpy(float)

    # NASA records discharge current as negative; convert to positive.
    current_a = -wide["Current_measured"].to_numpy(float)

    if len(time_s) < 3:
        raise ValueError("Not enough samples to validate.")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing.")
    if not np.all(np.isfinite(current_a)):
        raise ValueError("Current contains non-finite values.")

    # Coulomb-count SOC, using cycle 4's recorded capacity.
    dt_s = np.diff(time_s)
    delta_ah = (
        0.5 * (current_a[:-1] + current_a[1:]) * dt_s / 3600.0
    )
    removed_ah = np.r_[0.0, np.cumsum(delta_ah)]
    soc_raw = INITIAL_SOC - removed_ah / capacity_ah
    soc = np.clip(soc_raw, 0.0, 1.0)

    # Illustrative OCV lookup. No calibration or fitting on cycle 4.
    ocv_v = np.interp(
        soc,
        SOC_KNOTS,
        OCV_KNOTS_V,
    )

    # Simulate polarization state over the complete cycle, including
    # low-current samples. Keep parameters fixed from cycle 2.
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

    # Evaluate only samples under substantial discharge current.
    active = current_a >= ACTIVE_CURRENT_A
    if active.sum() < 10:
        raise ValueError("Too few active-discharge samples for evaluation.")

    active_error = error_v[active]
    rmse_v = float(np.sqrt(np.mean(active_error**2)))
    mae_v = float(np.mean(np.abs(active_error)))
    bias_v = float(np.mean(active_error))

    result = pd.DataFrame(
        {
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"B0005_cycle_{TEST_CYCLE}_validation.csv"
    plot_path = OUT_DIR / f"B0005_cycle_{TEST_CYCLE}_validation.png"
    report_path = OUT_DIR / f"B0005_cycle_{TEST_CYCLE}_validation_report.txt"

    result.to_csv(csv_path, index=False)

    active_time = time_s[active]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(active_time, measured_v[active], label="Measured voltage")
    axes[0].plot(active_time, predicted_v[active], label="Fixed cycle-2 model")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title(
        f"Independent validation: B0005 recorded cycle {TEST_CYCLE}"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(active_time, error_v[active], label="Predicted − measured")
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("Voltage error (V)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    report = f"""Independent-cycle SOC-dependent ECM validation
================================================
Test cycle: {TEST_CYCLE}
Total aligned samples: {len(result)}
Active-discharge samples evaluated: {int(active.sum())}
Active-current threshold: {ACTIVE_CURRENT_A:.3f} A
Time range evaluated: {active_time.min():.3f} to {active_time.max():.3f} s

Initial SOC assumption: {INITIAL_SOC:.4f}
Cycle-specific recorded capacity: {capacity_ah:.9f} Ah
Integrated discharge: {removed_ah[-1]:.9f} Ah
Raw final SOC: {soc_raw[-1]:.6f}
Bounded final SOC: {soc[-1]:.6f}

Fixed parameters from cycle-2 fit:
R0: {R0_OHM:.6f} ohm
R1: {R1_OHM:.6f} ohm
Tau: {TAU_S:.6f} s

Active-discharge metrics (no parameter refit):
RMSE: {rmse_v:.6f} V
MAE: {mae_v:.6f} V
Mean error (predicted - measured): {bias_v:.6f} V

Limitations:
- The OCV-SOC curve is illustrative and not experimentally calibrated.
- Initial SOC is assumed, not independently measured.
- Parameters were fitted on cycle 2 and were not refitted here.
- Results assess this cycle under stated assumptions, not all batteries.
- This prototype is not suitable for safety-critical control.
"""
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
