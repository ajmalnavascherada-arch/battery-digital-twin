
"""Sensitivity diagnostic for the SOC-dependent Thevenin ECM.

This script does not fit parameters or modify existing model files.
It compares selected R1/tau combinations on active-discharge samples.
The illustrative OCV-SOC curve remains unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT
    / "results_nasa_original"
    / "ecm_soc_voltage"
    / "B0005_cycle_2_soc_ecm.csv"
)

OUT_DIR = ROOT / "results_nasa_original" / "ecm_parameter_sensitivity"

# Keep R0 fixed at the previously fitted value.
R0_OHM = 0.106110
ACTIVE_CURRENT_A = 0.5

# Explore a broad range rather than assuming the old optimum is correct.
R1_VALUES_OHM = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05]
TAU_VALUES_S = [1.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def simulate_voltage(
    time_s: np.ndarray,
    current_a: np.ndarray,
    ocv_v: np.ndarray,
    r1_ohm: float,
    tau_s: float,
    r0_ohm: float,
) -> np.ndarray:
    """Simulate voltage using the same first-order polarization equation."""
    vp_v = np.zeros_like(current_a)

    for k in range(1, len(current_a)):
        dt_s = time_s[k] - time_s[k - 1]
        decay = np.exp(-dt_s / tau_s)
        vp_v[k] = (
            decay * vp_v[k - 1]
            + r1_ohm * (1.0 - decay) * current_a[k - 1]
        )

    return ocv_v - current_a * r0_ohm - vp_v


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing model output: {INPUT}")

    data = pd.read_csv(INPUT)

    required = [
        "time_s",
        "current_discharge_positive_a",
        "measured_voltage_v",
        "illustrative_ocv_v",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    time_s = data["time_s"].to_numpy(float)
    current_a = data["current_discharge_positive_a"].to_numpy(float)
    measured_v = data["measured_voltage_v"].to_numpy(float)
    ocv_v = data["illustrative_ocv_v"].to_numpy(float)

    active = current_a >= ACTIVE_CURRENT_A
    if active.sum() < 10:
        raise ValueError("Too few active-discharge samples.")

    rows = []

    for r1 in R1_VALUES_OHM:
        for tau in TAU_VALUES_S:
            predicted_v = simulate_voltage(
                time_s=time_s,
                current_a=current_a,
                ocv_v=ocv_v,
                r1_ohm=r1,
                tau_s=tau,
                r0_ohm=R0_OHM,
            )

            error_v = predicted_v[active] - measured_v[active]

            rows.append(
                {
                    "R0_ohm": R0_OHM,
                    "R1_ohm": r1,
                    "tau_s": tau,
                    "active_samples": int(active.sum()),
                    "RMSE_V": float(np.sqrt(np.mean(error_v**2))),
                    "MAE_V": float(np.mean(np.abs(error_v))),
                    "mean_error_V": float(np.mean(error_v)),
                }
            )

    results = pd.DataFrame(rows).sort_values("RMSE_V")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "B0005_cycle_2_parameter_sensitivity.csv"
    results.to_csv(csv_path, index=False)

    print("ECM parameter sensitivity diagnostic")
    print("====================================")
    print(f"Input samples: {len(data)}")
    print(f"Active-discharge samples: {int(active.sum())}")
    print(f"Fixed R0: {R0_OHM:.6f} ohm")
    print("\nBest tested combinations (active-discharge RMSE):")
    print(results.head(10).to_string(index=False))
    print("\nFull grid saved to:")
    print(csv_path)
    print(
        "\nInterpretation caution: this is a finite grid search, "
        "not a calibrated parameter fit."
    )


if __name__ == "__main__":
    main()
