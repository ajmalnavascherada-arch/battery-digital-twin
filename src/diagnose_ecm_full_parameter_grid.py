
"""Full ECM parameter-grid diagnostic on active discharge.

Diagnostic only: this is not a calibrated parameter fit.
Existing scripts and results are not modified.
"""

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "results_nasa_original"
    / "ecm_soc_voltage"
    / "B0005_cycle_2_soc_ecm.csv"
)

OUT_DIR = ROOT / "results_nasa_original" / "ecm_full_parameter_grid"

# Broad, illustrative ranges for sensitivity testing.
R0_VALUES = [0.07, 0.09, 0.106110, 0.12, 0.15]
R1_VALUES = [0.0001, 0.001, 0.005, 0.01, 0.02]
TAU_VALUES = [1.0, 10.0, 50.0, 100.0, 300.0, 1000.0]


def predict(time_s, current_a, ocv_v, r0, r1, tau):
    vp = np.zeros_like(current_a)

    for k in range(1, len(current_a)):
        dt = time_s[k] - time_s[k - 1]
        decay = np.exp(-dt / tau)
        vp[k] = (
            decay * vp[k - 1]
            + r1 * (1.0 - decay) * current_a[k - 1]
        )

    return ocv_v - current_a * r0 - vp


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    data = pd.read_csv(INPUT)
    time_s = data["time_s"].to_numpy(float)
    current_a = data["current_discharge_positive_a"].to_numpy(float)
    measured_v = data["measured_voltage_v"].to_numpy(float)
    ocv_v = data["illustrative_ocv_v"].to_numpy(float)

    active = current_a >= 0.5
    if active.sum() < 10:
        raise ValueError("Too few active-discharge samples.")

    rows = []

    for r0, r1, tau in product(
        R0_VALUES, R1_VALUES, TAU_VALUES
    ):
        predicted_v = predict(
            time_s, current_a, ocv_v, r0, r1, tau
        )
        error = predicted_v[active] - measured_v[active]

        rows.append({
            "R0_ohm": r0,
            "R1_ohm": r1,
            "tau_s": tau,
            "RMSE_V": float(np.sqrt(np.mean(error**2))),
            "MAE_V": float(np.mean(np.abs(error))),
            "mean_error_V": float(np.mean(error)),
        })

    results = pd.DataFrame(rows).sort_values("RMSE_V")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUT_DIR / "B0005_cycle_2_full_parameter_grid.csv"
    results.to_csv(output, index=False)

    print("Full ECM parameter-grid diagnostic")
    print("==================================")
    print(f"Active samples: {int(active.sum())}")
    print(f"Combinations tested: {len(results)}")
    print("\nBest 15 combinations:")
    print(results.head(15).to_string(index=False))
    print("\nResults saved to:")
    print(output)
    print(
        "\nCaution: a low error does not establish physical identifiability. "
        "The OCV curve is illustrative and the data are used in-sample."
    )


if __name__ == "__main__":
    main()
