
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PRED_DIR = (
    ROOT / "results_nasa_original"
    / "soc_dependent_thermal_models"
)

OUT = ROOT / "results_nasa_original" / "hybrid_thermal_model"
OUT.mkdir(parents=True, exist_ok=True)

CYCLES = [2, 4, 6, 10, 20]
ACTIVE_CURRENT_A = 0.5

# Frozen cooling parameter fitted previously on cycle 2.
TAU_COOL_S = 974.0912
AMBIENT_C = 24.0


def evaluate_cycle(cycle):
    path = PRED_DIR / f"soc_dependent_heating_cycle_{cycle}_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path).sort_values("time_s").reset_index(drop=True)

    time = df["time_s"].to_numpy(dtype=float)
    current = df["discharge_current_a"].to_numpy(dtype=float)
    measured = df["temperature_measured_c"].to_numpy(dtype=float)
    active = current >= ACTIVE_CURRENT_A

    if not active.any():
        raise ValueError(f"No active-discharge samples in cycle {cycle}")

    last_active = int(np.flatnonzero(active)[-1])

    # Preserve the existing SOC-dependent model's predictions during discharge.
    hybrid = df["temperature_predicted_c"].to_numpy(dtype=float).copy()

    # Anchor cooling at the measured temperature at the end of active discharge.
    # This isolates cooling-shape validation from end-of-discharge bias.
    t_anchor = time[last_active]
    T_anchor = measured[last_active]

    for i in range(last_active + 1, len(df)):
        dt = time[i] - t_anchor
        hybrid[i] = AMBIENT_C + (
            T_anchor - AMBIENT_C
        ) * np.exp(-dt / TAU_COOL_S)

    error = hybrid - measured

    def score(mask):
        return {
            "rmse_c": float(np.sqrt(np.mean(error[mask] ** 2))),
            "mae_c": float(np.mean(np.abs(error[mask]))),
            "bias_c": float(np.mean(error[mask])),
        }

    post_mask = np.arange(len(df)) > last_active

    result = {
        "cycle_number": cycle,
        "active_samples": int(active.sum()),
        "post_samples": int(post_mask.sum()),
        "active_rmse_c": score(active)["rmse_c"],
        "active_mae_c": score(active)["mae_c"],
        "active_bias_c": score(active)["bias_c"],
        "post_rmse_c": score(post_mask)["rmse_c"] if post_mask.any() else np.nan,
        "post_mae_c": score(post_mask)["mae_c"] if post_mask.any() else np.nan,
        "post_bias_c": score(post_mask)["bias_c"] if post_mask.any() else np.nan,
        "full_rmse_c": score(np.ones(len(df), dtype=bool))["rmse_c"],
        "full_mae_c": score(np.ones(len(df), dtype=bool))["mae_c"],
        "full_bias_c": score(np.ones(len(df), dtype=bool))["bias_c"],
    }

    output = df[[
        "time_s",
        "temperature_measured_c",
        "discharge_current_a",
    ]].copy()
    output["hybrid_temperature_predicted_c"] = hybrid
    output["residual_predicted_minus_measured_c"] = error
    output["phase"] = np.where(active, "active_discharge", "post_discharge")

    output.to_csv(
        OUT / f"B0005_cycle_{cycle}_hybrid_predictions.csv",
        index=False,
    )

    return result


results = pd.DataFrame([evaluate_cycle(c) for c in CYCLES])
results.to_csv(OUT / "hybrid_thermal_metrics.csv", index=False)

print("=== HYBRID THERMAL MODEL VALIDATION ===")
print(f"Frozen post-discharge cooling tau: {TAU_COOL_S:.2f} s")
print("\nMetrics:")
print(results.round(4).to_string(index=False))
print(f"\nSaved results to: {OUT}")
