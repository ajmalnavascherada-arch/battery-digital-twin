
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed_nasa_original" / "B0005_signals_long.csv"
OUT = ROOT / "results_nasa_original" / "soc_dependent_thermal_models"
OUT.mkdir(parents=True, exist_ok=True)

AMBIENT_C = 24.0
ACTIVE_CURRENT_A = 0.5
FIT_CYCLE = 2
TEST_CYCLES = [4, 6, 10, 20]


def load_cycle(cycle_number):
    raw = pd.read_csv(DATA)
    raw = raw[
        (raw["cycle_number"] == cycle_number)
        & (raw["cycle_type"] == "discharge")
    ].copy()

    if raw.empty:
        raise ValueError(f"No discharge data for cycle {cycle_number}.")

    wide = raw.pivot_table(
        index=["sample_index", "time_s"],
        columns="signal",
        values="value",
        aggfunc="first",
    ).reset_index().sort_values("time_s").reset_index(drop=True)

    required = ["Temperature_measured", "Current_measured"]
    missing = [c for c in required if c not in wide.columns]
    if missing:
        raise ValueError(f"Missing signals: {missing}")

    wide["discharge_current_a"] = (
        -wide["Current_measured"]
    ).clip(lower=0)

    # Use the recorded discharge capacity for this cycle.
    summary_path = (
        ROOT / "data" / "processed_nasa_original"
        / "B0005_cycle_summary.csv"
    )
    summary = pd.read_csv(summary_path)
    cap_rows = summary[
        (summary["cycle_number"] == cycle_number)
        & (summary["cycle_type"] == "discharge")
    ]

    if cap_rows.empty or "capacity_Ah" not in cap_rows.columns:
        raise ValueError(
            "Could not locate recorded discharge capacity. "
            "Check B0005_cycle_summary.csv column names."
        )

    capacity_ah = float(cap_rows["capacity_Ah"].iloc[0])
    if not np.isfinite(capacity_ah) or capacity_ah <= 0:
        raise ValueError(f"Invalid capacity for cycle {cycle_number}.")

    time_s = wide["time_s"].to_numpy(dtype=float)
    current_a = wide["discharge_current_a"].to_numpy(dtype=float)

    # SOC is estimated from current integration, starting at 100%.
    # It is an estimate, not a direct SOC measurement.
    soc = np.ones(len(wide), dtype=float)
    for i in range(1, len(wide)):
        dt = time_s[i] - time_s[i - 1]
        if dt <= 0:
            raise ValueError("Timestamps must be strictly increasing.")
        soc[i] = soc[i - 1] - (
            current_a[i - 1] * dt / (3600.0 * capacity_ah)
        )

    wide["soc_raw"] = soc
    wide["soc_bounded"] = np.clip(soc, 0.0, 1.0)
    wide["active_discharge"] = (
        wide["discharge_current_a"] >= ACTIVE_CURRENT_A
    )
    return wide


def simulate(df, params, soc_dependent):
    time_s = df["time_s"].to_numpy(dtype=float)
    current = df["discharge_current_a"].to_numpy(dtype=float)
    soc = df["soc_bounded"].to_numpy(dtype=float)
    measured = df["Temperature_measured"].to_numpy(dtype=float)

    if soc_dependent:
        a0, a1, b = params
    else:
        a0, b = params
        a1 = 0.0

    pred = np.empty(len(df), dtype=float)
    pred[0] = measured[0]

    for i in range(1, len(df)):
        dt = time_s[i] - time_s[i - 1]
        if dt <= 0:
            raise ValueError("Timestamps must be strictly increasing.")

        heating_coeff = a0 + a1 * (1.0 - soc[i - 1])
        decay = np.exp(-b * dt)
        equilibrium_rise = (
            heating_coeff * current[i - 1] ** 2 / b
        )

        pred[i] = (
            AMBIENT_C
            + (pred[i - 1] - AMBIENT_C) * decay
            + equilibrium_rise * (1.0 - decay)
        )

    return pred


def fit_model(df, soc_dependent):
    measured = df["Temperature_measured"].to_numpy(dtype=float)
    active = df["active_discharge"].to_numpy(dtype=bool)

    def residuals(params):
        pred = simulate(df, params, soc_dependent)
        return pred[active] - measured[active]

    if soc_dependent:
        x0 = [0.0015, 0.001, 0.0006]
        lower = [0.0, -0.01, 1e-7]
        upper = [0.02, 0.02, 0.01]
    else:
        x0 = [0.002, 0.0006]
        lower = [0.0, 1e-7]
        upper = [0.01, 0.01]

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        loss="soft_l1",
        max_nfev=30000,
    )

    if not result.success:
        print("Warning:", result.message)

    return result.x


def score(df, params, soc_dependent, cycle, role, model_name):
    measured = df["Temperature_measured"].to_numpy(dtype=float)
    active = df["active_discharge"].to_numpy(dtype=bool)
    pred = simulate(df, params, soc_dependent)
    error = pred - measured

    def metrics(mask):
        return {
            "rmse_c": float(np.sqrt(np.mean(error[mask] ** 2))),
            "mae_c": float(np.mean(np.abs(error[mask]))),
            "bias_c": float(np.mean(error[mask])),
        }

    active_metrics = metrics(active)
    full_metrics = metrics(np.ones(len(df), dtype=bool))

    result = {
        "model": model_name,
        "cycle_number": cycle,
        "role": role,
        "active_samples": int(active.sum()),
        "active_rmse_c": active_metrics["rmse_c"],
        "active_mae_c": active_metrics["mae_c"],
        "active_bias_c": active_metrics["bias_c"],
        "full_cycle_rmse_c": full_metrics["rmse_c"],
        "full_cycle_mae_c": full_metrics["mae_c"],
        "full_cycle_bias_c": full_metrics["bias_c"],
    }

    output = pd.DataFrame({
        "time_s": df["time_s"],
        "temperature_measured_c": measured,
        "temperature_predicted_c": pred,
        "discharge_current_a": df["discharge_current_a"],
        "soc_estimated": df["soc_bounded"],
        "residual_predicted_minus_measured_c": error,
    })

    output.to_csv(
        OUT / f"{model_name}_cycle_{cycle}_predictions.csv",
        index=False,
    )
    return result


reference = load_cycle(FIT_CYCLE)

params_a = fit_model(reference, soc_dependent=False)
params_b = fit_model(reference, soc_dependent=True)

print("=== FROZEN MODEL PARAMETERS FROM CYCLE 2 ===")
print(f"Constant heating model: a={params_a[0]:.9g}, b={params_a[1]:.9g}")
print(
    "SOC-dependent model: "
    f"a0={params_b[0]:.9g}, a1={params_b[1]:.9g}, "
    f"b={params_b[2]:.9g}"
)

rows = []
cycles = [FIT_CYCLE] + TEST_CYCLES

for cycle in cycles:
    df = reference if cycle == FIT_CYCLE else load_cycle(cycle)
    role = "fit_cycle" if cycle == FIT_CYCLE else "held_out_cycle"

    rows.append(score(
        df, params_a, False, cycle, role, "constant_heating"
    ))
    rows.append(score(
        df, params_b, True, cycle, role, "soc_dependent_heating"
    ))

results = pd.DataFrame(rows)
results.to_csv(OUT / "soc_thermal_model_comparison.csv", index=False)

print("\n=== MODEL COMPARISON ===")
print(results.round(5).to_string(index=False))

test = results[results["role"] == "held_out_cycle"]
pivot = test.pivot(
    index="cycle_number",
    columns="model",
    values="active_rmse_c",
)
if {"constant_heating", "soc_dependent_heating"}.issubset(pivot.columns):
    improvement = (
        1.0 - pivot["soc_dependent_heating"]
        / pivot["constant_heating"]
    ) * 100.0
    print("\nHeld-out active RMSE improvement (%):")
    print(improvement.round(2).to_string())

print(f"\nSaved results to: {OUT}")
