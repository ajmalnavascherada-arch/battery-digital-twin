
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed_nasa_original" / "B0005_signals_long.csv"
OUT = ROOT / "results_nasa_original" / "effective_thermal_model_exact"
OUT.mkdir(parents=True, exist_ok=True)

AMBIENT_C = 24.0
ACTIVE_CURRENT_A = 0.5
FIT_CYCLE = 2
TEST_CYCLES = [4, 6, 10, 20]


def load_cycle(cycle_number):
    df = pd.read_csv(DATA)
    df = df[
        (df["cycle_number"] == cycle_number)
        & (df["cycle_type"] == "discharge")
    ].copy()

    if df.empty:
        raise ValueError(f"No discharge data for cycle {cycle_number}.")

    wide = df.pivot_table(
        index=["sample_index", "time_s"],
        columns="signal",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide = wide.sort_values("time_s").reset_index(drop=True)

    required = ["Temperature_measured", "Current_measured"]
    missing = [name for name in required if name not in wide.columns]
    if missing:
        raise ValueError(f"Missing signals: {missing}")

    wide["discharge_current_a"] = (
        -wide["Current_measured"]
    ).clip(lower=0)

    return wide


def simulate_exact(time_s, current_a, initial_temp_c, a, b):
    """
    Exact update for constant current over each sampling interval:
        dT/dt = a*I^2 - b*(T - ambient)
    """
    time_s = np.asarray(time_s, dtype=float)
    current_a = np.asarray(current_a, dtype=float)

    predicted = np.empty(len(time_s))
    predicted[0] = initial_temp_c

    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]
        if dt <= 0:
            raise ValueError("Time values must be strictly increasing.")

        decay = np.exp(-b * dt)
        equilibrium_rise = a * current_a[i - 1] ** 2 / b

        predicted[i] = (
            AMBIENT_C
            + (predicted[i - 1] - AMBIENT_C) * decay
            + equilibrium_rise * (1.0 - decay)
        )

    return predicted


def active_mask(df):
    return df["discharge_current_a"].to_numpy() >= ACTIVE_CURRENT_A


def fit_model(df):
    time_s = df["time_s"].to_numpy(dtype=float)
    current_a = df["discharge_current_a"].to_numpy(dtype=float)
    measured = df["Temperature_measured"].to_numpy(dtype=float)
    active = active_mask(df)
    initial_temp = float(measured[0])

    def residuals(params):
        a, b = params
        pred = simulate_exact(time_s, current_a, initial_temp, a, b)
        return pred[active] - measured[active]

    result = least_squares(
        residuals,
        x0=[0.002, 0.0006],
        bounds=([0.0, 1e-7], [0.01, 0.01]),
        loss="soft_l1",
        max_nfev=20000,
    )

    if not result.success:
        print("Warning:", result.message)

    return result.x


def score_cycle(df, a, b, cycle, role):
    time_s = df["time_s"].to_numpy(dtype=float)
    current_a = df["discharge_current_a"].to_numpy(dtype=float)
    measured = df["Temperature_measured"].to_numpy(dtype=float)

    predicted = simulate_exact(
        time_s, current_a, float(measured[0]), a, b
    )

    active = active_mask(df)
    error = predicted - measured

    def score(mask):
        return {
            "rmse_c": float(np.sqrt(np.mean(error[mask] ** 2))),
            "mae_c": float(np.mean(np.abs(error[mask]))),
            "bias_c": float(np.mean(error[mask])),
        }

    row = {
        "cycle_number": cycle,
        "role": role,
        "a_effective": a,
        "b_effective_per_s": b,
        "active_rmse_c": score(active)["rmse_c"],
        "active_mae_c": score(active)["mae_c"],
        "active_bias_c": score(active)["bias_c"],
        "full_cycle_rmse_c": score(np.ones(len(df), dtype=bool))["rmse_c"],
        "full_cycle_mae_c": score(np.ones(len(df), dtype=bool))["mae_c"],
        "full_cycle_bias_c": score(np.ones(len(df), dtype=bool))["bias_c"],
    }

    pd.DataFrame({
        "time_s": time_s,
        "temperature_measured_c": measured,
        "temperature_predicted_c": predicted,
        "discharge_current_a": current_a,
        "residual_predicted_minus_measured_c": error,
    }).to_csv(
        OUT / f"B0005_cycle_{cycle}_predictions.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time_s, measured, label="Measured")
    ax.plot(time_s, predicted, "--", label="Exact-update model")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Exact-update thermal model — cycle {cycle}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"B0005_cycle_{cycle}_comparison.png", dpi=160)
    plt.close(fig)

    return row


reference = load_cycle(FIT_CYCLE)
a_fit, b_fit = fit_model(reference)

print("=== EXACT-UPDATE EFFECTIVE THERMAL MODEL ===")
print(f"Fitted cycle: {FIT_CYCLE}")
print(f"a = {a_fit:.9g}")
print(f"b = {b_fit:.9g} 1/s")
print(f"Time scale 1/b = {1 / b_fit:.2f} s ({1 / b_fit / 60:.2f} min)")

rows = [
    score_cycle(reference, a_fit, b_fit, FIT_CYCLE, "fit_cycle")
]

for cycle in TEST_CYCLES:
    rows.append(
        score_cycle(
            load_cycle(cycle), a_fit, b_fit, cycle, "held_out_cycle"
        )
    )

results = pd.DataFrame(rows)
results.to_csv(OUT / "exact_thermal_model_metrics.csv", index=False)

print("\nCross-cycle metrics:")
print(results.round(5).to_string(index=False))
print(f"\nSaved outputs to: {OUT}")
