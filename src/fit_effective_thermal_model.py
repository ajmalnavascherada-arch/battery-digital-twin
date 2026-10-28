
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed_nasa_original" / "B0005_signals_long.csv"
OUT = ROOT / "results_nasa_original" / "effective_thermal_model"
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
        raise ValueError(f"No discharge rows for cycle {cycle_number}.")

    wide = df.pivot_table(
        index=["sample_index", "time_s"],
        columns="signal",
        values="value",
        aggfunc="first",
    ).reset_index()

    required = ["Temperature_measured", "Current_measured"]
    missing = [name for name in required if name not in wide.columns]
    if missing:
        raise ValueError(f"Missing signals: {missing}")

    wide = wide.sort_values("time_s").reset_index(drop=True)
    wide["discharge_current_a"] = (
        -wide["Current_measured"]
    ).clip(lower=0)

    # Integrate through the full cycle, including the relaxation period.
    # The model will use zero heating input when current is near zero.
    wide["heating_input_a2"] = wide["discharge_current_a"] ** 2
    return wide


def simulate_temperature(time_s, current_a, initial_temp_c, a, b):
    """Forward-Euler simulation of dT/dt = a*I^2 - b*(T - ambient)."""
    time_s = np.asarray(time_s, dtype=float)
    current_a = np.asarray(current_a, dtype=float)

    predicted = np.empty(len(time_s), dtype=float)
    predicted[0] = initial_temp_c

    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]
        if dt <= 0:
            raise ValueError("Timestamps must be strictly increasing.")

        # Explicit Euler step using the preceding sample's current.
        derivative = (
            a * current_a[i - 1] ** 2
            - b * (predicted[i - 1] - AMBIENT_C)
        )
        predicted[i] = predicted[i - 1] + dt * derivative

    return predicted


def active_mask(df):
    return df["discharge_current_a"].to_numpy() >= ACTIVE_CURRENT_A


def fit_model(df):
    time_s = df["time_s"].to_numpy(dtype=float)
    current_a = df["discharge_current_a"].to_numpy(dtype=float)
    measured = df["Temperature_measured"].to_numpy(dtype=float)
    active = active_mask(df)

    if active.sum() < 20:
        raise ValueError("Not enough active-discharge samples.")

    # Fit the heating and cooling coefficients on cycle 2.
    # Use a stable bounded optimizer with a fixed measured initial temperature.
    initial_temp = float(measured[0])

    def residuals(params):
        a, b = params
        pred = simulate_temperature(time_s, current_a, initial_temp, a, b)
        return (pred[active] - measured[active])

    result = least_squares(
        residuals,
        x0=[1e-4, 1e-4],
        bounds=([0.0, 1e-7], [0.01, 0.01]),
        loss="soft_l1",
        max_nfev=20000,
    )

    if not result.success:
        print("Warning: optimizer did not report success:", result.message)

    return result.x


def score_cycle(df, a, b, cycle_number, role):
    time_s = df["time_s"].to_numpy(dtype=float)
    current_a = df["discharge_current_a"].to_numpy(dtype=float)
    measured = df["Temperature_measured"].to_numpy(dtype=float)

    # Measured initial temperature is used as the starting condition.
    predicted = simulate_temperature(
        time_s, current_a, float(measured[0]), a, b
    )

    active = active_mask(df)
    error = predicted - measured

    def calc(mask):
        if not np.any(mask):
            return {"rmse_c": np.nan, "mae_c": np.nan, "bias_c": np.nan}
        return {
            "rmse_c": float(np.sqrt(np.mean(error[mask] ** 2))),
            "mae_c": float(np.mean(np.abs(error[mask]))),
            "bias_c": float(np.mean(error[mask])),
        }

    row = {
        "cycle_number": cycle_number,
        "role": role,
        "a_effective": a,
        "b_effective_per_s": b,
        "total_samples": len(df),
        "active_samples": int(active.sum()),
        "active_rmse_c": calc(active)["rmse_c"],
        "active_mae_c": calc(active)["mae_c"],
        "active_bias_c": calc(active)["bias_c"],
        "full_cycle_rmse_c": calc(np.ones(len(df), dtype=bool))["rmse_c"],
        "full_cycle_mae_c": calc(np.ones(len(df), dtype=bool))["mae_c"],
        "full_cycle_bias_c": calc(np.ones(len(df), dtype=bool))["bias_c"],
    }

    prediction_frame = pd.DataFrame({
        "time_s": time_s,
        "temperature_measured_c": measured,
        "temperature_predicted_c": predicted,
        "discharge_current_a": current_a,
        "phase": np.where(active, "active_discharge", "not_active"),
        "residual_predicted_minus_measured_c": error,
    })

    prediction_frame.to_csv(
        OUT / f"B0005_cycle_{cycle_number}_thermal_predictions.csv",
        index=False,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time_s, measured, label="Measured temperature")
    axes[0].plot(time_s, predicted, "--", label="Model prediction")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"Effective thermal model — cycle {cycle_number}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time_s, error)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Prediction error (°C)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUT / f"B0005_cycle_{cycle_number}_thermal_predictions.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    return row


reference = load_cycle(FIT_CYCLE)
a_fit, b_fit = fit_model(reference)

print("=== EFFECTIVE THERMAL MODEL ===")
print(f"Fit cycle: {FIT_CYCLE}")
print(f"Effective heating coefficient a: {a_fit:.8g}")
print(f"Effective cooling coefficient b: {b_fit:.8g} 1/s")
if b_fit > 0:
    print(f"Effective cooling time scale: {1 / b_fit:.2f} s "
          f"({1 / b_fit / 60:.2f} min)")

rows = []
rows.append(
    score_cycle(reference, a_fit, b_fit, FIT_CYCLE, "fit_cycle")
)

for cycle in TEST_CYCLES:
    test_df = load_cycle(cycle)
    rows.append(
        score_cycle(test_df, a_fit, b_fit, cycle, "held_out_cycle")
    )

results = pd.DataFrame(rows)
results.to_csv(OUT / "effective_thermal_model_metrics.csv", index=False)

print("\nCross-cycle metrics:")
print(results.round(5).to_string(index=False))
print(f"\nSaved outputs to: {OUT}")
