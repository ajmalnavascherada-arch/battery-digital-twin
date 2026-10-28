
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed_nasa_original"
    / "B0005_signals_long.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "thermal_model"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CYCLE_NUMBER = 2
MIN_CURRENT_A = 0.5

df = pd.read_csv(INPUT_FILE)

signals = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
]

wide = (
    df[df["signal"].isin(signals)]
    .pivot(
        index=["cycle_number", "cycle_type", "sample_index", "time_s"],
        columns="signal",
        values="value",
    )
    .reset_index()
)

cycle = wide[
    (wide["cycle_number"] == CYCLE_NUMBER)
    & (wide["cycle_type"] == "discharge")
].copy()

cycle = cycle.sort_values("time_s")

if cycle.empty:
    raise ValueError(f"Discharge cycle {CYCLE_NUMBER} was not found.")

# NASA discharge current is negative.
cycle["discharge_current_a"] = (
    -cycle["Current_measured"]
).clip(lower=0)

# Fit only the active-discharge interval, excluding post-discharge relaxation.
active = cycle[
    cycle["discharge_current_a"] >= MIN_CURRENT_A
].copy()

active = active.sort_values("time_s").reset_index(drop=True)

if len(active) < 10:
    raise ValueError("Too few active-discharge samples to fit the models.")

t = (
    active["time_s"].to_numpy(dtype=float)
    - active["time_s"].iloc[0]
)
temperature = active["Temperature_measured"].to_numpy(dtype=float)

# Candidate 1: first-order exponential heating.
def exponential_model(time, initial_temp, amplitude, tau):
    return initial_temp + amplitude * (
        1.0 - np.exp(-time / tau)
    )

initial_guess = [
    float(temperature[0]),
    float(temperature[-1] - temperature[0]),
    max(float(t[-1]) / 2, 1.0),
]

parameters, _ = curve_fit(
    exponential_model,
    t,
    temperature,
    p0=initial_guess,
    bounds=(
        [0.0, -50.0, 1.0],
        [100.0, 50.0, 100000.0],
    ),
    maxfev=50000,
)

predicted_exp = exponential_model(t, *parameters)

# Candidate 2: linear heating baseline.
linear_coefficients = np.polyfit(t, temperature, 1)
predicted_linear = np.polyval(linear_coefficients, t)

def metrics(measured, predicted):
    error = measured - predicted
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    return rmse, mae

exp_rmse, exp_mae = metrics(temperature, predicted_exp)
lin_rmse, lin_mae = metrics(temperature, predicted_linear)

initial_temp, amplitude, tau = parameters

# Save point-by-point predictions.
predictions = active[
    [
        "cycle_number",
        "sample_index",
        "time_s",
        "Temperature_measured",
        "discharge_current_a",
        "Voltage_measured",
    ]
].copy()

predictions["elapsed_time_s"] = t
predictions["exponential_prediction_c"] = predicted_exp
predictions["linear_prediction_c"] = predicted_linear
predictions["exponential_residual_c"] = (
    temperature - predicted_exp
)
predictions["linear_residual_c"] = (
    temperature - predicted_linear
)

predictions.to_csv(
    OUTPUT_DIR / f"B0005_cycle_{CYCLE_NUMBER}_thermal_fit.csv",
    index=False,
)

# Save a concise report.
report = f"""
NASA B0005 FIRST-ORDER THERMAL RESPONSE FIT
===========================================

Cycle: {CYCLE_NUMBER}
Active-discharge samples: {len(active)}
Elapsed time: {t[-1]:.3f} s

Measured initial temperature: {temperature[0]:.6f} °C
Measured final temperature: {temperature[-1]:.6f} °C
Measured temperature rise: {temperature[-1] - temperature[0]:.6f} °C

First-order exponential fit:
Fitted initial temperature: {initial_temp:.6f} °C
Fitted rise amplitude: {amplitude:.6f} °C
Fitted time constant: {tau:.6f} s
Fitted time constant: {tau / 60:.6f} min
RMSE: {exp_rmse:.6f} °C
MAE: {exp_mae:.6f} °C

Linear baseline:
Slope: {linear_coefficients[0]:.8f} °C/s
RMSE: {lin_rmse:.6f} °C
MAE: {lin_mae:.6f} °C

Interpretation:
These are empirical fits to one measured discharge cycle.
The fitted time constant is not automatically a physical thermal
time constant. Ambient temperature, cooling conditions, heat
generation and sensor location have not been independently identified.
"""

report_path = (
    OUTPUT_DIR / f"B0005_cycle_{CYCLE_NUMBER}_thermal_fit_report.txt"
)
report_path.write_text(report.strip() + "\n", encoding="utf-8")

# Plot measured temperature and both fitted curves.
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, temperature, label="Measured temperature")
axes[0].plot(t, predicted_exp, label="First-order exponential fit")
axes[0].plot(t, predicted_linear, label="Linear baseline")
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title(f"NASA B0005 — Cycle {CYCLE_NUMBER} thermal response")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, temperature - predicted_exp, label="Exponential residual")
axes[1].plot(t, temperature - predicted_linear, label="Linear residual")
axes[1].axhline(0, linewidth=1)
axes[1].set_xlabel("Elapsed active-discharge time (s)")
axes[1].set_ylabel("Residual (°C)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plot_path = OUTPUT_DIR / f"B0005_cycle_{CYCLE_NUMBER}_thermal_fit.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(report.strip())
print(f"\nPredictions saved: {OUTPUT_DIR}")
print(f"Report saved: {report_path}")
print(f"Plot saved: {plot_path}")
