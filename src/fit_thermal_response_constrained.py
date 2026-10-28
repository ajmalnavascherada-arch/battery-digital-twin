
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed_nasa_original" / "B0005_signals_long.csv"
OUT = ROOT / "results_nasa_original" / "thermal_model"
OUT.mkdir(parents=True, exist_ok=True)

CYCLE = 2
MIN_CURRENT_A = 0.5

df = pd.read_csv(INPUT)
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
    (wide["cycle_number"] == CYCLE)
    & (wide["cycle_type"] == "discharge")
].copy()

cycle["discharge_current_a"] = (
    -cycle["Current_measured"]
).clip(lower=0)

active = (
    cycle[cycle["discharge_current_a"] >= MIN_CURRENT_A]
    .sort_values("time_s")
    .reset_index(drop=True)
)

if len(active) < 10:
    raise ValueError("Insufficient active-discharge samples.")

t = active["time_s"].to_numpy(float)
t = t - t[0]
temp = active["Temperature_measured"].to_numpy(float)

# Keep the initial temperature equal to the first measured value.
T0 = float(temp[0])

def constrained_exponential(time, amplitude, tau):
    return T0 + amplitude * (1 - np.exp(-time / tau))

params, _ = curve_fit(
    constrained_exponential,
    t,
    temp,
    p0=[float(temp[-1] - T0), max(float(t[-1]) / 2, 1)],
    bounds=([-50, 1], [50, 100000]),
    maxfev=50000,
)

amplitude, tau = params
pred = constrained_exponential(t, amplitude, tau)

# Linear baseline, also anchored at the first measured temperature.
slope = np.sum(t * (temp - T0)) / np.sum(t**2)
linear_pred = T0 + slope * t

def report_metrics(y, yhat):
    err = y - yhat
    return (
        float(np.sqrt(np.mean(err**2))),
        float(np.mean(np.abs(err))),
    )

exp_rmse, exp_mae = report_metrics(temp, pred)
lin_rmse, lin_mae = report_metrics(temp, linear_pred)

result = active[
    ["cycle_number", "sample_index", "time_s",
     "Temperature_measured", "discharge_current_a"]
].copy()

result["elapsed_time_s"] = t
result["constrained_exponential_c"] = pred
result["anchored_linear_c"] = linear_pred
result["exponential_residual_c"] = temp - pred
result["linear_residual_c"] = temp - linear_pred

csv_path = OUT / f"B0005_cycle_{CYCLE}_thermal_constrained.csv"
result.to_csv(csv_path, index=False)

report = f"""
CONSTRAINED THERMAL RESPONSE — NASA B0005
Cycle: {CYCLE}
Samples: {len(active)}
Initial measured temperature: {T0:.6f} °C
Final measured temperature: {temp[-1]:.6f} °C
Measured rise: {temp[-1] - T0:.6f} °C

Constrained exponential:
Initial temperature fixed: {T0:.6f} °C
Rise amplitude: {amplitude:.6f} °C
Time constant: {tau:.6f} s ({tau / 60:.3f} min)
RMSE: {exp_rmse:.6f} °C
MAE: {exp_mae:.6f} °C

Anchored linear baseline:
Slope: {slope:.9f} °C/s
RMSE: {lin_rmse:.6f} °C
MAE: {lin_mae:.6f} °C

This is a constrained empirical fit to one discharge cycle,
not a validated physical thermal model.
"""

report_path = OUT / f"B0005_cycle_{CYCLE}_thermal_constrained_report.txt"
report_path.write_text(report.strip() + "\n", encoding="utf-8")

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axes[0].plot(t, temp, label="Measured")
axes[0].plot(t, pred, label="Constrained exponential")
axes[0].plot(t, linear_pred, label="Anchored linear")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, temp - pred, label="Exponential residual")
axes[1].plot(t, temp - linear_pred, label="Linear residual")
axes[1].axhline(0, linewidth=1)
axes[1].set_xlabel("Elapsed active-discharge time (s)")
axes[1].set_ylabel("Residual (°C)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plot_path = OUT / f"B0005_cycle_{CYCLE}_thermal_constrained.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(report.strip())
print(f"\nCSV: {csv_path}")
print(f"Report: {report_path}")
print(f"Plot: {plot_path}")
