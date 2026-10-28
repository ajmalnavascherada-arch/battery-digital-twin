
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = (
    ROOT / "results_nasa_original"
    / "thermal_phase_analysis"
    / "B0005_discharge_record_2_phases.csv"
)
OUT = ROOT / "results_nasa_original" / "post_discharge_cooling"
OUT.mkdir(parents=True, exist_ok=True)

AMBIENT_C = 24.0
ACTIVE_CURRENT_A = 0.5

df = pd.read_csv(INPUT).sort_values("time_s").reset_index(drop=True)

active = df["discharge_current_a"] >= ACTIVE_CURRENT_A
if not active.any():
    raise ValueError("No active-discharge samples found.")

last_active_position = np.flatnonzero(active.to_numpy())[-1]
cooling = df.iloc[last_active_position + 1:].copy()

if len(cooling) < 5:
    raise ValueError(
        f"Only {len(cooling)} post-discharge samples found; "
        "not enough for a useful cooling analysis."
    )

t0 = float(df.loc[last_active_position, "time_s"])
T0 = float(df.loc[last_active_position, "temperature_c"])

cooling["elapsed_after_discharge_s"] = cooling["time_s"] - t0
cooling["temperature_rise_above_ambient_c"] = (
    cooling["temperature_c"] - AMBIENT_C
)

# Fit T(t) = T_ambient + (T0 - T_ambient) * exp(-t/tau)
# with ambient and the starting temperature fixed.
t = cooling["elapsed_after_discharge_s"].to_numpy(dtype=float)
temperature = cooling["temperature_c"].to_numpy(dtype=float)

def predict_temperature(t_s, tau_s):
    return AMBIENT_C + (T0 - AMBIENT_C) * np.exp(-t_s / tau_s)

# Search a broad range of positive time constants without
# requiring another optimization library.
tau_grid = np.geomspace(1.0, 100000.0, 30000)

errors = np.array([
    np.mean((temperature - predict_temperature(t, tau)) ** 2)
    for tau in tau_grid
])
best_index = int(np.argmin(errors))
tau_best = float(tau_grid[best_index])

predicted = predict_temperature(t, tau_best)
residual = temperature - predicted

rmse = float(np.sqrt(np.mean(residual ** 2)))
mae = float(np.mean(np.abs(residual)))

cooling.to_csv(
    OUT / "B0005_cycle_2_post_discharge_cooling.csv",
    index=False,
)

metrics = pd.DataFrame([{
    "ambient_temperature_c": AMBIENT_C,
    "last_active_time_s": t0,
    "last_active_temperature_c": T0,
    "post_discharge_samples": len(cooling),
    "post_discharge_duration_s": float(t[-1] - t[0]),
    "first_post_discharge_temperature_c": float(temperature[0]),
    "last_temperature_c": float(temperature[-1]),
    "temperature_change_c": float(temperature[-1] - temperature[0]),
    "best_fit_tau_s": tau_best,
    "best_fit_tau_min": tau_best / 60.0,
    "rmse_c": rmse,
    "mae_c": mae,
}])

metrics.to_csv(
    OUT / "B0005_cycle_2_cooling_fit_metrics.csv",
    index=False,
)

print("=== POST-DISCHARGE COOLING ANALYSIS ===")
print(f"Last active time: {t0:.3f} s")
print(f"Last active temperature: {T0:.4f} °C")
print(f"Post-discharge samples: {len(cooling)}")
print(f"Post-discharge duration: {t[-1] - t[0]:.3f} s")
print(f"First post-discharge temperature: {temperature[0]:.4f} °C")
print(f"Last temperature: {temperature[-1]:.4f} °C")
print(f"Fitted time constant: {tau_best:.2f} s ({tau_best / 60:.2f} min)")
print(f"Cooling-fit RMSE: {rmse:.4f} °C")
print(f"Cooling-fit MAE: {mae:.4f} °C")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t, temperature, "o", label="Measured temperature")
ax.plot(t, predicted, "-", label="Single-exponential fit")
ax.axhline(AMBIENT_C, linestyle="--", label="Ambient metadata: 24 °C")
ax.set_xlabel("Time after last active-discharge sample (s)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("NASA B0005 Cycle 2 — Post-discharge cooling")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

plot_path = OUT / "B0005_cycle_2_post_discharge_cooling.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved results to: {OUT}")
