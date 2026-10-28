
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw" / "nasa_original" / "B0005.mat"
OUT = ROOT / "results_nasa_original" / "thermal_phase_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CYCLE_NUMBER = 2
ACTIVE_CURRENT_A = 0.5
NEAR_ZERO_CURRENT_A = 0.05

from scipy.io import loadmat


def get_cycle_data(mat_path, cycle_number):
    mat = loadmat(mat_path, simplify_cells=True)
    battery = mat["B0005"]
    cycles = np.asarray(battery["cycle"], dtype=object).reshape(-1)

    matches = [
        cycle for cycle in cycles
        if isinstance(cycle, dict)
        and cycle.get("type") == "discharge"
    ]

    # Cycle number refers to the recorded discharge cycle number,
    # not its position in the complete charge/discharge/impedance array.
    # Select by matching the source sequence where possible.
    target = None
    discharge_count = 0

    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue
        if cycle.get("type") == "discharge":
            discharge_count += 1
            if discharge_count == cycle_number:
                target = cycle
                break

    if target is None:
        raise ValueError(
            f"Could not locate discharge record {cycle_number}."
        )

    data = target["data"]

    frame = pd.DataFrame({
        "time_s": np.asarray(data["Time"], dtype=float).reshape(-1),
        "temperature_c": np.asarray(
            data["Temperature_measured"], dtype=float
        ).reshape(-1),
        "current_measured_a": np.asarray(
            data["Current_measured"], dtype=float
        ).reshape(-1),
        "voltage_v": np.asarray(
            data["Voltage_measured"], dtype=float
        ).reshape(-1),
    })

    frame["discharge_current_a"] = (
        -frame["current_measured_a"]
    ).clip(lower=0)

    frame["phase"] = np.select(
        [
            frame["discharge_current_a"] >= ACTIVE_CURRENT_A,
            frame["discharge_current_a"] < NEAR_ZERO_CURRENT_A,
        ],
        [
            "active_discharge",
            "near_zero_current",
        ],
        default="transition",
    )

    frame["elapsed_time_s"] = frame["time_s"] - frame["time_s"].iloc[0]
    return frame


def phase_summary(frame):
    rows = []

    for phase, group in frame.groupby("phase", sort=False):
        group = group.sort_values("time_s")
        duration = (
            float(group["time_s"].iloc[-1] - group["time_s"].iloc[0])
            if len(group) > 1 else 0.0
        )

        rows.append({
            "phase": phase,
            "samples": len(group),
            "start_time_s": float(group["time_s"].iloc[0]),
            "end_time_s": float(group["time_s"].iloc[-1]),
            "duration_s": duration,
            "temperature_start_c": float(group["temperature_c"].iloc[0]),
            "temperature_end_c": float(group["temperature_c"].iloc[-1]),
            "temperature_min_c": float(group["temperature_c"].min()),
            "temperature_max_c": float(group["temperature_c"].max()),
            "mean_current_a": float(group["discharge_current_a"].mean()),
        })

    return pd.DataFrame(rows)


frame = get_cycle_data(INPUT, CYCLE_NUMBER)
summary = phase_summary(frame)

frame.to_csv(
    OUT / f"B0005_discharge_record_{CYCLE_NUMBER}_phases.csv",
    index=False,
)
summary.to_csv(
    OUT / f"B0005_discharge_record_{CYCLE_NUMBER}_phase_summary.csv",
    index=False,
)

print(f"=== NASA B0005 DISCHARGE RECORD {CYCLE_NUMBER} ===")
print(f"Total samples: {len(frame)}")
print(f"Ambient-temperature metadata: 24.0 °C")
print("\nPhase summary:")
print(summary.round(4).to_string(index=False))

# Estimate the temperature slope over time using finite differences.
frame["temperature_slope_c_per_min"] = (
    frame["temperature_c"].diff()
    / frame["time_s"].diff()
    * 60
)

print("\nTemperature slope statistics (°C/min):")
print(
    frame.groupby("phase")["temperature_slope_c_per_min"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(5)
    .to_string()
)

# Plot temperature, discharge current and voltage on separate axes.
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

axes[0].plot(
    frame["elapsed_time_s"],
    frame["temperature_c"],
    marker=".",
    markersize=3,
)
axes[0].set_ylabel("Temperature (°C)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    frame["elapsed_time_s"],
    frame["discharge_current_a"],
)
axes[1].axhline(ACTIVE_CURRENT_A, linestyle="--")
axes[1].set_ylabel("Discharge current (A)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(
    frame["elapsed_time_s"],
    frame["voltage_v"],
)
axes[2].set_ylabel("Voltage (V)")
axes[2].set_xlabel("Elapsed time (s)")
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    f"NASA B0005 — Temperature and electrical response, "
    f"discharge record {CYCLE_NUMBER}"
)
fig.tight_layout()

plot_path = OUT / f"B0005_discharge_record_{CYCLE_NUMBER}_phases.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved outputs to: {OUT}")
print(f"Plot: {plot_path}")
