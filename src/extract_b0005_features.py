
"""Extract aligned cycle-level features from verified NASA B0005."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
MAT_PATH = ROOT / "data" / "raw" / "nasa_original" / "B0005.mat"
OUT_DIR = ROOT / "results_capacity_comparison" / "b0005_feature_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_CURRENT_A = 0.5
CELL_NAME = "B0005"


def array(data, key):
    return np.asarray(data[key], dtype=float).reshape(-1)


if not MAT_PATH.is_file():
    raise FileNotFoundError(f"Raw dataset not found: {MAT_PATH}")

mat = loadmat(str(MAT_PATH), simplify_cells=True)
cycles = np.asarray(mat[CELL_NAME]["cycle"], dtype=object).reshape(-1)

records = []
discharge_index = 0

for recorded_cycle, cycle in enumerate(cycles, start=1):
    if not isinstance(cycle, dict) or cycle.get("type") != "discharge":
        continue

    data = cycle["data"]
    time = array(data, "Time")
    voltage = array(data, "Voltage_measured")
    current = array(data, "Current_measured")
    temperature = array(data, "Temperature_measured")

    lengths = {len(time), len(voltage), len(current), len(temperature)}
    if len(lengths) != 1:
        raise ValueError(
            f"Unexpected signal mismatch in B0005 record {recorded_cycle}: "
            f"Time={len(time)}, Voltage={len(voltage)}, "
            f"Current={len(current)}, Temperature={len(temperature)}"
        )

    if len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError(f"Invalid time sequence in record {recorded_cycle}")

    discharge_index += 1

    # NASA Current_measured is negative during discharge.
    discharge_current = -current
    active = discharge_current >= ACTIVE_CURRENT_A

    if not active.any():
        active_start = 0
        active_end = len(time) - 1
        active_count = 0
    else:
        active_indices = np.flatnonzero(active)
        active_start = int(active_indices[0])
        active_end = int(active_indices[-1])
        active_count = int(active.sum())

    records.append({
        "cell_id": CELL_NAME,
        "recorded_cycle_number": recorded_cycle,
        "discharge_index": discharge_index,
        "ambient_temperature_C": float(cycle.get("ambient_temperature", np.nan)),
        "capacity_Ah": float(cycle.get("data", {}).get("Capacity", np.nan)),
        "discharge_duration_s": float(time[-1] - time[0]),
        "initial_voltage_V": float(voltage[0]),
        "final_voltage_V": float(voltage[-1]),
        "voltage_drop_V": float(voltage[0] - voltage[-1]),
        "initial_temperature_C": float(temperature[0]),
        "final_temperature_C": float(temperature[-1]),
        "temperature_rise_C": float(temperature[-1] - temperature[0]),
        "mean_temperature_C": float(np.mean(temperature)),
        "maximum_temperature_C": float(np.max(temperature)),
        "active_discharge_samples": active_count,
        "active_discharge_start_s": float(time[active_start]),
        "active_discharge_end_s": float(time[active_end]),
        "signal_sample_count": len(time),
        "signals_aligned": True,
    })

features = pd.DataFrame(records)

if features.empty:
    raise RuntimeError("No discharge records were extracted.")

reference_capacity = float(features["capacity_Ah"].iloc[0])
features["relative_capacity_retention_pct"] = (
    100.0 * features["capacity_Ah"] / reference_capacity
)

output = OUT_DIR / "B0005_cycle_features.csv"
features.to_csv(output, index=False)

print("Feature extraction completed.")
print("Cell:", CELL_NAME)
print("Discharge records:", len(features))
print("All signals aligned:", bool(features["signals_aligned"].all()))
print("First capacity (Ah):", round(reference_capacity, 6))
print("Feature columns:", len(features.columns))
print("Output:", output)
print("\nFeature summary:")
print(
    features[
        [
            "capacity_Ah",
            "discharge_duration_s",
            "voltage_drop_V",
            "temperature_rise_C",
            "maximum_temperature_C",
        ]
    ].describe().round(4).to_string()
)
