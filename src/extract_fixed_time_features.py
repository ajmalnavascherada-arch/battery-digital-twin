
"""Extract fixed-elapsed-time features from aligned NASA B0005 signals."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
MAT_PATH = ROOT / "data" / "raw" / "nasa_original" / "B0005.mat"
OUT_DIR = ROOT / "results_capacity_comparison" / "b0005_feature_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_ID = "B0005"
WINDOWS_S = (300, 600, 900)
ACTIVE_CURRENT_A = 0.5


def get_array(data, key):
    return np.asarray(data[key], dtype=float).reshape(-1)


mat = loadmat(str(MAT_PATH), simplify_cells=True)
cycles = np.asarray(mat[CELL_ID]["cycle"], dtype=object).reshape(-1)

rows = []
discharge_index = 0

for record_number, cycle in enumerate(cycles, start=1):
    if not isinstance(cycle, dict) or cycle.get("type") != "discharge":
        continue

    data = cycle["data"]
    time = get_array(data, "Time")
    voltage = get_array(data, "Voltage_measured")
    current = -get_array(data, "Current_measured")
    temperature = get_array(data, "Temperature_measured")
    capacity = float(data.get("Capacity", np.nan))

    if not (len(time) == len(voltage) == len(current) == len(temperature)):
        raise ValueError(f"Signal mismatch in raw record {record_number}")
    if len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError(f"Invalid time sequence in raw record {record_number}")

    discharge_index += 1
    elapsed = time - time[0]
    row = {
        "cell_id": CELL_ID,
        "recorded_cycle_number": record_number,
        "discharge_index": discharge_index,
        "capacity_Ah": capacity,
    }

    for window_s in WINDOWS_S:
        mask = elapsed <= window_s
        if not mask.any() or elapsed[-1] < window_s:
            raise ValueError(
                f"Record {record_number} lacks a complete {window_s}s window"
            )

        t = elapsed[mask]
        v = voltage[mask]
        i = current[mask]
        temp = temperature[mask]
        prefix = f"first_{window_s}s_"

        row[prefix + "sample_count"] = len(t)
        row[prefix + "last_sample_time_s"] = float(t[-1])
        row[prefix + "voltage_drop_V"] = float(v[0] - v[-1])
        row[prefix + "mean_voltage_V"] = float(np.mean(v))
        row[prefix + "final_voltage_V"] = float(v[-1])
        row[prefix + "voltage_slope_V_per_s"] = float(
            np.polyfit(t, v, 1)[0]
        )
        row[prefix + "temperature_rise_C"] = float(temp[-1] - temp[0])
        row[prefix + "mean_temperature_C"] = float(np.mean(temp))
        row[prefix + "maximum_temperature_C"] = float(np.max(temp))
        row[prefix + "temperature_slope_C_per_s"] = float(
            np.polyfit(t, temp, 1)[0]
        )
        active = i >= ACTIVE_CURRENT_A
        row[prefix + "active_sample_fraction"] = float(np.mean(active))
        row[prefix + "mean_discharge_current_A"] = float(np.mean(i))
        row[prefix + "mean_active_current_A"] = (
            float(np.mean(i[active])) if active.any() else np.nan
        )

    rows.append(row)

features = pd.DataFrame(rows)

if len(features) != 168 or features["discharge_index"].nunique() != 168:
    raise RuntimeError("Expected 168 unique B0005 discharge records.")

reference_capacity = float(features["capacity_Ah"].iloc[0])
features["relative_capacity_retention_pct"] = (
    100 * features["capacity_Ah"] / reference_capacity
)

output = OUT_DIR / "B0005_fixed_time_features.csv"
features.to_csv(output, index=False)

print("Fixed-time feature extraction completed.")
print("Discharge records:", len(features))
print("Unique indices:", features["discharge_index"].nunique())
print("Output columns:", len(features.columns))
print("Saved:", output)
print("\nFeature preview:")
print(
    features[
        [
            "first_300s_voltage_drop_V",
            "first_300s_temperature_rise_C",
            "first_600s_voltage_drop_V",
            "first_600s_temperature_rise_C",
            "first_900s_voltage_drop_V",
            "first_900s_temperature_rise_C",
        ]
    ].describe().round(4).to_string()
)
