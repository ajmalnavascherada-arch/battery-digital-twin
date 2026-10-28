
"""Extract early-discharge features from verified NASA B0005 data."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
MAT_PATH = ROOT / "data" / "raw" / "nasa_original" / "B0005.mat"
OUT_DIR = ROOT / "results_capacity_comparison" / "b0005_feature_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_ID = "B0005"
WINDOWS = (10, 20, 30)
ACTIVE_CURRENT_A = 0.5


def get_array(data, key):
    return np.asarray(data[key], dtype=float).reshape(-1)


if not MAT_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {MAT_PATH}")

mat = loadmat(str(MAT_PATH), simplify_cells=True)
cycles = np.asarray(mat[CELL_ID]["cycle"], dtype=object).reshape(-1)

records = []
discharge_index = 0

for recorded_cycle, cycle in enumerate(cycles, start=1):
    if not isinstance(cycle, dict) or cycle.get("type") != "discharge":
        continue

    data = cycle["data"]
    time = get_array(data, "Time")
    voltage = get_array(data, "Voltage_measured")
    current = get_array(data, "Current_measured")
    temperature = get_array(data, "Temperature_measured")
    capacity = float(data.get("Capacity", np.nan))

    lengths = {
        len(time), len(voltage), len(current), len(temperature)
    }
    if len(lengths) != 1:
        raise ValueError(
            f"Signal mismatch at record {recorded_cycle}: "
            f"time={len(time)}, voltage={len(voltage)}, "
            f"current={len(current)}, temperature={len(temperature)}"
        )

    if len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError(f"Invalid timestamps at record {recorded_cycle}")

    discharge_index += 1
    duration = float(time[-1] - time[0])
    discharge_current = -current

    row = {
        "cell_id": CELL_ID,
        "recorded_cycle_number": recorded_cycle,
        "discharge_index": discharge_index,
        "capacity_Ah": capacity,
        "discharge_duration_s": duration,
    }

    for pct in WINDOWS:
        cutoff_time = time[0] + duration * pct / 100.0
        mask = time <= cutoff_time

        t = time[mask]
        v = voltage[mask]
        i = discharge_current[mask]
        temp = temperature[mask]

        if len(t) < 2:
            raise ValueError(
                f"Too few samples in first {pct}% window "
                f"at record {recorded_cycle}"
            )

        prefix = f"first_{pct}pct_"
        row[prefix + "window_duration_s"] = float(t[-1] - t[0])
        row[prefix + "sample_count"] = int(len(t))

        row[prefix + "initial_voltage_V"] = float(v[0])
        row[prefix + "final_voltage_V"] = float(v[-1])
        row[prefix + "voltage_drop_V"] = float(v[0] - v[-1])
        row[prefix + "mean_voltage_V"] = float(np.mean(v))
        row[prefix + "voltage_slope_V_per_s"] = float(
            np.polyfit(t - t[0], v, 1)[0]
        )

        row[prefix + "initial_temperature_C"] = float(temp[0])
        row[prefix + "final_temperature_C"] = float(temp[-1])
        row[prefix + "temperature_rise_C"] = float(temp[-1] - temp[0])
        row[prefix + "mean_temperature_C"] = float(np.mean(temp))
        row[prefix + "maximum_temperature_C"] = float(np.max(temp))
        row[prefix + "temperature_slope_C_per_s"] = float(
            np.polyfit(t - t[0], temp, 1)[0]
        )

        active = i >= ACTIVE_CURRENT_A
        row[prefix + "active_sample_fraction"] = float(np.mean(active))
        row[prefix + "mean_discharge_current_A"] = float(np.mean(i))
        row[prefix + "mean_active_current_A"] = (
            float(np.mean(i[active])) if active.any() else np.nan
        )

    # Append once per discharge, after all three windows are calculated.
    records.append(row)

features = pd.DataFrame(records)

if len(features) != 168:
    raise RuntimeError(
        f"Expected 168 discharge records, got {len(features)}"
    )

if features["discharge_index"].nunique() != len(features):
    raise RuntimeError("Duplicate discharge indices detected.")

reference_capacity = float(features["capacity_Ah"].iloc[0])
features["relative_capacity_retention_pct"] = (
    100.0 * features["capacity_Ah"] / reference_capacity
)

output = OUT_DIR / "B0005_early_discharge_features.csv"
features.to_csv(output, index=False)

print("Early-discharge feature extraction completed.")
print("Cell:", CELL_ID)
print("Discharge records:", len(features))
print("Unique discharge indices:", features["discharge_index"].nunique())
print("Duplicate indices:", features["discharge_index"].duplicated().sum())
print("Windows (%):", WINDOWS)
print("Output columns:", len(features.columns))
print("Output:", output)

print("\nFeature-window sample counts:")
for pct in WINDOWS:
    col = f"first_{pct}pct_sample_count"
    print(
        f"First {pct}%: min={features[col].min()}, "
        f"median={features[col].median():.0f}, "
        f"max={features[col].max()}"
    )

summary_columns = [
    "first_10pct_voltage_drop_V",
    "first_10pct_temperature_rise_C",
    "first_20pct_voltage_drop_V",
    "first_20pct_temperature_rise_C",
    "first_30pct_voltage_drop_V",
    "first_30pct_temperature_rise_C",
]
print("\nEarly feature summary:")
print(features[summary_columns].describe().round(4).to_string())
