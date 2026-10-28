
"""Check sample availability for fixed-time early-discharge windows."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
MAT_PATH = ROOT / "data" / "raw" / "nasa_original" / "B0005.mat"
OUT_DIR = ROOT / "results_capacity_comparison" / "b0005_feature_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS_S = (300, 600, 900)
CELL_ID = "B0005"


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
    current = get_array(data, "Current_measured")
    temperature = get_array(data, "Temperature_measured")

    if not (
        len(time) == len(voltage) == len(current) == len(temperature)
    ):
        raise ValueError(f"Signal mismatch at record {record_number}")

    if len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError(f"Invalid timestamps at record {record_number}")

    discharge_index += 1
    elapsed = time - time[0]

    row = {
        "discharge_index": discharge_index,
        "recorded_cycle_number": record_number,
        "recorded_duration_s": float(elapsed[-1]),
        "capacity_Ah": float(data.get("Capacity", np.nan)),
    }

    for window_s in WINDOWS_S:
        mask = elapsed <= window_s
        prefix = f"first_{window_s}s_"

        row[prefix + "available"] = bool(elapsed[-1] >= window_s)
        row[prefix + "sample_count"] = int(mask.sum())

        if mask.sum() >= 2:
            row[prefix + "last_sample_time_s"] = float(elapsed[mask][-1])
            row[prefix + "voltage_drop_V"] = float(
                voltage[0] - voltage[mask][-1]
            )
            row[prefix + "temperature_rise_C"] = float(
                temperature[mask][-1] - temperature[0]
            )
        else:
            row[prefix + "last_sample_time_s"] = np.nan
            row[prefix + "voltage_drop_V"] = np.nan
            row[prefix + "temperature_rise_C"] = np.nan

    rows.append(row)

audit = pd.DataFrame(rows)
output = OUT_DIR / "B0005_fixed_time_window_audit.csv"
audit.to_csv(output, index=False)

print("Discharge records:", len(audit))
print("Unique discharge indices:", audit["discharge_index"].nunique())

for window_s in WINDOWS_S:
    prefix = f"first_{window_s}s_"
    available = int(audit[prefix + "available"].sum())
    print(
        f"{window_s}s window: available for {available}/{len(audit)} records; "
        f"minimum samples={audit[prefix + 'sample_count'].min()}, "
        f"median samples={audit[prefix + 'sample_count'].median():.0f}"
    )

print("Saved:", output)
