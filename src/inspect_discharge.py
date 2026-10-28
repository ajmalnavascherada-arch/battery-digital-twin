
"""Inspect original NASA MATLAB discharge signals without modifying them."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

FILE = Path("data/raw/nasa_original/B0005.mat")

mat = loadmat(FILE, struct_as_record=False, squeeze_me=True)
root_name = next(k for k in mat if not k.startswith("__"))
root = mat[root_name]
cycles = np.asarray(root.cycle, dtype=object).reshape(-1)


print("\nFirst 30 cycle types:")
for idx, cycle in enumerate(cycles[:30]):
    print(idx, str(cycle.type))

discharge_indices = [
    idx for idx, cycle in enumerate(cycles)
    if str(cycle.type).lower() == "discharge"
]

print("\nTotal cycles:", len(cycles))
print("Number of discharge cycles:", len(discharge_indices))
print("First 10 discharge indices:", discharge_indices[:10])

for idx in discharge_indices[:2]:
    cycle = cycles[idx]
    data = cycle.data

    print("\n" + "=" * 65)
    print(f"Cycle index: {idx} | Type: {cycle.type}")

    time = np.asarray(data.Time, dtype=float).reshape(-1)
    print("Time samples:", len(time))

    for name in [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
    ]:
        if hasattr(data, name):
            values = np.asarray(
                getattr(data, name), dtype=float
            ).reshape(-1)
            print(f"{name}: {len(values)} samples")

    print("Available fields:", data._fieldnames)

    cycle = cycles[idx]
    data = cycle.data

    print("\n" + "=" * 65)
    print(f"Cycle index: {idx} | Type: {cycle.type}")
    print(f"Ambient temperature: {cycle.ambient_temperature}")
    print("Cycle-level time:", np.asarray(cycle.time).shape)

    time = np.asarray(data.Time, dtype=float).reshape(-1)
    print(f"\ndata.Time: shape={time.shape}")
    print("First 5 timestamps:", time[:5])
    print("Last 5 timestamps:", time[-5:])
    print("All timestamps finite:", np.isfinite(time).all())
    print("Timestamps monotonically increasing:",
          bool(np.all(np.diff(time) > 0)))

    for name in [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "Current_load",
        "Voltage_load",
        "Current_charge",
        "Voltage_charge",
    ]:
        if not hasattr(data, name):
            print(f"\n{name}: not present in this cycle")
            continue

        values = np.asarray(
            getattr(data, name), dtype=float
        ).reshape(-1)

        print(f"\n{name}:")
        print("  Shape:", values.shape)
        print("  Finite samples:", int(np.isfinite(values).sum()))
        print("  First 5:", values[:5])
        print("  Last 5:", values[-5:])


