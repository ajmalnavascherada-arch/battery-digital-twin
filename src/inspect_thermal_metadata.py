
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
MAT_FILE = ROOT / "data" / "raw" / "nasa_original" / "B0005.mat"

mat = loadmat(MAT_FILE, simplify_cells=True)
battery = mat["B0005"]
cycles = np.asarray(battery["cycle"], dtype=object).reshape(-1)

print("=== ORIGINAL NASA B0005: CYCLE 2 ===")

# The MATLAB structure's second record is discharge cycle 2.
cycle = cycles[1]

if not isinstance(cycle, dict):
    raise TypeError(
        f"Unexpected cycle structure: {type(cycle)}. "
        "Inspect the structure before proceeding."
    )

print("Cycle type:", cycle.get("type"))
print("Ambient temperature field:", cycle.get("ambient_temperature"))

data = cycle.get("data")
if not isinstance(data, dict):
    raise TypeError("Cycle data is not a dictionary.")

print("\nAvailable data fields:")
for name, value in data.items():
    arr = np.asarray(value)
    if arr.size and np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)]
        if finite.size:
            print(
                f"{name}: shape={arr.shape}, "
                f"min={finite.min():.6g}, max={finite.max():.6g}"
            )
        else:
            print(f"{name}: shape={arr.shape}, no finite values")
    else:
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")

print("\n=== THERMAL MODEL INPUT CHECK ===")
for key in [
    "Time",
    "Temperature_measured",
    "Current_measured",
    "Voltage_measured",
    "Capacity",
]:
    print(f"{key}: {'available' if key in data else 'not available'}")

print(
    "\nNote: this is a metadata audit. It does not assume that "
    "terminal electrical power equals heat generation."
)
