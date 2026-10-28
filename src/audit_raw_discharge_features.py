
"""Audit raw discharge signal lengths and basic signal statistics."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "B0005": ROOT / "data" / "raw" / "nasa_original" / "B0005.mat",
    "B0006": ROOT / "data" / "raw" / "B0006.mat",
    "B0007": ROOT / "data" / "raw" / "B0007.mat",
}

SIGNALS = [
    "Time",
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
]

ACTIVE_CURRENT_A = 0.5


def as_array(value):
    return np.asarray(value, dtype=float).reshape(-1)


for cell_id, path in FILES.items():
    if not path.is_file():
        print(f"\n{cell_id}: FILE NOT FOUND: {path}")
        continue

    mat = loadmat(str(path), simplify_cells=True)
    cycles = np.asarray(mat[cell_id]["cycle"], dtype=object).reshape(-1)

    discharge_records = [
        (i + 1, c)
        for i, c in enumerate(cycles)
        if isinstance(c, dict) and c.get("type") == "discharge"
    ]

    selected_positions = sorted({
        0, len(discharge_records) // 2, len(discharge_records) - 1
    })

    print(f"\n{'=' * 72}")
    print(f"{cell_id}: {len(discharge_records)} discharge records")

    for pos in selected_positions:
        record_number, cycle = discharge_records[pos]
        data = cycle["data"]

        print(f"\nRaw record {record_number}; discharge position {pos + 1}")

        arrays = {}
        for name in SIGNALS:
            if name in data:
                arrays[name] = as_array(data[name])
                print(f"{name}: {len(arrays[name])} samples")
            else:
                print(f"{name}: MISSING")

        if not all(name in arrays for name in SIGNALS):
            continue

        lengths = [len(arrays[name]) for name in SIGNALS]
        print("All four lengths equal:", len(set(lengths)) == 1)

        time = arrays["Time"]
        voltage = arrays["Voltage_measured"]
        current = arrays["Current_measured"]
        temperature = arrays["Temperature_measured"]

        if len(time) > 1:
            print(f"Recorded time span: {time[-1] - time[0]:.3f} s")
            print("Time strictly increasing:", bool(np.all(np.diff(time) > 0)))

        # Statistics are descriptive only; they do not assert sample alignment.
        active_count = int(np.sum(np.abs(current) >= ACTIVE_CURRENT_A))
        print(
            f"Current samples with |I| >= {ACTIVE_CURRENT_A} A: "
            f"{active_count}/{len(current)}"
        )
        print(
            "Voltage range (V):",
            f"{np.nanmin(voltage):.4f} to {np.nanmax(voltage):.4f}"
        )
        print(
            "Temperature range (°C):",
            f"{np.nanmin(temperature):.4f} to {np.nanmax(temperature):.4f}"
        )

print("\nAudit complete. No source files were modified.")
