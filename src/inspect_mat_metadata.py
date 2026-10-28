
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
MAT_FILE = ROOT / "data" / "raw" / "nasa_original" / "B0005.mat"


def as_records(value):
    """Normalize MATLAB cycle structures into a Python list."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return list(value.reshape(-1))
    return [value]


def get_field(obj, name):
    """Read a field from a simplified MATLAB structure."""
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def describe_value(value, indent="  ", max_items=8):
    """Print a compact summary without dumping large signal arrays."""
    if value is None:
        print(f"{indent}None / not present")
    elif isinstance(value, dict):
        print(f"{indent}Structure fields: {list(value.keys())}")
    elif isinstance(value, (list, tuple)):
        print(f"{indent}Sequence length: {len(value)}")
        for item in value[:max_items]:
            describe_value(item, indent + "  ", max_items)
        if len(value) > max_items:
            print(f"{indent}  ...")
    else:
        arr = np.asarray(value)
        if arr.size == 1:
            print(f"{indent}Value: {arr.item()}")
        elif np.issubdtype(arr.dtype, np.number):
            finite = arr[np.isfinite(arr)] if np.issubdtype(
                arr.dtype, np.floating
            ) else arr.reshape(-1)

            if finite.size:
                print(
                    f"{indent}Array shape={arr.shape}, dtype={arr.dtype}, "
                    f"min={finite.min():.6g}, max={finite.max():.6g}"
                )
            else:
                print(f"{indent}Array shape={arr.shape}, no finite values")
        else:
            print(f"{indent}Array shape={arr.shape}, dtype={arr.dtype}")


def main():
    if not MAT_FILE.exists():
        raise FileNotFoundError(f"NASA MAT file not found: {MAT_FILE}")

    print(f"Reading: {MAT_FILE}")

    mat = loadmat(MAT_FILE, simplify_cells=True)
    top_keys = [key for key in mat if not key.startswith("__")]

    print("\n=== TOP-LEVEL MATLAB VARIABLES ===")
    print(top_keys)

    if "B0005" not in mat:
        raise KeyError(
            f"'B0005' variable not found. Available variables: {top_keys}"
        )

    battery = mat["B0005"]

    print("\n=== BATTERY STRUCTURE ===")
    describe_value(battery)

    if not isinstance(battery, dict):
        raise TypeError(
            "B0005 is not a dictionary after MATLAB simplification. "
            "Inspect the structure summary above before proceeding."
        )

    cycles_value = battery.get("cycle")
    if cycles_value is None:
        raise KeyError(
            f"No 'cycle' field found. Fields: {list(battery.keys())}"
        )

    cycles = as_records(cycles_value)

    print(f"\n=== CYCLE SUMMARY: {len(cycles)} records ===")
    print("Cycle | Type | Ambient temp | Data fields | Temp range (°C)")

    cycle_summaries = []

    for index, cycle in enumerate(cycles, start=1):
        cycle_type = get_field(cycle, "type")
        ambient = get_field(cycle, "ambient_temperature")
        data = get_field(cycle, "data")

        if isinstance(cycle_type, np.ndarray) and cycle_type.size == 1:
            cycle_type = cycle_type.item()

        if isinstance(ambient, np.ndarray) and ambient.size == 1:
            ambient = ambient.item()

        data_fields = (
            list(data.keys()) if isinstance(data, dict) else []
        )

        temp = (
            get_field(data, "Temperature_measured")
            if isinstance(data, dict)
            else None
        )

        if temp is not None:
            temp_arr = np.asarray(temp, dtype=float).reshape(-1)
            temp_arr = temp_arr[np.isfinite(temp_arr)]
        else:
            temp_arr = np.array([])

        if temp_arr.size:
            temp_range = (
                f"{temp_arr.min():.3f} to {temp_arr.max():.3f}"
            )
        else:
            temp_range = "not available"

        ambient_text = (
            f"{float(ambient):.3f}"
            if ambient is not None
            else "not available"
        )

        print(
            f"{index:5d} | {str(cycle_type):9s} | "
            f"{ambient_text:13s} | {data_fields} | {temp_range}"
        )

        cycle_summaries.append({
            "cycle_index": index,
            "cycle_type": str(cycle_type),
            "ambient_temperature": ambient,
            "data_fields": data_fields,
            "temperature_samples": int(temp_arr.size),
            "temperature_min_c": (
                float(temp_arr.min()) if temp_arr.size else None
            ),
            "temperature_max_c": (
                float(temp_arr.max()) if temp_arr.size else None
            ),
        })

    print("\n=== FIRST CHARGE CYCLE DATA FIELDS ===")
    for cycle in cycles:
        if get_field(cycle, "type") == "charge":
            data = get_field(cycle, "data")
            describe_value(data)
            break

    print("\n=== FIRST DISCHARGE CYCLE DATA FIELDS ===")
    for cycle in cycles:
        if get_field(cycle, "type") == "discharge":
            data = get_field(cycle, "data")
            describe_value(data)
            break

    print("\n=== INTERPRETATION CHECK ===")
    ambient_values = [
        item["ambient_temperature"]
        for item in cycle_summaries
        if item["ambient_temperature"] is not None
    ]

    if ambient_values:
        values = np.asarray(ambient_values, dtype=float).reshape(-1)
        print(
            "Ambient-temperature field found. "
            f"Range: {values.min():.3f} to {values.max():.3f}"
        )
    else:
        print("No ambient-temperature values found in cycle metadata.")

    print(
        "\nNote: the presence of a field does not guarantee that it is "
        "a measured time-varying ambient temperature. Inspect its meaning "
        "and units before using it in a physical model."
    )


if __name__ == "__main__":
    main()
