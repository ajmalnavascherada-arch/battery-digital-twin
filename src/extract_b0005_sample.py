
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "nasa_original" / "B0005.mat"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_20"
OUTPUT_FILE = OUTPUT_DIR / "B0005_first_20_rows.csv"

N_ROWS = 20


def as_list(value):
    """Convert MATLAB cycle structures into a Python list."""
    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.reshape(-1))

    return [value]


def to_array(value):
    """Convert a value into a NumPy array."""
    return np.asarray(value)


def main():
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            f"NASA dataset not found: {SOURCE_FILE}"
        )

    # Load the source without modifying it.
    mat = loadmat(SOURCE_FILE, simplify_cells=True)

    if "B0005" not in mat:
        raise KeyError(
            "The MATLAB file does not contain the expected "
            "'B0005' variable."
        )

    battery = mat["B0005"]
    cycles = as_list(battery["cycle"])

    rows = []

    # Preserve the dataset's original cycle order.
    for cycle_index, cycle in enumerate(cycles, start=1):
        if not isinstance(cycle, dict):
            continue

        data = cycle.get("data")
        if not isinstance(data, dict):
            continue

        arrays = {
            key: to_array(value)
            for key, value in data.items()
        }

        # Find the number of measurement samples in this cycle.
        lengths = [
            arr.shape[0] if arr.ndim > 1 else arr.size
            for arr in arrays.values()
            if arr.ndim > 0 and arr.size > 1
        ]

        if not lengths:
            continue

        n_samples = max(lengths)

        for sample_index in range(n_samples):
            row = {
                "cycle_index": cycle_index,
                "cycle_type": str(cycle.get("type", "unknown")),
                "sample_index": sample_index,
                "ambient_temperature": cycle.get(
                    "ambient_temperature", np.nan
                ),
            }

            for key, arr in arrays.items():
                if arr.ndim == 0 or arr.size == 1:
                    value = arr.item()
                elif sample_index < (
                    arr.shape[0] if arr.ndim > 1 else arr.size
                ):
                    value = arr[sample_index]
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                else:
                    value = np.nan

                # Keep the CSV cells simple and readable.
                if isinstance(value, list):
                    value = str(value)

                row[key] = value

            rows.append(row)

            # Stop after the first 20 measurement rows overall.
            if len(rows) >= N_ROWS:
                break

        if len(rows) >= N_ROWS:
            break

    if not rows:
        raise ValueError(
            "No measurement rows were extracted. "
            "Inspect the MATLAB structure before proceeding."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows[:N_ROWS])
    df.to_csv(OUTPUT_FILE, index=False)

    print("NASA B0005 extraction complete")
    print("==============================")
    print(f"Source: {SOURCE_FILE}")
    print(f"Rows exported: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Cycle types: {df['cycle_type'].unique().tolist()}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
