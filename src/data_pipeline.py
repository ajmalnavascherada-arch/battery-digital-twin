"""Extract NASA battery ageing MATLAB records into auditable CSV files.

Important: signals are exported independently. If a signal length differs from
the cycle's data.Time length, its time_s values are left blank rather than
inventing a row-by-row alignment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


SIGNAL_FIELDS = (
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
    "Current_charge",
    "Voltage_charge",
    "Capacity",
)


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a scipy MATLAB struct, dict, or structured ndarray."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return default


def as_float_array(value: Any) -> np.ndarray:
    """Convert a MATLAB numeric value to a flat float array."""
    if value is None:
        return np.array([], dtype=float)
    try:
        # Preserve every sample position. Dropping NaNs would shift indices and
        # could create a false alignment between signals and timestamps.
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def matlab_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return matlab_text(value.item())
    return str(value) if value is not None else "unknown"


def extract_cycles(mat_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cycle-level metadata and independent signal samples."""
    loaded = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    top_names = [k for k in loaded if not k.startswith("__")]
    if len(top_names) != 1:
        raise ValueError(
            f"Expected one top-level MATLAB variable, found {top_names}. "
            "Inspect the file and adapt the parser if needed."
        )

    root_name = top_names[0]
    root = loaded[root_name]
    cycles_raw = get_field(root, "cycle")
    if cycles_raw is None:
        raise ValueError(f"Could not find '{root_name}.cycle' in the MAT file.")

    cycles = np.asarray(cycles_raw, dtype=object).reshape(-1)
    cycle_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []

    for cycle_number, cycle in enumerate(cycles, start=1):
        cycle_type = matlab_text(get_field(cycle, "type", "unknown")).lower()
        ambient_raw = get_field(cycle, "ambient_temperature")
        ambient_arr = as_float_array(ambient_raw)
        ambient_c = (
            float(ambient_arr[0])
            if ambient_arr.size and np.isfinite(ambient_arr[0]) else np.nan
        )
        data = get_field(cycle, "data")
        time_s = as_float_array(get_field(data, "Time"))

        lengths: dict[str, int] = {}
        capacity_arr = as_float_array(get_field(data, "Capacity"))
        capacity_ah = (
            float(capacity_arr[0])
            if capacity_arr.size and np.isfinite(capacity_arr[0]) else np.nan
        )

        for field in SIGNAL_FIELDS:
            raw = get_field(data, field)
            values = as_float_array(raw)
            if values.size == 0:
                continue
            lengths[field] = int(values.size)

            # Capacity is a cycle-level scalar, not a sample time series.
            if field == "Capacity" and values.size == 1:
                continue

            aligned = values.size == time_s.size and time_s.size > 0
            for sample_idx, value in enumerate(values):
                signal_rows.append({
                    "source_file": mat_path.name,
                    "cell_id": root_name,
                    "cycle_number": cycle_number,
                    "cycle_type": cycle_type,
                    "signal": field,
                    "sample_index": sample_idx,
                    "time_s": float(time_s[sample_idx]) if aligned else np.nan,
                    "value": float(value),
                    "time_aligned": bool(aligned),
                })

        # Summarise lengths without discarding the original signals.
        time_length = int(time_s.size)
        mismatch_fields = [
            f"{field}={length}"
            for field, length in lengths.items()
            if field != "Capacity" and length != time_length
        ]
        cycle_rows.append({
            "source_file": mat_path.name,
            "cell_id": root_name,
            "cycle_number": cycle_number,
            "cycle_type": cycle_type,
            "ambient_temperature_C": ambient_c,
            "capacity_Ah": capacity_ah,
            "time_samples": time_length,
            "signal_lengths": ";".join(f"{k}={v}" for k, v in sorted(lengths.items())),
            "length_mismatch": bool(mismatch_fields),
            "mismatch_details": ";".join(mismatch_fields),
        })

    return pd.DataFrame(cycle_rows), pd.DataFrame(signal_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_file", type=Path, help="Input NASA .mat file")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/processed"),
        help="Output folder (default: data/processed)",
    )
    args = parser.parse_args()

    if not args.mat_file.is_file():
        parser.error(f"Input file not found: {args.mat_file}")
    if args.mat_file.suffix.lower() != ".mat":
        parser.error("Input must be a .mat file")

    cycles_df, signals_df = extract_cycles(args.mat_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.mat_file.stem
    cycles_path = args.out_dir / f"{stem}_cycle_summary.csv"
    signals_path = args.out_dir / f"{stem}_signals_long.csv"
    cycles_df.to_csv(cycles_path, index=False)
    signals_df.to_csv(signals_path, index=False)

    print(f"Input file: {args.mat_file}")
    print(f"Cycles extracted: {len(cycles_df)}")
    print(f"Cycle types: {cycles_df['cycle_type'].value_counts(dropna=False).to_dict()}")
    print(f"Signal samples exported: {len(signals_df)}")
    print(f"Signals with missing time alignment: "
          f"{int((~signals_df['time_aligned']).sum()) if not signals_df.empty else 0}")
    print(f"Cycle summary: {cycles_path}")
    print(f"Signal data:   {signals_path}")
    if not cycles_df.empty:
        discharges = cycles_df[
            (cycles_df["cycle_type"] == "discharge") &
            cycles_df["capacity_Ah"].notna()
        ]
        print(f"Discharge cycles with capacity: {len(discharges)}")
        if not discharges.empty:
            print(f"First discharge capacity: {discharges.iloc[0]['capacity_Ah']:.6f} Ah")


if __name__ == "__main__":
    main()
