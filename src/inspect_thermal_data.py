
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed_nasa_original"
    / "B0005_signals_long.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "thermal_diagnostic"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print("Available columns:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    # Identify likely column names without silently assuming a schema.
    cycle_col = next(
        (c for c in ["cycle_number", "cycle", "cycle_id"] if c in df.columns),
        None,
    )
    type_col = next(
        (c for c in ["cycle_type", "type"] if c in df.columns),
        None,
    )
    temp_col = next(
        (
            c for c in [
                "Temperature_measured",
                "temperature_measured",
                "temperature_c",
                "temperature",
            ]
            if c in df.columns
        ),
        None,
    )
    time_col = next(
        (
            c for c in [
                "time_s",
                "Time",
                "time",
                "relative_time_s",
            ]
            if c in df.columns
        ),
        None,
    )

    print("\nDetected columns:")
    print(f"Cycle: {cycle_col}")
    print(f"Cycle type: {type_col}")
    print(f"Temperature: {temp_col}")
    print(f"Time: {time_col}")

    if temp_col is None:
        raise ValueError(
            "Could not identify the temperature column. "
            "Use the printed column names to select the correct field."
        )

    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    valid_temp = df[temp_col].dropna()

    print("\nTemperature summary:")
    print(valid_temp.describe().to_string())

    print(f"\nRows with valid temperature: {len(valid_temp)}")
    print(f"Rows with missing temperature: {df[temp_col].isna().sum()}")

    if cycle_col is not None:
        print("\nCycle counts:")
        print(df[cycle_col].value_counts().sort_index().to_string())

    if type_col is not None:
        print("\nCycle-type counts:")
        print(df[type_col].value_counts(dropna=False).to_string())

    if cycle_col is not None and type_col is not None:
        print("\nTemperature availability by cycle type:")
        print(
            df.groupby(type_col, dropna=False)[temp_col]
            .agg(["count", "mean", "min", "max"])
            .to_string()
        )

    if time_col is not None:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        print("\nTime summary:")
        print(df[time_col].describe().to_string())

    # Save a compact audit report.
    report_file = OUTPUT_DIR / "thermal_data_audit.txt"
    with report_file.open("w", encoding="utf-8") as f:
        f.write("NASA B0005 thermal data audit\n")
        f.write("=============================\n\n")
        f.write(f"Input: {INPUT_FILE}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {df.columns.tolist()}\n")
        f.write(f"Temperature column: {temp_col}\n")
        f.write(f"Valid temperature rows: {len(valid_temp)}\n")
        f.write(
            f"Missing temperature rows: {df[temp_col].isna().sum()}\n\n"
        )
        f.write("Temperature summary:\n")
        f.write(valid_temp.describe().to_string() + "\n")

    print(f"\nAudit report saved to: {report_file}")


if __name__ == "__main__":
    main()
