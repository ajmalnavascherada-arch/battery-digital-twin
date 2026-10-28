
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed_nasa_original"
    / "B0005_signals_long.csv"
)

df = pd.read_csv(INPUT_FILE)

print("\n=== AVAILABLE SIGNAL TYPES ===")
print(df["signal"].value_counts().to_string())

print("\n=== SIGNALS BY CYCLE TYPE ===")
print(
    df.groupby(["cycle_type", "signal"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\n=== TEMPERATURE-RELATED SIGNALS ===")
temperature_signals = [
    signal
    for signal in df["signal"].dropna().unique()
    if "temp" in str(signal).lower()
]

if temperature_signals:
    print("Found:", temperature_signals)
    for signal in temperature_signals:
        subset = df.loc[df["signal"] == signal, "value"]
        print(
            f"{signal}: "
            f"count={subset.count()}, "
            f"min={subset.min():.4f}, "
            f"max={subset.max():.4f}"
        )
else:
    print("No temperature-related signal names found.")

print("\n=== CELL AND CYCLE SUMMARY ===")
print("Cell IDs:", df["cell_id"].unique().tolist())
print("Cycle types:", df["cycle_type"].unique().tolist())
print("Cycle range:", df["cycle_number"].min(), "to", df["cycle_number"].max())
print("Aligned rows:", int(df["time_aligned"].sum()))
print("Total rows:", len(df))
