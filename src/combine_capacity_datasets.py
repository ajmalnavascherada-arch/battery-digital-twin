from pathlib import Path
import pandas as pd

ROOT = Path.cwd()

SOURCES = {
    "B0005": ROOT / "results_nasa_original" / "B0005_discharge_capacity.csv",
    "B0006": ROOT / "results_B0006" / "B0006_discharge_capacity.csv",
    "B0007": ROOT / "results_B0007" / "B0007_discharge_capacity.csv",
}

REQUIRED = {"cycle_number", "capacity_Ah", "ambient_temperature_C"}

frames = []

for cell_id, path in SOURCES.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.copy()
    df.insert(0, "cell_id", cell_id)
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)

out_dir = ROOT / "results_capacity_comparison"
out_dir.mkdir(parents=True, exist_ok=True)

combined_path = out_dir / "combined_capacity_dataset.csv"
summary_path = out_dir / "capacity_comparison_summary.csv"

combined.to_csv(combined_path, index=False)

summary = (
    combined.groupby("cell_id", sort=True)
    .agg(
        valid_records=("capacity_Ah", "count"),
        first_recorded_cycle=("cycle_number", "min"),
        last_recorded_cycle=("cycle_number", "max"),
        first_capacity_Ah=("capacity_Ah", "first"),
        last_capacity_Ah=("capacity_Ah", "last"),
    )
    .reset_index()
)

summary["capacity_change_Ah"] = (
    summary["last_capacity_Ah"] - summary["first_capacity_Ah"]
)
summary["capacity_change_pct"] = (
    100 * summary["capacity_change_Ah"] / summary["first_capacity_Ah"]
)

summary.to_csv(summary_path, index=False)

print("Combined dataset created:", combined_path)
print("Comparison summary created:", summary_path)
print("\nRows per cell:")
print(combined.groupby("cell_id").size().to_string())
print("\nCapacity comparison:")
print(summary.to_string(index=False))
