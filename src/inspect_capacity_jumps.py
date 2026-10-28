
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results_capacity_comparison" / "combined_capacity_dataset.csv"

df = pd.read_csv(INPUT)

required = {"cell_id", "cycle_number", "capacity_Ah"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df = df.sort_values(["cell_id", "cycle_number"]).copy()

df["previous_capacity_Ah"] = (
    df.groupby("cell_id")["capacity_Ah"].shift(1)
)
df["capacity_change_Ah"] = (
    df["capacity_Ah"] - df["previous_capacity_Ah"]
)
df["capacity_change_pct"] = (
    100 * df["capacity_change_Ah"] / df["previous_capacity_Ah"]
)

jumps = df.loc[
    df["capacity_change_pct"].abs() > 5,
    [
        "cell_id",
        "cycle_number",
        "previous_capacity_Ah",
        "capacity_Ah",
        "capacity_change_Ah",
        "capacity_change_pct",
    ],
]

print("Observations per cell:")
print(df.groupby("cell_id").size().to_string())

print("\nCapacity changes exceeding 5%:")
if jumps.empty:
    print("None found.")
else:
    print(jumps.to_string(index=False))

out = ROOT / "results_capacity_comparison" / "capacity_jump_audit.csv"
jumps.to_csv(out, index=False)
print(f"\nAudit saved: {out}")
