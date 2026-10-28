
"""Plot normalized capacity retention for B0005, B0006, and B0007."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "B0005": ROOT / "data" / "processed_nasa_original" / "B0005_cycle_summary.csv",
    "B0006": ROOT / "data" / "processed_B0006" / "B0006_cycle_summary.csv",
    "B0007": ROOT / "data" / "processed_B0007" / "B0007_cycle_summary.csv",
}

OUT_DIR = ROOT / "results_capacity_comparison" / "cross_cell_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))

for cell_id, path in FILES.items():
    df = pd.read_csv(path)

    discharge = df.loc[
        df["cycle_type"].astype(str).str.lower().eq("discharge")
    ].copy()

    discharge["capacity_Ah"] = pd.to_numeric(
        discharge["capacity_Ah"], errors="coerce"
    )
    discharge["cycle_number"] = pd.to_numeric(
        discharge["cycle_number"], errors="coerce"
    )

    discharge = (
        discharge.dropna(subset=["capacity_Ah", "cycle_number"])
        .loc[lambda x: x["capacity_Ah"] > 0]
        .sort_values("cycle_number")
        .reset_index(drop=True)
    )

    if discharge.empty:
        print(f"WARNING: no valid discharge capacities for {cell_id}")
        continue

    initial_capacity = discharge["capacity_Ah"].iloc[0]
    retention = 100 * discharge["capacity_Ah"] / initial_capacity
    observation_index = range(1, len(discharge) + 1)

    ax.plot(
        observation_index,
        retention,
        marker=".",
        markersize=3,
        linewidth=1.3,
        label=cell_id,
    )

ax.set_xlabel("Discharge observation index within each cell")
ax.set_ylabel("Relative capacity retention (%)")
ax.set_title("NASA battery capacity-retention trajectories")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

output = OUT_DIR / "cross_cell_capacity_trajectories.png"
fig.savefig(output, dpi=160)
plt.close(fig)

print("Saved plot to:", output)
