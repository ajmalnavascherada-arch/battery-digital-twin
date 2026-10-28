
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results_capacity_comparison" / "combined_capacity_dataset.csv"
OUT_DIR = ROOT / "results_capacity_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

fig, ax = plt.subplots(figsize=(10, 6))

for cell_id, group in df.groupby("cell_id"):
    group = group.sort_values("cycle_number")
    ax.plot(
        group["cycle_number"],
        group["capacity_Ah"],
        marker="o",
        markersize=2.5,
        linewidth=1.1,
        label=cell_id,
    )

ax.set_title("NASA Battery Cells: Observed Discharge Capacity")
ax.set_xlabel("Recorded cycle number")
ax.set_ylabel("Discharge capacity (Ah)")
ax.grid(True, alpha=0.3)
ax.legend(title="Cell")
fig.tight_layout()

output = OUT_DIR / "capacity_trajectories_all_cells.png"
fig.savefig(output, dpi=180)
plt.close(fig)

print(f"Plot saved: {output}")
print("\nObservations per cell:")
print(df.groupby("cell_id")["capacity_Ah"].agg(["count", "min", "max"]).to_string())
