
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = (
    ROOT / "results_capacity_comparison"
    / "soh_cross_cell_validation"
    / "cross_cell_predictions.csv"
)
OUT = DATA.parent

df = pd.read_csv(DATA)

for cell_id, group in df.groupby("held_out_cell"):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for model in group["model"].unique():
        subset = group[group["model"] == model].sort_values("discharge_index")
        residual = (
            subset["predicted_soh_pct"] - subset["soh_pct"]
        )
        ax.plot(
            subset["discharge_index"],
            residual,
            label=model,
            linewidth=1.2,
        )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title(f"SOH prediction residuals — held-out {cell_id}")
    ax.set_xlabel("Discharge observation index")
    ax.set_ylabel("Prediction error (percentage points)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUT / f"residuals_{cell_id}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")

print("\nResidual convention: predicted SOH minus observed SOH.")
