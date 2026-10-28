
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results_nasa_original" / "thermal_model" / (
    "B0005_thermal_cross_cycle_predictions.csv"
)
OUT = ROOT / "results_nasa_original" / "thermal_model"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

cycles = [2, 4, 6, 10, 20]
fig, axes = plt.subplots(
    len(cycles), 1, figsize=(11, 14), sharex=True
)

for ax, cycle_number in zip(axes, cycles):
    part = df[df["cycle_number"] == cycle_number].sort_values(
        "elapsed_time_s"
    )

    ax.plot(
        part["elapsed_time_s"],
        part["Temperature_measured"],
        label="Measured",
    )
    ax.plot(
        part["elapsed_time_s"],
        part["predicted_temperature_c"],
        label="Predicted",
    )
    ax.set_ylabel("Temp (°C)")
    ax.set_title(f"Cycle {cycle_number}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

axes[-1].set_xlabel("Elapsed active-discharge time (s)")
fig.suptitle("NASA B0005 — Frozen thermal-model validation", y=1.01)
fig.tight_layout()

plot_path = OUT / "B0005_thermal_cross_cycle_validation.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {plot_path}")
