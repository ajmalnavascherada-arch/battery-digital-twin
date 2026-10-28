
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = (
    ROOT / "results_nasa_original"
    / "hybrid_thermal_model"
    / "hybrid_thermal_metrics.csv"
)
OUT = ROOT / "results_nasa_original" / "hybrid_thermal_model"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

test = df[df["cycle_number"].isin([4, 6, 10, 20])].copy()
if test.empty:
    raise ValueError("No held-out cycles found in the metrics file.")

metrics = [
    "active_rmse_c",
    "active_mae_c",
    "post_rmse_c",
    "post_mae_c",
    "full_rmse_c",
    "full_mae_c",
]

summary_rows = []
for metric in metrics:
    summary_rows.append({
        "metric": metric,
        "mean_held_out": float(test[metric].mean()),
        "minimum_held_out": float(test[metric].min()),
        "maximum_held_out": float(test[metric].max()),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "hybrid_held_out_summary.csv", index=False)

print("=== HYBRID THERMAL MODEL: HELD-OUT SUMMARY ===")
print(summary.round(4).to_string(index=False))

print("\nValidation scope:")
print("- Heating parameters were fitted on cycle 2.")
print("- Heating parameters were frozen for cycles 4, 6, 10 and 20.")
print("- Cooling time constant was fitted previously on cycle 2.")
print("- Cooling predictions are initialized using each cycle's measured")
print("  temperature at the end of active discharge.")
print("- These cycles all come from the same NASA B0005 cell.")
print("- This is cross-cycle validation, not independent-cell validation.")

print(f"\nSaved summary to: {OUT / 'hybrid_held_out_summary.csv'}")
