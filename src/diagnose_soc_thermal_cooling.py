
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = (
    ROOT / "results_nasa_original"
    / "soc_dependent_thermal_models"
)
OUT = ROOT / "results_nasa_original" / "soc_thermal_cooling_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

CYCLES = [2, 4, 6, 10, 20]
ACTIVE_CURRENT_A = 0.5

rows = []

for cycle in CYCLES:
    path = PRED_DIR / f"soc_dependent_heating_cycle_{cycle}_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path).sort_values("time_s").reset_index(drop=True)

    active = df["discharge_current_a"].to_numpy() >= ACTIVE_CURRENT_A
    if not active.any():
        print(f"Skipping cycle {cycle}: no active samples.")
        continue

    last_active = np.flatnonzero(active)[-1]
    post = df.iloc[last_active + 1:].copy()

    if post.empty:
        print(f"Skipping cycle {cycle}: no post-discharge samples.")
        continue

    error = (
        post["temperature_predicted_c"].to_numpy()
        - post["temperature_measured_c"].to_numpy()
    )

    rows.append({
        "cycle_number": cycle,
        "post_samples": len(post),
        "post_duration_s": float(
            post["time_s"].iloc[-1] - post["time_s"].iloc[0]
        ),
        "measured_start_c": float(
            post["temperature_measured_c"].iloc[0]
        ),
        "predicted_start_c": float(
            post["temperature_predicted_c"].iloc[0]
        ),
        "measured_end_c": float(
            post["temperature_measured_c"].iloc[-1]
        ),
        "predicted_end_c": float(
            post["temperature_predicted_c"].iloc[-1]
        ),
        "post_rmse_c": float(np.sqrt(np.mean(error ** 2))),
        "post_mae_c": float(np.mean(np.abs(error))),
        "post_bias_pred_minus_measured_c": float(np.mean(error)),
    })

results = pd.DataFrame(rows)
results.to_csv(OUT / "soc_dependent_post_discharge_metrics.csv", index=False)

print("=== SOC-DEPENDENT MODEL: POST-DISCHARGE DIAGNOSTIC ===")
print(results.round(4).to_string(index=False))
print(f"\nSaved to: {OUT}")
