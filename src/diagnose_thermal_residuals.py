
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "results_nasa_original" / "effective_thermal_model_exact"
SOC_FILE = (
    ROOT / "results_nasa_original" / "ecm_soc"
    / "B0005_cycle_2_soc.csv"
)
OUT = ROOT / "results_nasa_original" / "thermal_residual_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

CYCLES = [2, 4, 6, 10, 20]
ACTIVE_CURRENT_A = 0.5

all_frames = []

for cycle in CYCLES:
    path = PRED_DIR / f"B0005_cycle_{cycle}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}\n"
            "Run fit_effective_thermal_model_exact.py first."
        )

    df = pd.read_csv(path)
    df["cycle_number"] = cycle

    if "discharge_current_a" not in df.columns:
        raise ValueError(f"Current column missing in {path}")

    df["active_discharge"] = (
        df["discharge_current_a"] >= ACTIVE_CURRENT_A
    )
    df["residual_c"] = (
        df["temperature_measured_c"]
        - df["temperature_predicted_c"]
    )

    # Positive residual means the model underpredicts temperature.
    all_frames.append(df)

combined = pd.concat(all_frames, ignore_index=True)

# Save residual data for reproducibility.
combined.to_csv(OUT / "thermal_residuals_all_cycles.csv", index=False)

summary = (
    combined[combined["active_discharge"]]
    .groupby("cycle_number")
    .agg(
        samples=("residual_c", "size"),
        mean_residual_c=("residual_c", "mean"),
        median_residual_c=("residual_c", "median"),
        residual_std_c=("residual_c", "std"),
        minimum_residual_c=("residual_c", "min"),
        maximum_residual_c=("residual_c", "max"),
    )
    .reset_index()
)

summary.to_csv(OUT / "active_residual_summary.csv", index=False)

print("=== ACTIVE-DISCHARGE RESIDUAL SUMMARY ===")
print(summary.round(4).to_string(index=False))

# Cycle 2: inspect residuals versus elapsed time and current.
ref = combined[combined["cycle_number"] == 2].copy()
ref["elapsed_time_s"] = ref["time_s"] - ref["time_s"].iloc[0]

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].plot(
    ref.loc[ref["active_discharge"], "elapsed_time_s"],
    ref.loc[ref["active_discharge"], "residual_c"],
    marker=".",
)
axes[0].axhline(0, linestyle="--")
axes[0].set_ylabel("Measured − predicted (°C)")
axes[0].set_title("Cycle 2 active-discharge residuals")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(
    ref.loc[ref["active_discharge"], "discharge_current_a"],
    ref.loc[ref["active_discharge"], "residual_c"],
    s=18,
)
axes[1].axhline(0, linestyle="--")
axes[1].set_xlabel("Discharge current (A)")
axes[1].set_ylabel("Residual (°C)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(
    ref["time_s"],
    ref["temperature_measured_c"],
    label="Measured",
)
axes[2].plot(
    ref["time_s"],
    ref["temperature_predicted_c"],
    "--",
    label="Predicted",
)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Temperature (°C)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUT / "cycle_2_thermal_residual_diagnostics.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)

# Summarize residuals in time bins across all cycles.
combined["time_bin"] = pd.cut(
    combined["time_s"],
    bins=[-np.inf, 600, 1200, 1800, 2400, 3000, 3600, np.inf],
    labels=[
        "0–600 s", "600–1200 s", "1200–1800 s",
        "1800–2400 s", "2400–3000 s", "3000–3600 s",
        ">3600 s",
    ],
)

time_summary = (
    combined[combined["active_discharge"]]
    .groupby(["cycle_number", "time_bin"], observed=True)
    .agg(
        samples=("residual_c", "size"),
        mean_residual_c=("residual_c", "mean"),
        rmse_c=("residual_c", lambda x: np.sqrt(np.mean(x**2))),
    )
    .reset_index()
)

time_summary.to_csv(OUT / "thermal_residuals_by_time_bin.csv", index=False)

print("\n=== RESIDUALS BY TIME BIN ===")
print(time_summary.round(4).to_string(index=False))
print(f"\nSaved diagnostic files to: {OUT}")
