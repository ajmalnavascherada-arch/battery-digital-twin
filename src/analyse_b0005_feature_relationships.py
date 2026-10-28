
"""Explore relationships between B0005 cycle features and capacity retention."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset" / "B0005_cycle_features.csv"
)
OUT = ROOT / "results_capacity_comparison" / "b0005_feature_dataset"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA).sort_values("discharge_index").reset_index(drop=True)

target = "relative_capacity_retention_pct"
candidate_features = [
    "discharge_duration_s",
    "voltage_drop_V",
    "initial_voltage_V",
    "final_voltage_V",
    "initial_temperature_C",
    "final_temperature_C",
    "temperature_rise_C",
    "mean_temperature_C",
    "maximum_temperature_C",
    "active_discharge_samples",
    "signal_sample_count",
]

# Keep numeric columns with meaningful variation.
candidate_features = [
    c for c in candidate_features
    if c in df.columns and df[c].nunique(dropna=True) > 1
]

correlations = df[candidate_features + [target]].corr(method="spearman")[target]
correlations = correlations.drop(target).sort_values(
    key=lambda s: s.abs(), ascending=False
)

correlations.rename("spearman_correlation").to_csv(
    OUT / "B0005_feature_correlations.csv"
)

print("\nSpearman correlation with relative capacity retention")
print("(Exploratory only; correlations do not establish causation.)")
print(correlations.round(4).to_string())

# Plot the capacity trajectory.
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(df["discharge_index"], df[target], marker=".", linewidth=1)
ax.set_xlabel("Discharge observation index")
ax.set_ylabel("Relative capacity retention (%)")
ax.set_title("B0005 capacity retention across discharge observations")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "B0005_capacity_retention.png", dpi=160)
plt.close(fig)

# Plot feature relationships against relative capacity retention.
for feature in candidate_features:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[feature], df[target], alpha=0.75)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"B0005: {feature} vs capacity retention")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"relationship_{feature}.png", dpi=140)
    plt.close(fig)

print("\nRecords:", len(df))
print("Features examined:", len(candidate_features))
print("Saved correlation table and plots to:", OUT)
