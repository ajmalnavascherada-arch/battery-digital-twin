
"""Diagnose chronological early-SOH model predictions."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset" / "early_soh_baseline"
)
PREDICTIONS = BASE / "chronological_predictions.csv"
FEATURES = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset" / "B0005_fixed_time_features.csv"
)

OUT_DIR = BASE / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(PREDICTIONS)
features = pd.read_csv(FEATURES)

actual = (
    features[["discharge_index", "relative_capacity_retention_pct"]]
    .rename(columns={"relative_capacity_retention_pct": "actual_retention_pct"})
)

pred = pred.drop(columns=["actual_retention_pct"], errors="ignore")
pred = pred.merge(actual, on="discharge_index", how="left")
pred["residual_pp"] = (
    pred["actual_retention_pct"] - pred["predicted_retention_pct"]
)

print("Residual definition: actual minus predicted.")
print("\nResidual summary by model:")
for model_name, group in pred.groupby("model"):
    print(f"\n{model_name}")
    print("Mean residual:", round(group["residual_pp"].mean(), 4))
    print("MAE:", round(group["residual_pp"].abs().mean(), 4))
    print("First test cycle residual:", round(group["residual_pp"].iloc[0], 4))
    print("Last test cycle residual:", round(group["residual_pp"].iloc[-1], 4))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        group["discharge_index"],
        group["actual_retention_pct"],
        marker=".",
        label="Actual retention",
    )
    ax.plot(
        group["discharge_index"],
        group["predicted_retention_pct"],
        marker=".",
        label="Predicted retention",
    )
    ax.set_xlabel("Discharge observation index")
    ax.set_ylabel("Relative capacity retention (%)")
    ax.set_title(f"Chronological SOH predictions: {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_trajectory.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, linestyle="--")
    ax.plot(group["discharge_index"], group["residual_pp"], marker=".")
    ax.set_xlabel("Discharge observation index")
    ax.set_ylabel("Residual (percentage points)")
    ax.set_title(f"Chronological residuals: {model_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}_residuals.png", dpi=160)
    plt.close(fig)

pred.to_csv(OUT_DIR / "prediction_residuals.csv", index=False)
print("\nSaved plots and residual data to:", OUT_DIR)
