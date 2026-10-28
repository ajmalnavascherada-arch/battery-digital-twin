
"""Benchmark SOH models using leave-one-cell-out validation."""
from pathlib import Path
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    "B0005": ROOT / "results_nasa_original" / "B0005_discharge_capacity.csv",
    "B0006": ROOT / "results_B0006" / "B0006_discharge_capacity.csv",
    "B0007": ROOT / "results_B0007" / "B0007_discharge_capacity.csv",
}

OUT = ROOT / "results_capacity_comparison" / "soh_model_benchmarks"
OUT.mkdir(parents=True, exist_ok=True)


def load_cell(cell_id, path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")

    df = pd.read_csv(path)
    required = {"cycle_number", "capacity_Ah"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["cycle_number"] = pd.to_numeric(df["cycle_number"], errors="coerce")
    df["capacity_Ah"] = pd.to_numeric(df["capacity_Ah"], errors="coerce")
    df = df.dropna(subset=["cycle_number", "capacity_Ah"])
    df = df[np.isfinite(df["capacity_Ah"]) & (df["capacity_Ah"] > 0)]
    df = df.sort_values("cycle_number").drop_duplicates(
        "cycle_number", keep="first"
    ).reset_index(drop=True)

    if len(df) < 3:
        raise ValueError(f"Too few observations for {cell_id}")

    df["cell_id"] = cell_id
    df["discharge_index"] = np.arange(1, len(df) + 1)
    df["soh_pct"] = 100.0 * df["capacity_Ah"] / df["capacity_Ah"].iloc[0]
    return df


cells = {cell: load_cell(cell, path) for cell, path in SOURCES.items()}

model_templates = {
    "LinearRegression": LinearRegression(),
    "HuberRegressor": HuberRegressor(
        epsilon=1.35, alpha=0.0001, max_iter=2000
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    ),
}

metric_rows = []
prediction_frames = []

for held_out, test in cells.items():
    train_cells = [name for name in cells if name != held_out]
    train = pd.concat([cells[name] for name in train_cells], ignore_index=True)

    x_train = train[["discharge_index"]].to_numpy(dtype=float) / 100.0
    y_train = train["soh_pct"].to_numpy(dtype=float)
    x_test = test[["discharge_index"]].to_numpy(dtype=float) / 100.0
    y_test = test["soh_pct"].to_numpy(dtype=float)

    reference_capacity = float(test["capacity_Ah"].iloc[0])
    true_capacity = test["capacity_Ah"].to_numpy(dtype=float)

    predictions_by_model = {
        "Persistence_100pct": np.full(len(test), 100.0)
    }

    for model_name, template in model_templates.items():
        model = copy.deepcopy(template)
        model.fit(x_train, y_train)
        predictions_by_model[model_name] = model.predict(x_test)

    for model_name, pred_soh in predictions_by_model.items():
        pred_capacity = pred_soh / 100.0 * reference_capacity

        metric_rows.append({
            "held_out_cell": held_out,
            "training_cells": "+".join(train_cells),
            "model": model_name,
            "n_train": len(train),
            "n_test": len(test),
            "SOH_MAE_pp": mean_absolute_error(y_test, pred_soh),
            "SOH_RMSE_pp": np.sqrt(mean_squared_error(y_test, pred_soh)),
            "SOH_R2": r2_score(y_test, pred_soh),
            "capacity_MAE_Ah": mean_absolute_error(
                true_capacity, pred_capacity
            ),
            "capacity_RMSE_Ah": np.sqrt(
                mean_squared_error(true_capacity, pred_capacity)
            ),
        })

        pred = test[
            ["cell_id", "cycle_number", "discharge_index",
             "capacity_Ah", "soh_pct"]
        ].copy()
        pred["held_out_cell"] = held_out
        pred["model"] = model_name
        pred["predicted_soh_pct"] = pred_soh
        pred["predicted_capacity_Ah"] = pred_capacity
        pred_frames = pred.copy()
        prediction_frames.append(pred_frames)

metrics = pd.DataFrame(metric_rows)
predictions = pd.concat(prediction_frames, ignore_index=True)

metrics.to_csv(OUT / "benchmark_metrics_by_cell.csv", index=False)
predictions.to_csv(OUT / "benchmark_predictions.csv", index=False)

mean_metrics = metrics.groupby("model", as_index=False).agg(
    mean_SOH_MAE_pp=("SOH_MAE_pp", "mean"),
    mean_SOH_RMSE_pp=("SOH_RMSE_pp", "mean"),
    mean_SOH_R2=("SOH_R2", "mean"),
    mean_capacity_MAE_Ah=("capacity_MAE_Ah", "mean"),
    mean_capacity_RMSE_Ah=("capacity_RMSE_Ah", "mean"),
)
mean_metrics.to_csv(OUT / "benchmark_mean_metrics.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, cell_id in zip(axes, cells):
    observed = cells[cell_id]
    ax.plot(
        observed["discharge_index"], observed["soh_pct"],
        label="Observed", linewidth=2
    )

    for model_name in predictions["model"].unique():
        subset = predictions[
            (predictions["held_out_cell"] == cell_id)
            & (predictions["model"] == model_name)
        ].sort_values("discharge_index")

        ax.plot(
            subset["discharge_index"],
            subset["predicted_soh_pct"],
            label=model_name,
            linewidth=1.1,
        )

    ax.set_title(f"Held-out {cell_id}")
    ax.set_xlabel("Discharge observation index")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Relative capacity retention / SOH (%)")
axes[-1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "benchmark_predictions.png", dpi=180)
plt.close(fig)

print("Per-cell benchmark metrics:")
print(metrics.round(4).to_string(index=False))

print("\nMean metrics across held-out cells:")
print(mean_metrics.round(4).to_string(index=False))

print(f"\nAll benchmark outputs saved to: {OUT}")
print(
    "\nCaution: this tests relative capacity retention using observation index. "
    "It does not predict absolute capacity, future RUL, or other battery designs."
)
