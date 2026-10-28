
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "B0005": ROOT / "results_nasa_original" / "B0005_discharge_capacity.csv",
    "B0006": ROOT / "results_B0006" / "B0006_discharge_capacity.csv",
    "B0007": ROOT / "results_B0007" / "B0007_discharge_capacity.csv",
}
OUT = ROOT / "results_capacity_comparison" / "soh_quadratic_validation"
OUT.mkdir(parents=True, exist_ok=True)


def load_cell(cell_id, path):
    df = pd.read_csv(path)
    df["cycle_number"] = pd.to_numeric(df["cycle_number"], errors="coerce")
    df["capacity_Ah"] = pd.to_numeric(df["capacity_Ah"], errors="coerce")
    df = df.dropna(subset=["cycle_number", "capacity_Ah"])
    df = df[np.isfinite(df["capacity_Ah"]) & (df["capacity_Ah"] > 0)]
    df = df.sort_values("cycle_number").drop_duplicates(
        "cycle_number", keep="first"
    ).reset_index(drop=True)

    df["cell_id"] = cell_id
    df["discharge_index"] = np.arange(1, len(df) + 1)
    df["soh_pct"] = 100 * df["capacity_Ah"] / df["capacity_Ah"].iloc[0]
    return df


cells = {cell: load_cell(cell, path) for cell, path in SOURCES.items()}
models = {
    "Linear": make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False), LinearRegression()
    ),
    "Quadratic": make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
    ),
}

metric_rows = []
prediction_frames = []

for held_out, test in cells.items():
    train = pd.concat(
        [df for cell, df in cells.items() if cell != held_out],
        ignore_index=True,
    )

    # Scale index for numerical stability.
    x_train = train[["discharge_index"]].to_numpy(dtype=float) / 100
    y_train = train["soh_pct"].to_numpy(dtype=float)
    x_test = test[["discharge_index"]].to_numpy(dtype=float) / 100
    y_test = test["soh_pct"].to_numpy(dtype=float)

    reference_capacity = float(test["capacity_Ah"].iloc[0])
    true_capacity = test["capacity_Ah"].to_numpy(dtype=float)

    for model_name, template in models.items():
        import copy
        model = copy.deepcopy(template)
        model.fit(x_train, y_train)
        predicted_soh = model.predict(x_test)
        predicted_capacity = predicted_soh / 100 * reference_capacity

        metric_rows.append({
            "held_out_cell": held_out,
            "model": model_name,
            "SOH_MAE_pp": mean_absolute_error(y_test, predicted_soh),
            "SOH_RMSE_pp": np.sqrt(mean_squared_error(y_test, predicted_soh)),
            "SOH_R2": r2_score(y_test, predicted_soh),
            "capacity_MAE_Ah": mean_absolute_error(
                true_capacity, predicted_capacity
            ),
            "capacity_RMSE_Ah": np.sqrt(
                mean_squared_error(true_capacity, predicted_capacity)
            ),
        })

        pred = test[
            ["cell_id", "cycle_number", "discharge_index", "capacity_Ah", "soh_pct"]
        ].copy()
        pred["model"] = model_name
        pred["predicted_soh_pct"] = predicted_soh
        pred["residual_pp"] = predicted_soh - y_test
        prediction_frames.append(pred)

metrics = pd.DataFrame(metric_rows)
predictions = pd.concat(prediction_frames, ignore_index=True)

metrics.to_csv(OUT / "quadratic_model_metrics.csv", index=False)
predictions.to_csv(OUT / "quadratic_model_predictions.csv", index=False)

print("Per-cell results:")
print(metrics.round(4).to_string(index=False))

print("\nMean metrics across the three held-out cells:")
print(
    metrics.groupby("model")[
        ["SOH_MAE_pp", "SOH_RMSE_pp", "SOH_R2",
         "capacity_MAE_Ah", "capacity_RMSE_Ah"]
    ].mean().round(4).to_string()
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, cell_id in zip(axes, cells):
    obs = cells[cell_id]
    ax.plot(
        obs["discharge_index"], obs["soh_pct"],
        label="Observed", linewidth=1.5
    )
    for model_name in models:
        subset = predictions[
            (predictions["cell_id"] == cell_id)
            & (predictions["model"] == model_name)
        ]
        ax.plot(
            subset["discharge_index"], subset["predicted_soh_pct"],
            label=model_name, linewidth=1.2
        )
    ax.set_title(f"Held-out {cell_id}")
    ax.set_xlabel("Discharge observation index")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Relative capacity retention (%)")
axes[-1].legend()
fig.tight_layout()
fig.savefig(OUT / "quadratic_model_predictions.png", dpi=180)
plt.close(fig)

print(f"\nResults saved to: {OUT}")
print("Each cell is held out completely during model fitting.")
