
"""Evaluate cell-specific capacity-trend calibration on NASA battery data.

For each held-out cell:
- Use its first 20% of valid discharge observations for calibration.
- Predict the remaining 80% without using their capacity values for fitting.
- Compare linear-trend extrapolation against persistence.
- Report metrics separately for each cell.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "B0005": ROOT / "data" / "processed_nasa_original" / "B0005_cycle_summary.csv",
    "B0006": ROOT / "data" / "processed_B0006" / "B0006_cycle_summary.csv",
    "B0007": ROOT / "data" / "processed_B0007" / "B0007_cycle_summary.csv",
}

CALIBRATION_FRACTION = 0.20

OUT_DIR = (
    ROOT / "results_capacity_comparison"
    / "calibrated_cross_cell_validation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cell(cell_id, file_path):
    df = pd.read_csv(file_path)

    required = {"cycle_type", "cycle_number", "capacity_Ah"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{file_path} is missing columns: {sorted(missing)}"
        )

    discharge = df.loc[
        df["cycle_type"].astype(str).str.lower().eq("discharge")
    ].copy()

    discharge["cycle_number"] = pd.to_numeric(
        discharge["cycle_number"], errors="coerce"
    )
    discharge["capacity_Ah"] = pd.to_numeric(
        discharge["capacity_Ah"], errors="coerce"
    )

    discharge = discharge.dropna(
        subset=["cycle_number", "capacity_Ah"]
    )
    discharge = discharge.loc[
        discharge["capacity_Ah"] > 0
    ]
    discharge = (
        discharge.sort_values("cycle_number")
        .reset_index(drop=True)
    )

    if len(discharge) < 10:
        raise ValueError(
            f"Too few valid discharge observations for {cell_id}: "
            f"{len(discharge)}"
        )

    initial_capacity = discharge["capacity_Ah"].iloc[0]
    discharge["retention_pct"] = (
        100.0 * discharge["capacity_Ah"] / initial_capacity
    )
    discharge["observation_index"] = range(1, len(discharge) + 1)
    discharge["cell_id"] = cell_id

    return discharge[
        [
            "cell_id",
            "cycle_number",
            "observation_index",
            "capacity_Ah",
            "retention_pct",
        ]
    ]


cells = {
    cell_id: load_cell(cell_id, path)
    for cell_id, path in FILES.items()
}

results = []
all_predictions = []

for cell_id, data in cells.items():
    calibration_count = int(len(data) * CALIBRATION_FRACTION)

    # Ensure enough observations to fit a linear model.
    calibration_count = max(3, calibration_count)

    if calibration_count >= len(data):
        raise ValueError(
            f"Calibration segment leaves no test data for {cell_id}."
        )

    calibration = data.iloc[:calibration_count].copy()
    test = data.iloc[calibration_count:].copy()

    X_cal = calibration[["observation_index"]]
    y_cal = calibration["retention_pct"]
    X_test = test[["observation_index"]]
    y_test = test["retention_pct"]

    # Fit only on the calibration segment of this cell.
    model = LinearRegression()
    model.fit(X_cal, y_cal)
    trend_predictions = model.predict(X_test)

    # Persistence baseline: hold retention at the last calibration value.
    persistence_predictions = [
        float(y_cal.iloc[-1])
    ] * len(test)

    candidates = {
        "calibrated_linear_trend": trend_predictions,
        "calibration_persistence": persistence_predictions,
    }

    for model_name, predicted in candidates.items():
        results.append({
            "cell_id": cell_id,
            "model": model_name,
            "total_observations": len(data),
            "calibration_observations": len(calibration),
            "test_observations": len(test),
            "calibration_end_index": int(
                calibration["observation_index"].iloc[-1]
            ),
            "test_start_index": int(
                test["observation_index"].iloc[0]
            ),
            "test_end_index": int(
                test["observation_index"].iloc[-1]
            ),
            "MAE_pp": mean_absolute_error(y_test, predicted),
            "RMSE_pp": mean_squared_error(y_test, predicted) ** 0.5,
            "R2": r2_score(y_test, predicted),
            "mean_residual_pp": (
                y_test.to_numpy() - predicted
            ).mean(),
        })

        for index, actual, estimate in zip(
            test["observation_index"],
            y_test,
            predicted,
        ):
            all_predictions.append({
                "cell_id": cell_id,
                "model": model_name,
                "observation_index": int(index),
                "actual_retention_pct": float(actual),
                "predicted_retention_pct": float(estimate),
                "residual_pp": float(actual - estimate),
            })

    print(
        f"{cell_id}: total={len(data)}, "
        f"calibration={len(calibration)}, test={len(test)}, "
        f"calibration_end_retention={y_cal.iloc[-1]:.2f}%"
    )

results_df = pd.DataFrame(results)
predictions_df = pd.DataFrame(all_predictions)

results_df.to_csv(OUT_DIR / "metrics.csv", index=False)
predictions_df.to_csv(OUT_DIR / "predictions.csv", index=False)

print("\nCALIBRATED CROSS-CELL RESULTS")
print(
    results_df[
        [
            "cell_id",
            "model",
            "calibration_observations",
            "test_observations",
            "MAE_pp",
            "RMSE_pp",
            "R2",
            "mean_residual_pp",
        ]
    ].round(4).to_string(index=False)
)

# Plot actual and predicted retention for each cell.
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

for ax, cell_id in zip(axes, cells):
    cell_data = cells[cell_id]
    cell_predictions = predictions_df.loc[
        predictions_df["cell_id"] == cell_id
    ]

    calibration_count = int(
        len(cell_data) * CALIBRATION_FRACTION
    )
    calibration_count = max(3, calibration_count)

    ax.plot(
        cell_data["observation_index"],
        cell_data["retention_pct"],
        label="Actual retention",
        marker=".",
        markersize=3,
    )

    for model_name, group in cell_predictions.groupby("model"):
        ax.plot(
            group["observation_index"],
            group["predicted_retention_pct"],
            label=model_name,
        )

    ax.axvline(
        calibration_count,
        linestyle="--",
        label="Calibration/test boundary",
    )
    ax.set_title(cell_id)
    ax.set_xlabel("Discharge observation index")
    ax.set_ylabel("Capacity retention (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.tight_layout()
plot_path = OUT_DIR / "calibrated_predictions.png"
fig.savefig(plot_path, dpi=160)
plt.close(fig)

print("\nSaved metrics and predictions to:", OUT_DIR)
print("Saved plot to:", plot_path)
