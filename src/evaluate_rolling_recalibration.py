
"""Evaluate one-step-ahead capacity prediction with rolling recalibration."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "B0005": ROOT / "data" / "processed_nasa_original" / "B0005_cycle_summary.csv",
    "B0006": ROOT / "data" / "processed_B0006" / "B0006_cycle_summary.csv",
    "B0007": ROOT / "data" / "processed_B0007" / "B0007_cycle_summary.csv",
}

INITIAL_FRACTION = 0.20

OUT_DIR = (
    ROOT / "results_capacity_comparison"
    / "rolling_recalibration"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cell(cell_id, path):
    df = pd.read_csv(path)

    df = df.loc[
        df["cycle_type"].astype(str).str.lower().eq("discharge")
    ].copy()

    df["cycle_number"] = pd.to_numeric(
        df["cycle_number"], errors="coerce"
    )
    df["capacity_Ah"] = pd.to_numeric(
        df["capacity_Ah"], errors="coerce"
    )

    df = df.dropna(subset=["cycle_number", "capacity_Ah"])
    df = df.loc[df["capacity_Ah"] > 0]
    df = df.sort_values("cycle_number").reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Too few valid discharge observations for {cell_id}")

    initial_capacity = df["capacity_Ah"].iloc[0]
    df["retention_pct"] = 100 * df["capacity_Ah"] / initial_capacity
    df["observation_index"] = range(1, len(df) + 1)
    df["cell_id"] = cell_id

    return df[
        ["cell_id", "cycle_number", "observation_index",
         "capacity_Ah", "retention_pct"]
    ]


all_metrics = []
all_predictions = []

for cell_id, path in FILES.items():
    df = load_cell(cell_id, path)

    initial_count = max(3, int(len(df) * INITIAL_FRACTION))
    if initial_count >= len(df) - 1:
        raise ValueError(f"Not enough evaluation observations for {cell_id}")

    rows = []

    # Predict one observation at a time.
    # At each step, fit only on observations preceding the target.
    for target_position in range(initial_count, len(df)):
        history = df.iloc[:target_position]
        target = df.iloc[target_position]

        X_history = history[["observation_index"]]
        y_history = history["retention_pct"]
        X_target = pd.DataFrame({
            "observation_index": [target["observation_index"]]
        })

        model = LinearRegression()
        model.fit(X_history, y_history)
        trend_prediction = float(model.predict(X_target)[0])

        # Persistence predicts the most recently observed retention.
        persistence_prediction = float(y_history.iloc[-1])

        for model_name, predicted in [
            ("rolling_linear_trend", trend_prediction),
            ("persistence", persistence_prediction),
        ]:
            rows.append({
                "cell_id": cell_id,
                "model": model_name,
                "observation_index": int(target["observation_index"]),
                "cycle_number": target["cycle_number"],
                "actual_retention_pct": float(target["retention_pct"]),
                "predicted_retention_pct": predicted,
                "residual_pp": float(target["retention_pct"] - predicted),
            })

    predictions = pd.DataFrame(rows)
    all_predictions.append(predictions)

    for model_name, group in predictions.groupby("model"):
        actual = group["actual_retention_pct"]
        predicted = group["predicted_retention_pct"]

        all_metrics.append({
            "cell_id": cell_id,
            "model": model_name,
            "initial_calibration_observations": initial_count,
            "evaluated_observations": len(group),
            "MAE_pp": mean_absolute_error(actual, predicted),
            "RMSE_pp": mean_squared_error(actual, predicted) ** 0.5,
            "R2": r2_score(actual, predicted),
            "mean_residual_pp": group["residual_pp"].mean(),
        })

    print(
        f"{cell_id}: total={len(df)}, "
        f"initial calibration={initial_count}, "
        f"one-step predictions={len(df) - initial_count}"
    )

metrics_df = pd.DataFrame(all_metrics)
predictions_df = pd.concat(all_predictions, ignore_index=True)

metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
predictions_df.to_csv(OUT_DIR / "predictions.csv", index=False)

print("\nROLLING RECALIBRATION RESULTS")
print(metrics_df.round(4).to_string(index=False))

print("\nSaved metrics to:", OUT_DIR / "metrics.csv")
print("Saved predictions to:", OUT_DIR / "predictions.csv")
