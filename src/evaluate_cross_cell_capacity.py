
"""Evaluate capacity-retention models on held-out NASA battery cells.

Uses cycle-level capacity summaries only. Does not use raw voltage,
current, or temperature signals.
"""
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

OUT_DIR = ROOT / "results_capacity_comparison" / "cross_cell_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cell(cell_id, file_path):
    df = pd.read_csv(file_path)

    required = {"cycle_type", "cycle_number", "capacity_Ah"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_path} is missing columns: {sorted(missing)}")

    discharge = df.loc[
        df["cycle_type"].astype(str).str.lower().eq("discharge")
    ].copy()

    discharge["capacity_Ah"] = pd.to_numeric(
        discharge["capacity_Ah"], errors="coerce"
    )
    discharge["cycle_number"] = pd.to_numeric(
        discharge["cycle_number"], errors="coerce"
    )

    discharge = discharge.dropna(
        subset=["cycle_number", "capacity_Ah"]
    )
    discharge = discharge.loc[
        discharge["capacity_Ah"] > 0
    ].sort_values("cycle_number").reset_index(drop=True)

    if len(discharge) < 5:
        raise ValueError(
            f"Too few valid discharge capacity records for {cell_id}: "
            f"{len(discharge)}"
        )

    # Relative retention uses this cell's own first valid discharge capacity.
    initial_capacity = discharge["capacity_Ah"].iloc[0]
    discharge["retention_pct"] = (
        100.0 * discharge["capacity_Ah"] / initial_capacity
    )

    # Position within the ordered discharge observations, not physical time.
    discharge["observation_index"] = range(1, len(discharge) + 1)
    discharge["cell_id"] = cell_id

    return discharge[
        ["cell_id", "cycle_number", "observation_index",
         "capacity_Ah", "retention_pct"]
    ]


cells = {
    cell_id: load_cell(cell_id, path)
    for cell_id, path in FILES.items()
}

print("VALID DISCHARGE RECORDS")
for cell_id, data in cells.items():
    print(
        f"{cell_id}: n={len(data)}, "
        f"initial capacity={data['capacity_Ah'].iloc[0]:.6f} Ah, "
        f"final retention={data['retention_pct'].iloc[-1]:.2f}%"
    )

# Save normalized cycle-level dataset for traceability.
combined = pd.concat(cells.values(), ignore_index=True)
combined.to_csv(OUT_DIR / "normalized_capacity_records.csv", index=False)

results = []

for held_out_cell in cells:
    train_cells = [
        cell_id for cell_id in cells if cell_id != held_out_cell
    ]

    train = pd.concat(
        [cells[cell_id] for cell_id in train_cells],
        ignore_index=True,
    )
    test = cells[held_out_cell].copy()

    X_train = train[["observation_index"]]
    y_train = train["retention_pct"]
    X_test = test[["observation_index"]]
    y_test = test["retention_pct"]

    # Baseline: pooled mean retention from the two training cells.
    baseline_pred = [y_train.mean()] * len(test)

    # Trend: one pooled line fitted to observation index across training cells.
    model = LinearRegression()
    model.fit(X_train, y_train)
    trend_pred = model.predict(X_test)

    for model_name, predicted in [
        ("training_cells_mean", baseline_pred),
        ("pooled_linear_trend", trend_pred),
    ]:
        results.append({
            "held_out_cell": held_out_cell,
            "training_cells": ",".join(train_cells),
            "model": model_name,
            "train_rows": len(train),
            "test_rows": len(test),
            "MAE_pp": mean_absolute_error(y_test, predicted),
            "RMSE_pp": mean_squared_error(y_test, predicted) ** 0.5,
            "R2": r2_score(y_test, predicted),
            "mean_residual_pp": (
                y_test.to_numpy() - predicted
            ).mean(),
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "metrics.csv", index=False)

print("\nCROSS-CELL RESULTS")
print(results_df.round(4).to_string(index=False))
print("\nSaved results to:", OUT_DIR)
