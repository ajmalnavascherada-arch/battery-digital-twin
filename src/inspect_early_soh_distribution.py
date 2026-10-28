
"""Inspect chronological train/test feature and target distributions."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "B0005_fixed_time_features.csv"
)

OUT_DIR = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "early_soh_baseline"
    / "diagnostics"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURE_FILE).sort_values("discharge_index").reset_index(drop=True)

target = "relative_capacity_retention_pct"
feature_cols = [
    "first_300s_voltage_drop_V",
    "first_600s_voltage_drop_V",
    "first_300s_temperature_rise_C",
    "first_600s_temperature_rise_C",
]

split = int(len(df) * 0.70)
df["period"] = ["train"] * split + ["test"] * (len(df) - split)

print("Rows:", len(df))
print("Training rows:", split)
print("Test rows:", len(df) - split)
print("Training indices:",
      df.iloc[0]["discharge_index"], "to",
      df.iloc[split - 1]["discharge_index"])
print("Test indices:",
      df.iloc[split]["discharge_index"], "to",
      df.iloc[-1]["discharge_index"])

columns = [target] + feature_cols

summary = df.groupby("period")[columns].agg(
    ["min", "median", "max", "mean", "std"]
)

print("\nTRAIN/TEST DISTRIBUTION SUMMARY")
print(summary.round(4).to_string())

summary.to_csv(OUT_DIR / "train_test_distribution_summary.csv")

print("\nFEATURE RANGE OVERLAP")
for col in columns:
    train = df.loc[df["period"] == "train", col].dropna()
    test = df.loc[df["period"] == "test", col].dropna()

    overlap = max(
        0.0,
        min(train.max(), test.max()) - max(train.min(), test.min())
    )
    print(
        f"{col}: "
        f"train=[{train.min():.4f}, {train.max():.4f}], "
        f"test=[{test.min():.4f}, {test.max():.4f}], "
        f"range_overlap={overlap:.4f}"
    )

print("\nSaved summary to:", OUT_DIR / "train_test_distribution_summary.csv")
