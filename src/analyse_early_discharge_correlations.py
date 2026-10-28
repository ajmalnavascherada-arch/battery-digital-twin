
"""Analyse early-discharge features against B0005 capacity retention."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "B0005_early_discharge_features.csv"
)
OUT_DIR = DATA.parent

df = pd.read_csv(DATA)

if len(df) != 168:
    raise ValueError(f"Expected 168 records, found {len(df)}")

if df["discharge_index"].nunique() != 168:
    raise ValueError("Discharge indices are not unique.")

target = "relative_capacity_retention_pct"

# Candidate predictors use only the specified early windows.
# Exclude capacity, cycle index, and full-discharge duration.
feature_columns = [
    col for col in df.columns
    if col.startswith(("first_10pct_", "first_20pct_", "first_30pct_"))
]

results = []

for feature in feature_columns:
    valid = df[[feature, target]].dropna()

    if len(valid) < 3 or valid[feature].nunique() < 2:
        continue

    rho = valid[feature].corr(valid[target], method="spearman")

    results.append({
        "feature": feature,
        "spearman_rho": rho,
        "absolute_rho": abs(rho),
        "valid_records": len(valid),
        "missing_values": int(df[feature].isna().sum()),
    })

result_df = pd.DataFrame(results).sort_values(
    "absolute_rho", ascending=False
)

output = OUT_DIR / "B0005_early_feature_correlations.csv"
result_df.to_csv(output, index=False)

print("Records:", len(df))
print("Early-window features tested:", len(feature_columns))
print("Features with valid correlations:", len(result_df))
print("Target:", target)
print("Saved:", output)

print("\nTop 20 early features by absolute Spearman correlation:")
print(
    result_df[
        ["feature", "spearman_rho", "valid_records", "missing_values"]
    ].head(20).round(4).to_string(index=False)
)

print("\nStrongest features by window:")
for pct in (10, 20, 30):
    subset = result_df[
        result_df["feature"].str.startswith(f"first_{pct}pct_")
    ]
    print(f"\nFirst {pct}%:")
    print(
        subset[["feature", "spearman_rho"]]
        .head(5)
        .round(4)
        .to_string(index=False)
    )
