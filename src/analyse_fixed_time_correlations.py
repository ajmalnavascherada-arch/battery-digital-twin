
"""Rank fixed-time early-discharge features against B0005 capacity retention."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "B0005_fixed_time_features.csv"
)
OUT_DIR = DATA.parent

df = pd.read_csv(DATA)

if len(df) != 168:
    raise ValueError(f"Expected 168 records, found {len(df)}")

if df["discharge_index"].nunique() != 168:
    raise ValueError("Duplicate discharge indices detected.")

target = "relative_capacity_retention_pct"

# Restrict predictors to features computed within the fixed-time windows.
feature_columns = [
    col for col in df.columns
    if col.startswith(("first_300s_", "first_600s_", "first_900s_"))
    and not col.endswith(("sample_count", "last_sample_time_s"))
]

results = []

for feature in feature_columns:
    valid = df[[feature, target]].dropna()

    if len(valid) < 3 or valid[feature].nunique() < 2:
        continue

    results.append({
        "feature": feature,
        "spearman_rho": valid[feature].corr(
            valid[target], method="spearman"
        ),
        "absolute_rho": abs(
            valid[feature].corr(valid[target], method="spearman")
        ),
        "valid_records": len(valid),
        "missing_values": int(df[feature].isna().sum()),
    })

result_df = pd.DataFrame(results).sort_values(
    "absolute_rho", ascending=False
)

output = OUT_DIR / "B0005_fixed_time_correlations.csv"
result_df.to_csv(output, index=False)

print("Records:", len(df))
print("Candidate features:", len(feature_columns))
print("Features ranked:", len(result_df))
print("Saved:", output)

print("\nTop 15 features by absolute Spearman correlation:")
print(
    result_df[
        ["feature", "spearman_rho", "valid_records", "missing_values"]
    ].head(15).round(4).to_string(index=False)
)

for window in (300, 600, 900):
    subset = result_df[
        result_df["feature"].str.startswith(f"first_{window}s_")
    ]
    print(f"\nTop features at {window} seconds:")
    print(
        subset[["feature", "spearman_rho"]]
        .head(5).round(4).to_string(index=False)
    )
