
"""Audit cycle-summary features across NASA cells without modifying source data."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    "B0005": ROOT / "data" / "processed_nasa_original" / "B0005_cycle_summary.csv",
    "B0006": ROOT / "data" / "processed_B0006" / "B0006_cycle_summary.csv",
    "B0007": ROOT / "data" / "processed_B0007" / "B0007_cycle_summary.csv",
}

OUT = ROOT / "results_capacity_comparison" / "feature_audit"
OUT.mkdir(parents=True, exist_ok=True)

all_rows = []

for cell_id, path in SOURCES.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary file: {path}")

    df = pd.read_csv(path)
    print(f"\n{'=' * 72}")
    print(f"{cell_id}: {path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "cycle_type" in df.columns:
        print("\nCycle counts:")
        print(df["cycle_type"].value_counts(dropna=False).to_string())

    for col in df.columns:
        series = df[col]
        row = {
            "cell_id": cell_id,
            "column": col,
            "dtype": str(series.dtype),
            "rows": len(series),
            "missing_count": int(series.isna().sum()),
            "missing_pct": 100.0 * series.isna().mean(),
            "unique_nonmissing": int(series.nunique(dropna=True)),
        }
        all_rows.append(row)

        print(
            f"{col}: dtype={row['dtype']}, "
            f"missing={row['missing_count']} ({row['missing_pct']:.2f}%), "
            f"unique={row['unique_nonmissing']}"
        )

    if "ambient_temperature_C" in df.columns:
        print("\nAmbient-temperature values:")
        print(
            df["ambient_temperature_C"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

audit = pd.DataFrame(all_rows)
audit_path = OUT / "cycle_summary_feature_audit.csv"
audit.to_csv(audit_path, index=False)

print(f"\nFeature audit saved: {audit_path}")
print(
    "\nImportant: this script describes available fields and missingness. "
    "It does not select predictive features or modify any data."
)
