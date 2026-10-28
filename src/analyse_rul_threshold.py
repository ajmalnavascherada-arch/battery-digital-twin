
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CELL_FILES = {
    "B0005": ROOT / "data/processed_nasa_original/B0005_cycle_summary.csv",
    "B0006": ROOT / "data/processed_B0006/B0006_cycle_summary.csv",
    "B0007": ROOT / "data/processed_B0007/B0007_cycle_summary.csv",
}

EOL_THRESHOLD_PCT = 80.0

OUTPUT_DIR = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
)


def load_discharge_capacity(cell_id, file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"{cell_id} file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required = {"cycle_type", "capacity_Ah"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{cell_id}: missing columns {sorted(missing)}"
        )

    df = df.loc[
        df["cycle_type"].astype(str).str.lower().eq("discharge")
    ].copy()

    df["capacity_Ah"] = pd.to_numeric(
        df["capacity_Ah"], errors="coerce"
    )

    df = df.loc[
        np.isfinite(df["capacity_Ah"])
        & (df["capacity_Ah"] > 0)
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{cell_id}: no valid discharge capacities.")

    # Normalize capacity to the first valid discharge observation.
    initial_capacity = df["capacity_Ah"].iloc[0]
    df["discharge_index"] = np.arange(1, len(df) + 1)
    df["capacity_retention_pct"] = (
        100.0 * df["capacity_Ah"] / initial_capacity
    )

    df["cell_id"] = cell_id
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data = []
    summary_rows = []

    for cell_id, file_path in CELL_FILES.items():
        df = load_discharge_capacity(cell_id, file_path)
        all_data.append(df)

        below_threshold = df.loc[
            df["capacity_retention_pct"] <= EOL_THRESHOLD_PCT
        ]

        first_below = (
            int(below_threshold["discharge_index"].iloc[0])
            if not below_threshold.empty
            else None
        )

        summary_rows.append({
            "cell_id": cell_id,
            "valid_discharge_observations": len(df),
            "initial_capacity_Ah": df["capacity_Ah"].iloc[0],
            "final_capacity_Ah": df["capacity_Ah"].iloc[-1],
            "final_retention_pct": df["capacity_retention_pct"].iloc[-1],
            "minimum_retention_pct": df["capacity_retention_pct"].min(),
            "eol_threshold_pct": EOL_THRESHOLD_PCT,
            "first_observation_at_or_below_threshold": first_below,
            "threshold_reached": first_below is not None,
        })

    combined = pd.concat(all_data, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    combined_path = OUTPUT_DIR / "capacity_retention_trajectories.csv"
    summary_path = OUTPUT_DIR / "rul_threshold_summary.csv"

    combined.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("RUL THRESHOLD ANALYSIS")
    print(f"EOL threshold: {EOL_THRESHOLD_PCT:.1f}% retention\n")
    print(summary.round(4).to_string(index=False))

    print(f"\nSaved trajectories: {combined_path}")
    print(f"Saved threshold summary: {summary_path}")

    print(
        "\nNote: this identifies observed threshold crossings only. "
        "It does not extrapolate RUL for cells that remain above 80%."
    )


if __name__ == "__main__":
    main()
