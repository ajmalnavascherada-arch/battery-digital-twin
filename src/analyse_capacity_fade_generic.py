"""Analyse observed discharge-capacity fade for any processed NASA battery cell."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED = {"cycle_number", "cycle_type", "capacity_Ah", "ambient_temperature_C"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Input cycle-summary CSV")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--cell", required=True,
                        help="Cell label, e.g. B0006")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    missing = REQUIRED - set(df.columns)
    if missing:
        parser.error(f"Input is missing columns: {sorted(missing)}")

    dis = df.loc[
        df["cycle_type"].astype(str).str.lower().eq("discharge")
    ].copy()
    dis["cycle_number"] = pd.to_numeric(dis["cycle_number"], errors="coerce")
    dis["capacity_Ah"] = pd.to_numeric(dis["capacity_Ah"], errors="coerce")

    invalid = (
        dis["cycle_number"].isna()
        | ~np.isfinite(dis["cycle_number"])
        | dis["capacity_Ah"].isna()
        | ~np.isfinite(dis["capacity_Ah"])
        | (dis["capacity_Ah"] <= 0)
    )
    invalid_count = int(invalid.sum())
    valid = dis.loc[~invalid].sort_values("cycle_number").copy()

    if len(valid) < 2:
        parser.error("Need at least two valid discharge capacities.")

    duplicate_count = int(valid["cycle_number"].duplicated().sum())
    valid = valid.drop_duplicates(subset="cycle_number", keep="first")

    first = valid.iloc[0]
    last = valid.iloc[-1]
    first_capacity = float(first["capacity_Ah"])
    last_capacity = float(last["capacity_Ah"])
    change_ah = last_capacity - first_capacity
    change_pct = 100.0 * change_ah / first_capacity

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.cell.upper()
    csv_path = args.out_dir / f"{prefix}_discharge_capacity.csv"
    plot_path = args.out_dir / f"{prefix}_capacity_fade.png"
    report_path = args.out_dir / f"{prefix}_capacity_report.txt"

    valid[["cycle_number", "capacity_Ah", "ambient_temperature_C"]].to_csv(
        csv_path, index=False
    )

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(valid["cycle_number"], valid["capacity_Ah"],
            marker="o", markersize=3.0, linewidth=1.2)
    ax.set_title(f"NASA {prefix}: observed discharge capacity")
    ax.set_xlabel("Recorded cycle number")
    ax.set_ylabel("Discharge capacity (Ah)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    report = [
        f"NASA {prefix} — discharge capacity data-quality report",
        "=" * 60,
        f"Rows classified as discharge: {len(dis)}",
        f"Invalid capacity or cycle-number records: {invalid_count}",
        f"Duplicate cycle numbers among valid records: {duplicate_count}",
        f"Unique valid discharge capacity records analysed: {len(valid)}",
        f"First analysed recorded cycle: {int(first['cycle_number'])}",
        f"First analysed discharge capacity: {first_capacity:.6f} Ah",
        f"Last analysed recorded cycle: {int(last['cycle_number'])}",
        f"Last analysed discharge capacity: {last_capacity:.6f} Ah",
        f"Absolute capacity change: {change_ah:.6f} Ah",
        f"Relative change from first to last analysed point: {change_pct:.3f}%",
        "",
        "Interpretation limits:",
        "- This is an observed endpoint comparison, not a fitted degradation rate.",
        "- Recorded cycle numbers include charge, discharge and impedance records.",
        "- No end-of-life threshold or remaining-useful-life prediction is applied.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nCapacity CSV: {csv_path}")
    print(f"Capacity plot: {plot_path}")
    print(f"Text report: {report_path}")


if __name__ == "__main__":
    main()
