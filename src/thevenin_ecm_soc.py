
"""Estimate battery SOC by coulomb counting for a NASA B0005 discharge cycle.

Convention:
- NASA discharge current is negative.
- This script converts discharge current to positive.
- Initial SOC is an explicit assumption, not a measured value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate SOC using coulomb counting."
    )
    parser.add_argument(
        "--signals",
        default="data/processed_nasa_original/B0005_signals_long.csv",
    )
    parser.add_argument("--cycle", type=int, default=2)
    parser.add_argument(
        "--capacity-ah",
        type=float,
        default=1.856487,
        help="Usable capacity in Ah; default is the recorded cycle-2 capacity.",
    )
    parser.add_argument(
        "--initial-soc",
        type=float,
        default=1.0,
        help="Assumed SOC at the first sample, between 0 and 1.",
    )
    parser.add_argument(
        "--out-dir",
        default="results_nasa_original/ecm_soc",
    )
    args = parser.parse_args()

    if args.capacity_ah <= 0:
        raise ValueError("--capacity-ah must be positive.")
    if not 0.0 <= args.initial_soc <= 1.0:
        raise ValueError("--initial-soc must be between 0 and 1.")

    input_path = Path(args.signals)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.loc[
        (df["cycle_number"] == args.cycle)
        & (df["cycle_type"] == "discharge")
        & (df["signal"].isin(["Voltage_measured", "Current_measured"]))
    ].copy()

    if df.empty:
        raise ValueError(
            f"No discharge voltage/current samples found for cycle {args.cycle}."
        )

    wide = df.pivot(
        index="sample_index",
        columns="signal",
        values="value",
    ).sort_index()

    required = ["Voltage_measured", "Current_measured"]
    if any(col not in wide.columns for col in required):
        raise ValueError("Voltage or current signal is missing.")

    # Retain only rows with both voltage and current.
    wide = wide.dropna(subset=required)

    time_df = (
        df.loc[df["signal"] == "Voltage_measured",
               ["sample_index", "time_s", "time_aligned"]]
        .drop_duplicates("sample_index")
        .set_index("sample_index")
        .sort_index()
    )
    wide = wide.join(time_df, how="inner").dropna(subset=["time_s"])

    aligned = (
        wide["time_aligned"].astype(str).str.lower().isin(["true", "1"])
    )
    if not aligned.all():
        raise ValueError("Some selected samples do not have aligned timestamps.")

    time_s = wide["time_s"].to_numpy(dtype=float)
    voltage_v = wide["Voltage_measured"].to_numpy(dtype=float)

    # NASA records discharge current as negative.
    current_a = -wide["Current_measured"].to_numpy(dtype=float)

    if len(time_s) < 2:
        raise ValueError("At least two aligned samples are required.")
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(current_a)):
        raise ValueError("Time/current contains non-finite values.")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Timestamps must be strictly increasing.")

    # Trapezoidal integration of current: Ah = integral(I dt) / 3600.
    dt_s = np.diff(time_s)
    delta_ah = 0.5 * (current_a[:-1] + current_a[1:]) * dt_s / 3600.0
    removed_ah = np.concatenate(([0.0], np.cumsum(delta_ah)))

    soc_raw = args.initial_soc - removed_ah / args.capacity_ah
    soc_clipped = np.clip(soc_raw, 0.0, 1.0)

    result = pd.DataFrame(
        {
            "sample_index": wide.index.to_numpy(),
            "time_s": time_s,
            "voltage_v": voltage_v,
            "current_discharge_positive_a": current_a,
            "cumulative_removed_ah": removed_ah,
            "soc_raw": soc_raw,
            "soc_estimated": soc_clipped,
        }
    )

    output_csv = output_dir / f"B0005_cycle_{args.cycle}_soc.csv"
    output_png = output_dir / f"B0005_cycle_{args.cycle}_soc.png"
    result.to_csv(output_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(time_s, soc_clipped * 100, label="Coulomb-counted SOC")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Estimated SOC (%)")
    ax.set_title(f"B0005 recorded cycle {args.cycle}: SOC estimate")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)

    print("Coulomb-counting SOC estimate")
    print("=============================")
    print(f"Cycle: {args.cycle}")
    print(f"Samples: {len(result)}")
    print(f"Assumed initial SOC: {args.initial_soc:.1%}")
    print(f"Capacity used: {args.capacity_ah:.6f} Ah")
    print(f"Integrated discharge: {removed_ah[-1]:.6f} Ah")
    print(f"Raw final SOC: {soc_raw[-1]:.2%}")
    print(f"Clipped final SOC: {soc_clipped[-1]:.2%}")
    print("Note: initial SOC and capacity are assumptions/approximations.")
    print(f"CSV: {output_csv}")
    print(f"Plot: {output_png}")


if __name__ == "__main__":
    main()
