"""Audit signal lengths and timestamp alignment before electrical modelling.

Reads the Stage 1 long-format signal CSV and cycle summary. Does not interpolate,
truncate, or silently pair measurements from different sample counts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CORE_SIGNALS = ["Voltage_measured", "Current_measured", "Temperature_measured"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path,
                        default=Path("data/processed/B0005_signals_long.csv"))
    parser.add_argument("--cycles", type=Path,
                        default=Path("data/processed/B0005_cycle_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    for path in (args.signals, args.cycles):
        if not path.is_file():
            parser.error(f"Required file not found: {path}. Run Stage 1 first.")

    signals = pd.read_csv(args.signals)
    cycles = pd.read_csv(args.cycles)

    needed_signals = {"cycle_number", "cycle_type", "signal", "sample_index",
                      "time_s", "value", "time_aligned"}
    if not needed_signals.issubset(signals.columns):
        parser.error(f"Signal CSV must contain: {sorted(needed_signals)}")

    discharge = cycles.loc[
        cycles["cycle_type"].astype(str).str.lower().eq("discharge"),
        ["cycle_number", "capacity_Ah", "ambient_temperature_C", "time_samples",
         "length_mismatch", "mismatch_details"]
    ].copy()

    # Audit each cycle by comparing the counts of each core signal and whether
    # every core signal has timestamps aligned with its own sample indices.
    rows = []
    for _, cyc in discharge.iterrows():
        number = int(cyc["cycle_number"])
        one = signals.loc[signals["cycle_number"].eq(number)]
        signal_counts = {
            sig: int(one.loc[one["signal"].eq(sig)].shape[0])
            for sig in CORE_SIGNALS
        }
        aligned_flags = {
            sig: bool(
                one.loc[one["signal"].eq(sig), "time_aligned"].astype(str)
                .str.lower().isin(["true", "1"]).all()
            ) and signal_counts[sig] > 0
            for sig in CORE_SIGNALS
        }
        lengths_match = len(set(signal_counts.values())) == 1 and all(
            count > 0 for count in signal_counts.values()
        )
        all_aligned = all(aligned_flags.values())
        rows.append({
            "cycle_number": number,
            "capacity_Ah": cyc["capacity_Ah"],
            "ambient_temperature_C": cyc["ambient_temperature_C"],
            **{f"{sig}_samples": signal_counts[sig] for sig in CORE_SIGNALS},
            "core_signal_lengths_match": lengths_match,
            "all_core_signals_have_time_alignment": all_aligned,
            "candidate_for_time_based_ECM": lengths_match and all_aligned,
        })

    audit = pd.DataFrame(rows).sort_values("cycle_number")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "B0005_signal_alignment_audit.csv"
    report_path = args.out_dir / "B0005_signal_alignment_report.txt"
    plot_path = args.out_dir / "B0005_first_discharge_signals_by_sample.png"
    audit.to_csv(audit_path, index=False)

    candidate_count = int(audit["candidate_for_time_based_ECM"].sum())
    mismatch_count = int((~audit["core_signal_lengths_match"]).sum())
    no_time_count = int((~audit["all_core_signals_have_time_alignment"]).sum())

    # Plot the first discharge's signals separately against sample index.
    # Separate axes and x values avoid implying alignment when lengths differ.
    first_num = int(audit.iloc[0]["cycle_number"])
    first = signals.loc[signals["cycle_number"].eq(first_num)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    units = {
        "Voltage_measured": ("Voltage measured", "Voltage (V)"),
        "Current_measured": ("Current measured", "Current (A)"),
        "Temperature_measured": ("Temperature measured", "Temperature (°C)"),
    }
    for ax, sig in zip(axes, CORE_SIGNALS):
        part = first.loc[first["signal"].eq(sig)].sort_values("sample_index")
        ax.plot(part["sample_index"], part["value"], linewidth=1.0)
        ax.set_title(units[sig][0])
        ax.set_ylabel(units[sig][1])
        ax.set_xlabel("Sample index (not time)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"B0005 discharge cycle {first_num}: independent signal traces")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    report_lines = [
        "NASA B0005 — core signal alignment audit",
        "=" * 45,
        f"Discharge cycles inspected: {len(audit)}",
        f"Cycles with matching voltage/current/temperature sample counts: {len(audit) - mismatch_count}",
        f"Cycles with at least one core signal lacking time alignment: {no_time_count}",
        f"Candidate cycles for time-based ECM (matching lengths and aligned time): {candidate_count}",
        "",
        "Important:",
        "- This audit does not claim sample-index correspondence is physically valid.",
        "- No signal was interpolated or truncated.",
        "- Candidate cycles still require inspection of time monotonicity, units, current sign convention, and rest/charge/discharge segmentation.",
        "- If few or no candidate cycles exist, revisit the source dataset format before fitting an ECM.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nAudit CSV: {audit_path}")
    print(f"Signal plot: {plot_path}")
    print(f"Report: {report_path}")
    print("\nFirst 10 discharge audit rows:")
    print(audit.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
