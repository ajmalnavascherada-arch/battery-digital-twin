
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed_nasa_original"
    / "B0005_signals_long.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "thermal_diagnostic"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Convert long-format signals into one row per sample.
signals = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
]

wide = (
    df[df["signal"].isin(signals)]
    .pivot(
        index=["cell_id", "cycle_number", "cycle_type",
               "sample_index", "time_s"],
        columns="signal",
        values="value",
    )
    .reset_index()
)

required = set(signals)
missing = required - set(wide.columns)
if missing:
    raise ValueError(f"Missing required signals: {sorted(missing)}")

wide = wide.sort_values(
    ["cycle_number", "sample_index"]
).reset_index(drop=True)

# NASA records discharge current as negative.
wide["discharge_current_a"] = (
    -wide["Current_measured"]
).clip(lower=0)

wide["power_w"] = (
    wide["Voltage_measured"]
    * wide["discharge_current_a"]
)

# Focus on discharge cycles.
discharge = wide[wide["cycle_type"] == "discharge"].copy()

# Summarise each discharge cycle.
rows = []

for cycle_number, group in discharge.groupby("cycle_number"):
    group = group.sort_values("time_s")
    active = group[group["discharge_current_a"] >= 0.5]

    if active.empty:
        continue

    rows.append({
        "cycle_number": cycle_number,
        "samples": len(group),
        "active_samples": len(active),
        "duration_s": (
            active["time_s"].max() - active["time_s"].min()
        ),
        "temperature_start_c": active["Temperature_measured"].iloc[0],
        "temperature_end_c": active["Temperature_measured"].iloc[-1],
        "temperature_min_c": active["Temperature_measured"].min(),
        "temperature_max_c": active["Temperature_measured"].max(),
        "temperature_rise_c": (
            active["Temperature_measured"].iloc[-1]
            - active["Temperature_measured"].iloc[0]
        ),
        "mean_current_a": active["discharge_current_a"].mean(),
        "mean_voltage_v": active["Voltage_measured"].mean(),
        "mean_power_w": active["power_w"].mean(),
        "discharged_energy_wh": (
            (active["power_w"] *
             active["time_s"].diff().fillna(0) / 3600).sum()
        ),
    })

summary = pd.DataFrame(rows)

summary_path = OUTPUT_DIR / "B0005_discharge_thermal_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n=== DISCHARGE THERMAL SUMMARY ===")
print(f"Discharge cycles analysed: {len(summary)}")
print(f"Summary saved: {summary_path}")

if not summary.empty:
    print("\nFirst five cycles:")
    print(summary.head().round(4).to_string(index=False))

    print("\nAcross-cycle statistics:")
    print(
        summary[
            [
                "temperature_rise_c",
                "temperature_max_c",
                "mean_current_a",
                "mean_power_w",
            ]
        ].describe().round(4).to_string()
    )

    # Plot one representative discharge cycle.
    selected_cycle = 2
    cycle = discharge[
        discharge["cycle_number"] == selected_cycle
    ].sort_values("time_s")

    if cycle.empty:
        selected_cycle = int(summary.iloc[0]["cycle_number"])
        cycle = discharge[
            discharge["cycle_number"] == selected_cycle
        ].sort_values("time_s")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(
        cycle["time_s"], cycle["Temperature_measured"]
    )
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        cycle["time_s"], cycle["Voltage_measured"]
    )
    axes[1].set_ylabel("Voltage (V)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        cycle["time_s"], cycle["discharge_current_a"]
    )
    axes[2].set_ylabel("Discharge current (A)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"NASA B0005 — Discharge thermal response, cycle {selected_cycle}"
    )
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "B0005_cycle_thermal_response.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlot saved: {plot_path}")
else:
    print("No discharge cycles had sufficient active-current samples.")
