
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed_nasa_original" / "B0005_signals_long.csv"
OUT = ROOT / "results_nasa_original" / "post_discharge_cooling_validation"
OUT.mkdir(parents=True, exist_ok=True)

AMBIENT_C = 24.0
ACTIVE_CURRENT_A = 0.5
NEAR_ZERO_CURRENT_A = 0.05

REFERENCE_CYCLE = 2
TEST_CYCLES = [4, 6, 10, 20]


def load_cycle(cycle_number):
    df = pd.read_csv(DATA)
    df = df[
        (df["cycle_number"] == cycle_number)
        & (df["cycle_type"] == "discharge")
    ].copy()

    if df.empty:
        raise ValueError(f"No discharge rows found for cycle {cycle_number}.")

    wide = df.pivot_table(
        index=["sample_index", "time_s"],
        columns="signal",
        values="value",
        aggfunc="first",
    ).reset_index()

    required = [
        "Temperature_measured",
        "Current_measured",
    ]
    missing = [col for col in required if col not in wide.columns]
    if missing:
        raise ValueError(f"Missing signals for cycle {cycle_number}: {missing}")

    wide = wide.sort_values("time_s").reset_index(drop=True)
    wide["discharge_current_a"] = (-wide["Current_measured"]).clip(lower=0)
    return wide


def get_post_discharge(df):
    active = df["discharge_current_a"] >= ACTIVE_CURRENT_A
    if not active.any():
        return None

    last_active = np.flatnonzero(active.to_numpy())[-1]
    post = df.iloc[last_active + 1:].copy()

    # Keep only near-zero-current samples after the final active sample.
    post = post[
        post["discharge_current_a"] < NEAR_ZERO_CURRENT_A
    ].copy()

    if len(post) < 5:
        return None

    t_end = float(df.loc[last_active, "time_s"])
    t = post["time_s"].to_numpy(dtype=float) - t_end
    temp = post["Temperature_measured"].to_numpy(dtype=float)
    t_start = float(df.loc[last_active, "Temperature_measured"])

    return t, temp, t_start, post


def fit_tau(t, temp, t_start):
    # Fit cycle 2 only. Keep ambient and starting temperature fixed.
    tau_grid = np.geomspace(1.0, 100000.0, 30000)
    errors = []

    for tau in tau_grid:
        pred = AMBIENT_C + (t_start - AMBIENT_C) * np.exp(-t / tau)
        errors.append(np.mean((temp - pred) ** 2))

    return float(tau_grid[int(np.argmin(errors))])


def metrics(t, temp, t_start, tau):
    pred = AMBIENT_C + (t_start - AMBIENT_C) * np.exp(-t / tau)
    residual = temp - pred
    return {
        "samples": len(temp),
        "duration_s": float(t[-1] - t[0]),
        "measured_final_c": float(temp[-1]),
        "predicted_final_c": float(pred[-1]),
        "rmse_c": float(np.sqrt(np.mean(residual ** 2))),
        "mae_c": float(np.mean(np.abs(residual))),
        "mean_bias_c": float(np.mean(residual)),
    }, pred


reference = get_post_discharge(load_cycle(REFERENCE_CYCLE))
if reference is None:
    raise ValueError("Insufficient post-discharge samples in reference cycle.")

t_ref, temp_ref, t_start_ref, _ = reference
tau_fixed = fit_tau(t_ref, temp_ref, t_start_ref)

print(f"Reference cycle: {REFERENCE_CYCLE}")
print(f"Frozen cooling time constant: {tau_fixed:.2f} s "
      f"({tau_fixed / 60:.2f} min)")

rows = []
plot_data = []

for cycle in [REFERENCE_CYCLE] + TEST_CYCLES:
    result = get_post_discharge(load_cycle(cycle))

    if result is None:
        print(f"Skipping cycle {cycle}: insufficient cooling samples.")
        continue

    t, temp, t_start, post = result
    score, pred = metrics(t, temp, t_start, tau_fixed)

    rows.append({
        "cycle_number": cycle,
        "role": "reference_in_sample" if cycle == REFERENCE_CYCLE
                else "independent_cycle_test",
        "fixed_tau_s": tau_fixed,
        "start_temperature_c": t_start,
        **score,
    })

    plot_data.append((cycle, t, temp, pred))

results = pd.DataFrame(rows)
results.to_csv(OUT / "cooling_cross_cycle_metrics.csv", index=False)

print("\nCross-cycle results:")
print(results.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
for cycle, t, temp, pred in plot_data:
    ax.plot(t, temp, marker="o", markersize=3, label=f"Cycle {cycle} measured")
    ax.plot(t, pred, linestyle="--", label=f"Cycle {cycle} predicted")

ax.set_xlabel("Time after final active-discharge sample (s)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Fixed-parameter post-discharge cooling validation")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()

plot_path = OUT / "cooling_cross_cycle_validation.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved metrics and plot to: {OUT}")
