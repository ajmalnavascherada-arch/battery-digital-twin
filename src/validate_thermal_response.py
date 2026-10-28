
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed_nasa_original" / "B0005_signals_long.csv"
OUT = ROOT / "results_nasa_original" / "thermal_model"
OUT.mkdir(parents=True, exist_ok=True)

REFERENCE_CYCLE = 2
TEST_CYCLES = [4, 6, 10, 20]
MIN_CURRENT_A = 0.5

df = pd.read_csv(INPUT)

signals = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
]

wide = (
    df[df["signal"].isin(signals)]
    .pivot(
        index=["cycle_number", "cycle_type", "sample_index", "time_s"],
        columns="signal",
        values="value",
    )
    .reset_index()
)

def get_active_discharge(cycle_number):
    cycle = wide[
        (wide["cycle_number"] == cycle_number)
        & (wide["cycle_type"] == "discharge")
    ].copy()

    if cycle.empty:
        raise ValueError(f"Discharge cycle {cycle_number} not found.")

    cycle["discharge_current_a"] = (
        -cycle["Current_measured"]
    ).clip(lower=0)

    active = cycle[
        cycle["discharge_current_a"] >= MIN_CURRENT_A
    ].sort_values("time_s").reset_index(drop=True)

    if len(active) < 10:
        raise ValueError(f"Too few active samples in cycle {cycle_number}.")

    active["elapsed_time_s"] = (
        active["time_s"] - active["time_s"].iloc[0]
    )

    return active

# Fit the constrained model on reference cycle 2 only.
reference = get_active_discharge(REFERENCE_CYCLE)
t_ref = reference["elapsed_time_s"].to_numpy(float)
T_ref = reference["Temperature_measured"].to_numpy(float)

T0_ref = float(T_ref[0])

def constrained_model(time, amplitude, tau):
    return T0_ref + amplitude * (1.0 - np.exp(-time / tau))

from scipy.optimize import curve_fit

params, _ = curve_fit(
    constrained_model,
    t_ref,
    T_ref,
    p0=[float(T_ref[-1] - T0_ref), 1800.0],
    bounds=([-50.0, 1.0], [50.0, 100000.0]),
    maxfev=50000,
)

amplitude_ref, tau_ref = params

print("FROZEN REFERENCE PARAMETERS")
print(f"Reference cycle: {REFERENCE_CYCLE}")
print(f"Initial temperature: {T0_ref:.6f} °C")
print(f"Rise amplitude: {amplitude_ref:.6f} °C")
print(f"Time constant: {tau_ref:.6f} s")
print(f"Time constant: {tau_ref / 60:.3f} min")

all_results = []
all_predictions = []

for cycle_number in [REFERENCE_CYCLE] + TEST_CYCLES:
    active = get_active_discharge(cycle_number)

    t = active["elapsed_time_s"].to_numpy(float)
    measured = active["Temperature_measured"].to_numpy(float)

    # For independent cycles, anchor at their own first measured temperature.
    # Only amplitude and time constant are frozen from cycle 2.
    prediction = (
        float(measured[0])
        + amplitude_ref * (1.0 - np.exp(-t / tau_ref))
    )

    error = measured - prediction
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))

    all_results.append({
        "cycle_number": cycle_number,
        "evaluation": (
            "reference_in_sample"
            if cycle_number == REFERENCE_CYCLE
            else "independent_cycle"
        ),
        "samples": len(active),
        "duration_s": float(t[-1]),
        "initial_temperature_c": float(measured[0]),
        "final_temperature_c": float(measured[-1]),
        "measured_rise_c": float(measured[-1] - measured[0]),
        "predicted_final_temperature_c": float(prediction[-1]),
        "rmse_c": rmse,
        "mae_c": mae,
        "bias_c": bias,
    })

    part = active[
        ["cycle_number", "sample_index", "time_s",
         "Temperature_measured", "discharge_current_a"]
    ].copy()
    part["elapsed_time_s"] = t
    part["predicted_temperature_c"] = prediction
    part["residual_c"] = error
    all_predictions.append(part)

metrics = pd.DataFrame(all_results)
predictions = pd.concat(all_predictions, ignore_index=True)

metrics_path = OUT / "B0005_thermal_cross_cycle_metrics.csv"
predictions_path = OUT / "B0005_thermal_cross_cycle_predictions.csv"

metrics.to_csv(metrics_path, index=False)
predictions.to_csv(predictions_path, index=False)

print("\nCROSS-CYCLE VALIDATION")
print(metrics.round(4).to_string(index=False))

test = metrics[metrics["evaluation"] == "independent_cycle"]

print("\nINDEPENDENT TEST CYCLE SUMMARY")
print(f"Mean test RMSE: {test['rmse_c'].mean():.4f} °C")
print(f"Mean test MAE: {test['mae_c'].mean():.4f} °C")
print(f"Mean absolute bias: {test['bias_c'].abs().mean():.4f} °C")

print(f"\nMetrics saved: {metrics_path}")
print(f"Predictions saved: {predictions_path}")
