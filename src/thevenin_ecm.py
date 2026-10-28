"""First-order Thevenin ECM baseline for one verified NASA discharge cycle.

This is an initial engineering baseline, not a production BMS model.
It uses a simple constant-OCV approximation over a selected discharge segment
and fits R0, R1, and tau by bounded least squares. Parameters must be checked
for identifiability and physical plausibility before interpretation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def load_cycle(signals_path: Path, cycle_number: int) -> pd.DataFrame:
    df = pd.read_csv(signals_path)
    required = {"cycle_number", "cycle_type", "signal", "sample_index", "time_s", "value", "time_aligned"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Signal file missing columns: {sorted(missing)}")

    one = df.loc[df["cycle_number"].eq(cycle_number)]
    if one.empty:
        raise ValueError(f"Cycle {cycle_number} not found in {signals_path}")

    cycle_types = one["cycle_type"].dropna().astype(str).str.lower().unique()
    if len(cycle_types) != 1 or cycle_types[0] != "discharge":
        raise ValueError(f"Cycle {cycle_number} is not a unique discharge cycle: {cycle_types}")

    columns = {}
    for signal_name in ["Voltage_measured", "Current_measured", "Temperature_measured"]:
        part = one.loc[one["signal"].eq(signal_name)].sort_values("sample_index")
        if part.empty:
            raise ValueError(f"Signal {signal_name} is missing in cycle {cycle_number}")
        if not part["time_aligned"].astype(str).str.lower().isin(["true", "1"]).all():
            raise ValueError(f"{signal_name} does not have time-aligned samples")
        columns[signal_name] = part["value"].to_numpy(dtype=float)
        columns[f"{signal_name}_time_s"] = part["time_s"].to_numpy(dtype=float)

    t = columns["Voltage_measured_time_s"]
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("Voltage timestamps must be finite and strictly increasing")
    for signal_name in ["Current_measured", "Temperature_measured"]:
        if len(columns[signal_name]) != len(t) or not np.allclose(columns[f"{signal_name}_time_s"], t):
            raise ValueError(f"{signal_name} timestamps do not match voltage timestamps")

    out = pd.DataFrame({
        "time_s": t,
        "voltage_V": columns["Voltage_measured"],
        "current_A": columns["Current_measured"],
        "temperature_C": columns["Temperature_measured"],
    })
    if not np.isfinite(out[["voltage_V", "current_A", "temperature_C"]].to_numpy()).all():
        raise ValueError("Non-finite values found in selected cycle")
    return out


def select_load_segment(df: pd.DataFrame, current_threshold_A: float = 0.5) -> pd.DataFrame:
    """Select the contiguous high-current segment, retaining original time."""
    current = df["current_A"].to_numpy()
    active = np.abs(current) >= current_threshold_A
    indices = np.flatnonzero(active)
    if len(indices) < 20:
        raise ValueError("Could not find a sufficiently long active-current segment")
    # Use the longest contiguous active run; do not concatenate separate segments.
    splits = np.where(np.diff(indices) > 1)[0] + 1
    runs = np.split(indices, splits)
    run = max(runs, key=len)
    if len(run) < 20:
        raise ValueError("Longest active-current segment has fewer than 20 samples")
    return df.iloc[run].reset_index(drop=True)


def simulate_voltage(t: np.ndarray, current: np.ndarray, ocv: float,
                     r0: float, r1: float, tau: float) -> np.ndarray:
    """Discrete first-order Thevenin model with current-positive-for-discharge."""
    n = len(t)
    vp = np.zeros(n, dtype=float)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        if dt <= 0:
            raise ValueError("Time must be strictly increasing")
        decay = np.exp(-dt / tau)
        # Discretised polarization voltage: dVp/dt = -Vp/tau + (R1/tau)*I
        vp[k] = decay * vp[k - 1] + r1 * (1.0 - decay) * current[k - 1]
    return ocv - current * r0 - vp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path,
                        default=Path("data/processed_nasa_original/B0005_signals_long.csv"))
    parser.add_argument("--cycle", type=int, default=2,
                        help="Recorded cycle number; default 2 is first discharge")
    parser.add_argument("--out-dir", type=Path, default=Path("results_nasa_original/ecm"))
    parser.add_argument("--current-threshold", type=float, default=0.5,
                        help="Absolute current threshold in A for selecting active segment")
    args = parser.parse_args()

    if not args.signals.is_file():
        parser.error(f"Signal CSV not found: {args.signals}. Run the data pipeline first.")

    try:
        full = load_cycle(args.signals, args.cycle)
        segment = select_load_segment(full, args.current_threshold)
    except ValueError as exc:
        parser.error(str(exc))

    t_abs = segment["time_s"].to_numpy()
    t = t_abs - t_abs[0]
    voltage = segment["voltage_V"].to_numpy()
    measured_current = segment["current_A"].to_numpy()

    # NASA discharge current is generally negative. Convert convention so
    # positive current denotes discharge in V = OCV - I*R0 - Vp.
    current = -measured_current

    # OCV is approximated as a constant over this short segment. This is a
    # deliberately limited baseline; OCV-SOC dependence is a later stage.
    ocv_initial = float(np.percentile(voltage + current * 0.02, 95))
    voltage_span = float(np.ptp(voltage))
    r0_upper = max(0.15, min(1.0, voltage_span / max(np.ptp(current), 0.1)))
    initial = np.array([ocv_initial, 0.02, 0.02, 20.0])
    lower = np.array([float(np.min(voltage)), 1e-5, 1e-5, 0.2])
    upper = np.array([float(np.max(voltage) + 0.5), r0_upper, 1.0, 1000.0])

    def residual(params: np.ndarray) -> np.ndarray:
        ocv, r0, r1, tau = params
        return simulate_voltage(t, current, ocv, r0, r1, tau) - voltage

    fit = least_squares(
        residual, x0=np.clip(initial, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper), loss="soft_l1", max_nfev=5000
    )
    ocv, r0, r1, tau = fit.x
    predicted = simulate_voltage(t, current, ocv, r0, r1, tau)
    errors = predicted - voltage
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.out_dir / f"B0005_cycle_{args.cycle}_ecm_predictions.csv"
    output_plot = args.out_dir / f"B0005_cycle_{args.cycle}_ecm_voltage.png"
    output_report = args.out_dir / f"B0005_cycle_{args.cycle}_ecm_report.txt"

    results = segment.copy()
    results["current_positive_discharge_A"] = current
    results["predicted_voltage_V"] = predicted
    results["voltage_error_V"] = errors
    results.to_csv(output_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t, voltage, label="Measured voltage", linewidth=1.5)
    ax.plot(t, predicted, label="Thevenin ECM baseline", linewidth=1.3)
    ax.set_title(f"B0005 recorded cycle {args.cycle}: voltage prediction")
    ax.set_xlabel("Elapsed time in selected segment (s)")
    ax.set_ylabel("Terminal voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    report_lines = [
        "First-order Thevenin ECM — initial baseline",
        "=" * 47,
        f"Source cycle: {args.cycle}",
        f"Selected active segment samples: {len(segment)}",
        f"Segment duration: {t[-1] - t[0]:.3f} s",
        f"Fitted constant OCV approximation: {ocv:.6f} V",
        f"Fitted R0: {r0:.6f} ohm",
        f"Fitted R1: {r1:.6f} ohm",
        f"Fitted tau: {tau:.6f} s",
        f"Voltage RMSE: {rmse:.6f} V",
        f"Voltage MAE: {mae:.6f} V",
        f"Optimizer success: {fit.success}",
        f"Optimizer message: {fit.message}",
        "",
        "Limitations:",
        "- Constant OCV is a crude approximation across changing SOC.",
        "- Parameters are fitted on the same segment used for reported errors.",
        "- Metrics are in-sample fit metrics, not independent validation.",
        "- Parameter identifiability and physical plausibility need review.",
        "- Do not use this model for safety-critical control.",
    ]
    output_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nPredictions: {output_csv}")
    print(f"Plot: {output_plot}")
    print(f"Report: {output_report}")


if __name__ == "__main__":
    main()
