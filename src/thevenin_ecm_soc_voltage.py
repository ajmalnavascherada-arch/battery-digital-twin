
"""SOC-dependent first-order Thevenin ECM prototype.

NASA B0005 discharge current is converted to a positive-discharge convention.
The OCV-SOC curve below is illustrative, not experimentally calibrated.
Do not use this prototype for safety-critical battery control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# Illustrative curve only. Replace with validated OCV-SOC data when available.
SOC_KNOTS = np.array([0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00])
OCV_KNOTS_V = np.array([3.00, 3.30, 3.45, 3.58, 3.70, 3.82, 3.98, 4.10, 4.20])


def ocv_from_soc(soc: np.ndarray) -> np.ndarray:
    """Interpolate the illustrative OCV-SOC lookup curve."""
    return np.interp(np.clip(soc, 0.0, 1.0), SOC_KNOTS, OCV_KNOTS_V)


def load_discharge_cycle(path: Path, cycle_number: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[
        (df["cycle_number"] == cycle_number)
        & (df["cycle_type"] == "discharge")
        & (
            df["signal"].isin(
                ["Voltage_measured", "Current_measured", "Temperature_measured"]
            )
        )
    ].copy()

    if df.empty:
        raise ValueError(f"No discharge samples found for cycle {cycle_number}.")

    wide = df.pivot(index="sample_index", columns="signal", values="value")
    wide = wide.sort_index()

    required = ["Voltage_measured", "Current_measured"]
    if any(col not in wide.columns for col in required):
        raise ValueError("Voltage or current measurements are missing.")

    time_rows = (
        df.loc[
            df["signal"] == "Voltage_measured",
            ["sample_index", "time_s", "time_aligned"],
        ]
        .drop_duplicates("sample_index")
        .set_index("sample_index")
    )
    wide = wide.join(time_rows, how="inner")
    wide = wide.dropna(subset=required + ["time_s"])

    aligned = wide["time_aligned"].astype(str).str.lower().isin(["true", "1"])
    if not aligned.all():
        raise ValueError("Some voltage samples lack aligned timestamps.")

    wide = wide.reset_index().sort_values("time_s").reset_index(drop=True)
    if len(wide) < 3:
        raise ValueError("At least three samples are required.")

    time_s = wide["time_s"].to_numpy(float)
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing.")

    wide["current_a"] = -wide["Current_measured"]  # discharge positive
    wide["voltage_v"] = wide["Voltage_measured"]
    return wide


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signals",
        default=Path(
            r"C:\Users\Ajmal\battery-digital-twin\data\processed_nasa_original\B0005_signals_long.csv"
        ),
    )
    parser.add_argument("--cycle", type=int, default=2)
    parser.add_argument("--capacity-ah", type=float, default=1.856487)
    parser.add_argument("--initial-soc", type=float, default=1.0)
    parser.add_argument("--out-dir", default="results_nasa_original/ecm_soc_voltage")
    args = parser.parse_args()

    if args.capacity_ah <= 0:
        raise ValueError("--capacity-ah must be positive.")
    if not 0 <= args.initial_soc <= 1:
        raise ValueError("--initial-soc must be between 0 and 1.")

    data = load_discharge_cycle(Path(args.signals), args.cycle)
    time_s = data["time_s"].to_numpy(float)
    current_a = data["current_a"].to_numpy(float)
    measured_v = data["voltage_v"].to_numpy(float)

    # Coulomb-counted SOC. Raw values are retained for diagnostics;
    # OCV evaluation uses the physically bounded SOC.
    dt = np.diff(time_s)
    delta_ah = 0.5 * (current_a[:-1] + current_a[1:]) * dt / 3600.0
    removed_ah = np.r_[0.0, np.cumsum(delta_ah)]
    soc_raw = args.initial_soc - removed_ah / args.capacity_ah
    soc = np.clip(soc_raw, 0.0, 1.0)
    ocv_v = ocv_from_soc(soc)

    # Fit R0, R1, and tau to this cycle. OCV curve and SOC are held fixed.
    # This is a diagnostic in-sample fit, not independent validation.
    def simulate(params: np.ndarray) -> np.ndarray:
        r0, r1, tau = params
        vp = np.zeros_like(current_a)
        for k in range(1, len(current_a)):
            step = time_s[k] - time_s[k - 1]
            decay = np.exp(-step / tau)
            vp[k] = (
                decay * vp[k - 1]
                + r1 * (1.0 - decay) * current_a[k - 1]
            )
        return ocv_v - current_a * r0 - vp

    fit = least_squares(
        lambda params: simulate(params) - measured_v,
        x0=np.array([0.03, 0.02, 100.0]),
        bounds=(
            np.array([0.0001, 0.0001, 1.0]),
            np.array([0.20, 0.20, 5000.0]),
        ),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=5000,
    )

    predicted_v = simulate(fit.x)
    error = predicted_v - measured_v
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))

    result = pd.DataFrame(
        {
            "sample_index": data["sample_index"],
            "time_s": time_s,
            "current_discharge_positive_a": current_a,
            "measured_voltage_v": measured_v,
            "cumulative_removed_ah": removed_ah,
            "soc_raw": soc_raw,
            "soc_bounded": soc,
            "illustrative_ocv_v": ocv_v,
            "predicted_voltage_v": predicted_v,
            "voltage_error_v": error,
        }
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"B0005_cycle_{args.cycle}_soc_ecm.csv"
    plot_path = out / f"B0005_cycle_{args.cycle}_soc_ecm.png"
    report_path = out / f"B0005_cycle_{args.cycle}_soc_ecm_report.txt"
    result.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_s, measured_v, label="Measured voltage")
    axes[0].plot(time_s, predicted_v, label="SOC-dependent ECM")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title(f"B0005 recorded cycle {args.cycle}: ECM fit")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time_s, error, label="Predicted − measured")
    axes[1].axhline(0, linewidth=1)
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("Voltage error (V)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    report = f"""SOC-dependent Thevenin ECM prototype
=====================================
Cycle: {args.cycle}
Samples: {len(data)}
Assumed initial SOC: {args.initial_soc:.4f}
Capacity used: {args.capacity_ah:.6f} Ah
Integrated discharge: {removed_ah[-1]:.6f} Ah
Raw final SOC: {soc_raw[-1]:.6f}
Fitted R0: {fit.x[0]:.6f} ohm
Fitted R1: {fit.x[1]:.6f} ohm
Fitted tau: {fit.x[2]:.6f} s
In-sample voltage RMSE: {rmse:.6f} V
In-sample voltage MAE: {mae:.6f} V
Optimizer success: {fit.success}
Optimizer message: {fit.message}

IMPORTANT LIMITATIONS
- The OCV-SOC lookup curve is illustrative, not calibrated from NASA data.
- Initial SOC and capacity are assumptions/approximations.
- SOC is inferred by integrating current, not measured directly.
- Parameters and errors are fitted/evaluated on the same cycle.
- Do not interpret the fitted parameters as validated cell properties.
- Not for safety-critical use.
"""
    report_path.write_text(report, encoding="utf-8")

    print("SOC-dependent Thevenin ECM prototype")
    print("====================================")
    print(f"Cycle: {args.cycle}")
    print(f"Samples: {len(data)}")
    print(f"Integrated discharge: {removed_ah[-1]:.6f} Ah")
    print(f"Raw final SOC: {soc_raw[-1]:.2%}")
    print(f"Fitted R0: {fit.x[0]:.6f} ohm")
    print(f"Fitted R1: {fit.x[1]:.6f} ohm")
    print(f"Fitted tau: {fit.x[2]:.6f} s")
    print(f"Voltage RMSE: {rmse:.6f} V")
    print(f"Voltage MAE: {mae:.6f} V")
    print(f"Optimizer success: {fit.success}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Report: {report_path}")
    print("Warning: illustrative OCV curve; in-sample fit only.")


if __name__ == "__main__":
    main()
