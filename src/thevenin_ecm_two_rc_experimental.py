
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "ecm_soc_voltage"
    / "B0005_cycle_2_soc_ecm.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "ecm_two_rc_experimental"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Discharge current is positive in this input file.
# The OCV-SOC curve is illustrative, not experimentally calibrated.
MIN_R0 = 0.001
MAX_R0 = 0.30
MIN_R = 0.0001
MAX_R = 0.50
MIN_TAU = 1.0
MAX_TAU = 2000.0


def simulate_two_rc(time_s, current_a, ocv_v, r0, r1, tau1, r2, tau2):
    """
    Simulate terminal voltage using two first-order polarization branches.

    Convention:
        Positive current = discharge.
        V_terminal = OCV - I*R0 - Vp1 - Vp2.

    For each branch:
        dVp/dt = -Vp/tau + (R/tau)*I

    The exact zero-order-hold update is used over each sample interval.
    Initial polarization voltages are assumed to be zero.
    """
    n = len(time_s)
    vp1 = np.zeros(n, dtype=float)
    vp2 = np.zeros(n, dtype=float)

    for k in range(1, n):
        dt = time_s[k] - time_s[k - 1]
        if dt <= 0:
            raise ValueError("Timestamps must be strictly increasing.")

        a1 = np.exp(-dt / tau1)
        a2 = np.exp(-dt / tau2)

        # Assume current is constant between consecutive samples.
        vp1[k] = a1 * vp1[k - 1] + r1 * (1.0 - a1) * current_a[k - 1]
        vp2[k] = a2 * vp2[k - 1] + r2 * (1.0 - a2) * current_a[k - 1]

    predicted_v = ocv_v - current_a * r0 - vp1 - vp2
    return predicted_v, vp1, vp2


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "time_s",
        "current_discharge_positive_a",
        "measured_voltage_v",
        "soc_bounded",
        "illustrative_ocv_v",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.sort_values("time_s").reset_index(drop=True)

    time_s = df["time_s"].to_numpy(dtype=float)
    current_a = df["current_discharge_positive_a"].to_numpy(dtype=float)
    measured_v = df["measured_voltage_v"].to_numpy(dtype=float)
    soc = df["soc_bounded"].to_numpy(dtype=float)
    ocv_v = df["illustrative_ocv_v"].to_numpy(dtype=float)

    valid = (
        np.isfinite(time_s)
        & np.isfinite(current_a)
        & np.isfinite(measured_v)
        & np.isfinite(soc)
        & np.isfinite(ocv_v)
    )

    time_s = time_s[valid]
    current_a = current_a[valid]
    measured_v = measured_v[valid]
    soc = soc[valid]
    ocv_v = ocv_v[valid]

    if len(time_s) < 20:
        raise ValueError("Too few valid samples for model fitting.")

    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Time values must be strictly increasing.")

    # Fit and report the active-discharge interval separately.
    active_mask = current_a >= 0.5
    if active_mask.sum() < 10:
        raise ValueError("Too few active-discharge samples.")

    # Parameterization ensures tau1 < tau2.
    # Parameters are [R0, R1, tau1, R2, delta_tau].
    # tau2 = tau1 + delta_tau.
    def unpack(p):
        r0, r1, tau1, r2, delta_tau = p
        return r0, r1, tau1, r2, tau1 + delta_tau

    def predict(p):
        r0, r1, tau1, r2, tau2 = unpack(p)
        predicted, _, _ = simulate_two_rc(
            time_s, current_a, ocv_v, r0, r1, tau1, r2, tau2
        )
        return predicted

    # Initial estimates are numerical starting values, not measured
    # physical parameters. Try several starts to reduce local-fit risk.
    initial_guesses = [
        [0.08, 0.01, 30.0, 0.02, 130.0],
        [0.10, 0.02, 20.0, 0.03, 180.0],
        [0.06, 0.03, 50.0, 0.01, 100.0],
    ]

    lower = [MIN_R0, MIN_R, MIN_TAU, MIN_R, 1.0]
    upper = [MAX_R0, MAX_R, 500.0, MAX_R, MAX_TAU]

    fit_results = []

    for start_id, initial in enumerate(initial_guesses, start=1):
        fit = least_squares(
            lambda p: predict(p)[active_mask] - measured_v[active_mask],
            x0=initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.02,
            max_nfev=30000,
        )

        pred = predict(fit.x)
        residual = measured_v - pred
        active_residual = residual[active_mask]

        rmse = float(np.sqrt(np.mean(active_residual**2)))
        mae = float(np.mean(np.abs(active_residual)))
        bias = float(np.mean(active_residual))

        r0, r1, tau1, r2, tau2 = unpack(fit.x)

        fit_results.append({
            "start_id": start_id,
            "r0_ohm": r0,
            "r1_ohm": r1,
            "tau1_s": tau1,
            "c1_f": tau1 / r1,
            "r2_ohm": r2,
            "tau2_s": tau2,
            "c2_f": tau2 / r2,
            "active_rmse_v": rmse,
            "active_mae_v": mae,
            "active_bias_v": bias,
            "optimizer_success": fit.success,
            "optimizer_message": fit.message,
            "parameters": fit.x,
        })

    # Select the fit with the lowest active-discharge RMSE.
    best = min(fit_results, key=lambda item: item["active_rmse_v"])
    p_best = best["parameters"]
    r0, r1, tau1, r2, tau2 = unpack(p_best)

    predicted_v, vp1, vp2 = simulate_two_rc(
        time_s, current_a, ocv_v, r0, r1, tau1, r2, tau2
    )

    residual_v = measured_v - predicted_v

    # Compare on exactly the same active-discharge samples.
    active_residual = residual_v[active_mask]
    active_rmse = float(np.sqrt(np.mean(active_residual**2)))
    active_mae = float(np.mean(np.abs(active_residual)))
    active_bias = float(np.mean(active_residual))

    output = df.loc[valid].copy()
    output["two_rc_vp1_v"] = vp1
    output["two_rc_vp2_v"] = vp2
    output["two_rc_predicted_voltage_v"] = predicted_v
    output["two_rc_voltage_error_v"] = residual_v
    output["active_discharge_sample"] = active_mask

    predictions_file = OUTPUT_DIR / "B0005_cycle_2_two_rc_predictions.csv"
    output.to_csv(predictions_file, index=False)

    # Save all multi-start fits for comparison.
    fit_table = pd.DataFrame([
        {k: v for k, v in item.items() if k != "parameters"}
        for item in fit_results
    ])
    fits_file = OUTPUT_DIR / "B0005_cycle_2_two_rc_multistart_fits.csv"
    fit_table.to_csv(fits_file, index=False)

    parameter_report = OUTPUT_DIR / "B0005_cycle_2_two_rc_report.txt"
    with parameter_report.open("w", encoding="utf-8") as f:
        f.write("Experimental two-RC Thevenin ECM\n")
        f.write("================================\n")
        f.write("Dataset: NASA B0005, recorded cycle 2\n")
        f.write("OCV-SOC curve: illustrative, not calibrated\n")
        f.write("Fitting interval: active discharge, current >= 0.5 A\n")
        f.write("Initial polarization voltages assumed zero\n\n")
        f.write(f"Samples total: {len(time_s)}\n")
        f.write(f"Active samples: {active_mask.sum()}\n")
        f.write(f"R0: {r0:.8f} ohm\n")
        f.write(f"R1: {r1:.8f} ohm\n")
        f.write(f"tau1: {tau1:.8f} s\n")
        f.write(f"C1: {tau1 / r1:.8f} F\n")
        f.write(f"R2: {r2:.8f} ohm\n")
        f.write(f"tau2: {tau2:.8f} s\n")
        f.write(f"C2: {tau2 / r2:.8f} F\n")
        f.write(f"Active RMSE: {active_rmse:.8f} V\n")
        f.write(f"Active MAE: {active_mae:.8f} V\n")
        f.write(f"Active bias (measured - predicted): {active_bias:.8f} V\n")
        f.write(f"Optimizer success: {best['optimizer_success']}\n\n")
        f.write("WARNING: This is an in-sample experimental fit.\n")
        f.write("The illustrative OCV curve and assumed zero initial\n")
        f.write("polarization voltage can confound fitted parameters.\n")
        f.write("Do not interpret fitted R/C values as validated physical\n")
        f.write("properties without independent validation and identifiability checks.\n")

    # Plot voltage predictions and active-discharge residuals.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(time_s, measured_v, label="Measured voltage")
    axes[0].plot(time_s, predicted_v, "--", label="Experimental two-RC")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("NASA B0005 cycle 2: experimental two-RC ECM")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, linewidth=1)
    axes[1].plot(
        time_s[active_mask],
        residual_v[active_mask],
        marker="o",
        label="Active-discharge residual",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Measured − predicted (V)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plot_file = OUTPUT_DIR / "B0005_cycle_2_two_rc_plot.png"
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)

    print("\nEXPERIMENTAL TWO-RC THEVENIN ECM")
    print("================================")
    print(f"Total samples: {len(time_s)}")
    print(f"Active-discharge samples: {active_mask.sum()}")
    print(f"R0: {r0:.6f} ohm")
    print(f"R1: {r1:.6f} ohm")
    print(f"Tau1: {tau1:.3f} s")
    print(f"C1: {tau1 / r1:.3f} F")
    print(f"R2: {r2:.6f} ohm")
    print(f"Tau2: {tau2:.3f} s")
    print(f"C2: {tau2 / r2:.3f} F")
    print(f"Active RMSE: {active_rmse:.6f} V")
    print(f"Active MAE: {active_mae:.6f} V")
    print(f"Active bias: {active_bias:.6f} V")
    print("\nMulti-start fits:")
    print(fit_table.to_string(index=False))
    print(f"\nPredictions: {predictions_file}")
    print(f"Report: {parameter_report}")
    print(f"Plot: {plot_file}")
    print(f"Fit comparison: {fits_file}")


if __name__ == "__main__":
    main()
