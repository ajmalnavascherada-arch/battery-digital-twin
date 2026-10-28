
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "voltage_relaxation"
    / "B0005_cycle_2_relaxation_samples.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "relaxation_model_comparison"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MIN_SAMPLES = 8
RANDOM_SEED = 42
BOOTSTRAP_REPEATS = 100


def single_exponential(t, v0, amplitude, tau):
    """Voltage recovery with one exponential component."""
    return v0 + amplitude * (1.0 - np.exp(-t / tau))


def double_exponential(t, v0, amplitude_fast, tau_fast,
                       amplitude_slow, tau_slow):
    """Voltage recovery with fast and slow exponential components."""
    return (
        v0
        + amplitude_fast * (1.0 - np.exp(-t / tau_fast))
        + amplitude_slow * (1.0 - np.exp(-t / tau_slow))
    )


def calculate_metrics(measured, predicted, parameter_count):
    residuals = measured - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    sse = float(np.sum(residuals**2))
    n = len(measured)

    # AIC based on Gaussian residuals; useful for relative comparison.
    aic = (
        n * np.log(max(sse / n, np.finfo(float).tiny))
        + 2 * parameter_count
    )

    # Corrected AIC; undefined when there are too few observations.
    if n > parameter_count + 1:
        aicc = aic + (
            2 * parameter_count * (parameter_count + 1)
            / (n - parameter_count - 1)
        )
    else:
        aicc = np.nan

    return {
        "rmse_v": rmse,
        "mae_v": mae,
        "sse_v2": sse,
        "aic": float(aic),
        "aicc": float(aicc) if np.isfinite(aicc) else np.nan,
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n"
            "Check that the relaxation analysis has already been run."
        )

    df = pd.read_csv(INPUT_FILE)

    print("Input columns:", list(df.columns))
    print("Input rows:", len(df))

    # Detect the columns from common names used by the existing script.
    time_candidates = [
        "time_since_rest_s",
        "time_since_start_s",
        "time_s",
        "relative_time_s",
    ]
    voltage_candidates = [
        "voltage_v",
        "measured_voltage_v",
        "voltage",
    ]

    time_col = next((c for c in time_candidates if c in df.columns), None)
    voltage_col = next(
        (c for c in voltage_candidates if c in df.columns), None
    )

    # If the original file uses different names, stop rather than guessing.
    if time_col is None or voltage_col is None:
        raise ValueError(
            "Could not identify time and voltage columns safely.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Select the longest segment if a segment identifier is available.
    if "segment_id" in df.columns:
        segment_counts = df.groupby("segment_id").size()
        selected_segment = segment_counts.idxmax()
        df = df[df["segment_id"] == selected_segment].copy()
        print("Selected segment:", selected_segment)

    df = df[[time_col, voltage_col]].copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[voltage_col] = pd.to_numeric(df[voltage_col], errors="coerce")
    df = df.dropna().sort_values(time_col)

    # Reset time to zero at the start of the selected rest segment.
    t = df[time_col].to_numpy(dtype=float)
    voltage = df[voltage_col].to_numpy(dtype=float)
    t = t - t[0]

    # Remove duplicate timestamps, if any.
    unique_mask = np.concatenate(([True], np.diff(t) > 0))
    t = t[unique_mask]
    voltage = voltage[unique_mask]

    if len(t) < MIN_SAMPLES:
        raise ValueError(
            f"Only {len(t)} valid samples; need at least {MIN_SAMPLES}."
        )

    # Use observed recovery span for initial estimates.
    v0_guess = float(voltage[0])
    amplitude_guess = max(float(voltage[-1] - voltage[0]), 1e-4)
    duration = max(float(t[-1] - t[0]), 1.0)

    # -----------------------------------------------------
    # Fit single-exponential model
    # -----------------------------------------------------
    single_initial = [v0_guess, amplitude_guess, duration / 3.0]

    single_lower = [voltage.min() - 0.2, 0.0, 1.0]
    single_upper = [voltage.max() + 0.2, 1.0, 10000.0]

    fit_single = least_squares(
        lambda p: single_exponential(t, *p) - voltage,
        x0=single_initial,
        bounds=(single_lower, single_upper),
        loss="linear",
        max_nfev=20000,
    )

    pred_single = single_exponential(t, *fit_single.x)
    metrics_single = calculate_metrics(voltage, pred_single, 3)

    # -----------------------------------------------------
    # Fit two-exponential model
    # -----------------------------------------------------
    # Enforce tau_fast < tau_slow by using log(tau_fast) and
    # log(tau_slow / tau_fast) as the fitted time parameters.
    def double_from_transformed(t_values, p):
        v0, a_fast, a_slow, log_tau_fast, log_tau_ratio = p
        tau_fast = np.exp(log_tau_fast)
        tau_slow = tau_fast * np.exp(log_tau_ratio)
        return double_exponential(
            t_values, v0, a_fast, tau_fast, a_slow, tau_slow
        )

    double_initial = [
        v0_guess,
        amplitude_guess * 0.65,
        amplitude_guess * 0.35,
        np.log(max(duration / 15.0, 1.0)),
        np.log(10.0),
    ]

    double_lower = [
        voltage.min() - 0.2,
        0.0,
        0.0,
        np.log(1.0),
        np.log(1.01),
    ]

    double_upper = [
        voltage.max() + 0.2,
        1.0,
        1.0,
        np.log(max(duration, 2.0)),
        np.log(10000.0),
    ]

    fit_double = least_squares(
        lambda p: double_from_transformed(t, p) - voltage,
        x0=double_initial,
        bounds=(double_lower, double_upper),
        loss="linear",
        max_nfev=50000,
    )

    pred_double = double_from_transformed(t, fit_double.x)
    metrics_double = calculate_metrics(voltage, pred_double, 5)

    v0_double, a_fast, a_slow, log_tau_fast, log_tau_ratio = fit_double.x
    tau_fast = float(np.exp(log_tau_fast))
    tau_slow = float(tau_fast * np.exp(log_tau_ratio))

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------
    results = pd.DataFrame([
        {
            "model": "single_exponential",
            "n_samples": len(t),
            "v0_v": fit_single.x[0],
            "amplitude_fast_v": fit_single.x[1],
            "tau_fast_s": fit_single.x[2],
            "amplitude_slow_v": np.nan,
            "tau_slow_s": np.nan,
            **metrics_single,
            "optimizer_success": fit_single.success,
        },
        {
            "model": "double_exponential",
            "n_samples": len(t),
            "v0_v": v0_double,
            "amplitude_fast_v": a_fast,
            "tau_fast_s": tau_fast,
            "amplitude_slow_v": a_slow,
            "tau_slow_s": tau_slow,
            **metrics_double,
            "optimizer_success": fit_double.success,
        },
    ])

    results_file = OUTPUT_DIR / "relaxation_model_metrics.csv"
    results.to_csv(results_file, index=False)

    predictions = pd.DataFrame({
        "time_since_rest_s": t,
        "measured_voltage_v": voltage,
        "single_exponential_voltage_v": pred_single,
        "double_exponential_voltage_v": pred_double,
        "single_residual_v": voltage - pred_single,
        "double_residual_v": voltage - pred_double,
    })
    predictions_file = OUTPUT_DIR / "relaxation_model_predictions.csv"
    predictions.to_csv(predictions_file, index=False)

    # -----------------------------------------------------
    # Plot voltage fit and residuals
    # -----------------------------------------------------
    dense_t = np.linspace(t.min(), t.max(), 300)
    dense_single = single_exponential(dense_t, *fit_single.x)
    dense_double = double_from_transformed(
        dense_t, fit_double.x
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].scatter(
        t, voltage, label="Measured voltage", s=35
    )
    axes[0].plot(
        dense_t, dense_single, label="Single exponential"
    )
    axes[0].plot(
        dense_t, dense_double, "--", label="Two exponentials"
    )
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("NASA B0005 cycle 2: relaxation model comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, linewidth=1)
    axes[1].plot(
        t, voltage - pred_single, marker="o",
        label="Single-exponential residual"
    )
    axes[1].plot(
        t, voltage - pred_double, marker="s",
        label="Two-exponential residual"
    )
    axes[1].set_xlabel("Time since rest segment began (s)")
    axes[1].set_ylabel("Measured − predicted (V)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plot_file = OUTPUT_DIR / "relaxation_model_comparison.png"
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)

    print("\nRELAXATION MODEL COMPARISON")
    print("===========================")
    print(f"Samples: {len(t)}")
    print(f"Duration: {duration:.3f} s")
    print(f"Single-exponential tau: {fit_single.x[2]:.3f} s")
    print(f"Two-exponential fast tau: {tau_fast:.3f} s")
    print(f"Two-exponential slow tau: {tau_slow:.3f} s")
    print(f"Single RMSE: {metrics_single['rmse_v']:.6f} V")
    print(f"Double RMSE: {metrics_double['rmse_v']:.6f} V")
    print(f"Single MAE: {metrics_single['mae_v']:.6f} V")
    print(f"Double MAE: {metrics_double['mae_v']:.6f} V")
    print(f"Single AICc: {metrics_single['aicc']:.3f}")
    print(f"Double AICc: {metrics_double['aicc']:.3f}")
    print(f"\nMetrics: {results_file}")
    print(f"Predictions: {predictions_file}")
    print(f"Plot: {plot_file}")

    print(
        "\nCAUTION: A lower error from the two-exponential fit does not "
        "alone establish physical validity. With only a small number of "
        "rest samples, two time constants may be weakly identifiable."
    )


if __name__ == "__main__":
    main()
