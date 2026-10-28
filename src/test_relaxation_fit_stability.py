
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "relaxation_model_comparison"
    / "relaxation_model_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_nasa_original"
    / "relaxation_stability"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def double_exponential(t, v0, a_fast, tau_fast, a_slow, tau_slow):
    return (
        v0
        + a_fast * (1.0 - np.exp(-t / tau_fast))
        + a_slow * (1.0 - np.exp(-t / tau_slow))
    )


def fit_model(t, voltage, start, end, initial_guess):
    """Fit two positive-amplitude exponentials to a selected time window."""
    mask = (t >= start) & (t <= end)
    tw = t[mask]
    vw = voltage[mask]

    if len(tw) < 7:
        return None

    duration = max(float(tw[-1] - tw[0]), 1.0)

    lower = [
        float(vw.min() - 0.2),
        0.0,
        1.0,
        0.0,
        1.01,
    ]
    upper = [
        float(vw.max() + 0.2),
        1.0,
        max(duration * 10.0, 10.0),
        1.0,
        10000.0,
    ]

    # Parameterize slow time constant as tau_fast + delta.
    # This guarantees tau_slow > tau_fast.
    def unpack(p):
        v0, a_fast, tau_fast, a_slow, delta_tau = p
        return v0, a_fast, tau_fast, a_slow, tau_fast + delta_tau

    # Bounds are adjusted for the transformed delta.
    lower_p = [lower[0], lower[1], lower[2], lower[3], 0.01]
    upper_p = [
        upper[0],
        upper[1],
        min(upper[2], 5000.0),
        upper[3],
        10000.0,
    ]

    p0 = np.asarray(initial_guess, dtype=float)
    p0[2] = np.clip(p0[2], lower_p[2] + 1e-4, upper_p[2] - 1e-4)
    p0[4] = max(p0[4] - p0[2], 0.02)

    def predict(p):
        return double_exponential(tw, *unpack(p))

    result = least_squares(
        lambda p: predict(p) - vw,
        x0=p0,
        bounds=(lower_p, upper_p),
        loss="linear",
        max_nfev=30000,
    )

    fitted = predict(result.x)
    residual = vw - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    v0, a_fast, tau_fast, a_slow, tau_slow = unpack(result.x)

    return {
        "window_start_s": start,
        "window_end_s": end,
        "n_samples": len(tw),
        "v0_v": v0,
        "amplitude_fast_v": a_fast,
        "tau_fast_s": tau_fast,
        "amplitude_slow_v": a_slow,
        "tau_slow_s": tau_slow,
        "rmse_v": rmse,
        "mae_v": mae,
        "optimizer_success": result.success,
        "optimizer_message": result.message,
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"time_since_rest_s", "measured_voltage_v"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    t = df["time_since_rest_s"].to_numpy(dtype=float)
    voltage = df["measured_voltage_v"].to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(voltage)
    t, voltage = t[valid], voltage[valid]
    order = np.argsort(t)
    t, voltage = t[order], voltage[order]

    duration = float(t[-1] - t[0])
    v0 = float(voltage[0])
    amplitude = max(float(voltage[-1] - voltage[0]), 1e-4)

    # Different initial guesses, including substantially different
    # fast and slow time constants.
    initial_guesses = [
        [v0, 0.65 * amplitude, 20.0, 0.35 * amplitude, 150.0],
        [v0, 0.50 * amplitude, 40.0, 0.50 * amplitude, 250.0],
        [v0, 0.80 * amplitude, 60.0, 0.20 * amplitude, 400.0],
        [v0, 0.30 * amplitude, 10.0, 0.70 * amplitude, 100.0],
        [v0, 0.60 * amplitude, 100.0, 0.40 * amplitude, 300.0],
        [v0, 0.40 * amplitude, 25.0, 0.60 * amplitude, 600.0],
    ]

    # Full window and shortened windows, preserving the rest-start time.
    windows = [
        (float(t[0]), float(t[-1])),
        (float(t[0]), 0.80 * duration),
        (float(t[0]), 0.65 * duration),
        (0.10 * duration, float(t[-1])),
    ]

    records = []

    for window_start, window_end in windows:
        for run_id, guess in enumerate(initial_guesses, start=1):
            try:
                result = fit_model(
                    t,
                    voltage,
                    window_start,
                    window_end,
                    guess.copy(),
                )
                if result is not None:
                    result["run_id"] = run_id
                    records.append(result)
            except (ValueError, FloatingPointError) as exc:
                records.append({
                    "window_start_s": window_start,
                    "window_end_s": window_end,
                    "run_id": run_id,
                    "optimizer_success": False,
                    "optimizer_message": str(exc),
                })

    results = pd.DataFrame(records)
    results_file = OUTPUT_DIR / "relaxation_fit_stability.csv"
    results.to_csv(results_file, index=False)

    successful = results[
        results["optimizer_success"].fillna(False)
        & results["tau_fast_s"].notna()
        & results["tau_slow_s"].notna()
    ].copy()

    if successful.empty:
        print("No successful fits. Inspect the error messages in the CSV.")
        print(f"Results: {results_file}")
        return

    summary = (
        successful.groupby(
            ["window_start_s", "window_end_s"], as_index=False
        )
        .agg(
            successful_fits=("optimizer_success", "sum"),
            fast_tau_median_s=("tau_fast_s", "median"),
            fast_tau_min_s=("tau_fast_s", "min"),
            fast_tau_max_s=("tau_fast_s", "max"),
            slow_tau_median_s=("tau_slow_s", "median"),
            slow_tau_min_s=("tau_slow_s", "min"),
            slow_tau_max_s=("tau_slow_s", "max"),
            rmse_median_v=("rmse_v", "median"),
            rmse_min_v=("rmse_v", "min"),
            rmse_max_v=("rmse_v", "max"),
        )
    )

    summary_file = OUTPUT_DIR / "relaxation_fit_stability_summary.csv"
    summary.to_csv(summary_file, index=False)

    print("\nRELAXATION FIT STABILITY")
    print("========================")
    print(f"Total attempted fits: {len(results)}")
    print(f"Successful fits: {len(successful)}")
    print(f"Observed duration: {duration:.3f} s")
    print("\nSummary by fitting window:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Plot the distribution of fitted time constants across all runs.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].scatter(
        successful["run_id"],
        successful["tau_fast_s"],
        c=successful["window_end_s"],
    )
    axes[0].set_xlabel("Initial-guess run ID")
    axes[0].set_ylabel("Fitted fast tau (s)")
    axes[0].set_title("Fast time-constant stability")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        successful["run_id"],
        successful["tau_slow_s"],
        c=successful["window_end_s"],
    )
    axes[1].set_xlabel("Initial-guess run ID")
    axes[1].set_ylabel("Fitted slow tau (s)")
    axes[1].set_title("Slow time-constant stability")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_file = OUTPUT_DIR / "relaxation_fit_stability.png"
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)

    print(f"\nDetailed results: {results_file}")
    print(f"Window summary: {summary_file}")
    print(f"Stability plot: {plot_file}")
    print(
        "\nInterpretation: stable time constants across initial guesses "
        "and windows support numerical robustness, but do not by themselves "
        "prove physical identifiability."
    )


if __name__ == "__main__":
    main()
