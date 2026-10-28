
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CYCLE_2_FILE = (
    PROJECT_ROOT / "results_nasa_original"
    / "ecm_soc_voltage" / "B0005_cycle_2_soc_ecm.csv"
)

VALIDATION_FILES = {
    4: PROJECT_ROOT / "results_nasa_original"
       / "ecm_validation" / "B0005_cycle_4_validation.csv",
    6: PROJECT_ROOT / "results_nasa_original"
       / "ecm_multi_validation" / "B0005_cycle_6_validation.csv",
    10: PROJECT_ROOT / "results_nasa_original"
        / "ecm_multi_validation" / "B0005_cycle_10_validation.csv",
    20: PROJECT_ROOT / "results_nasa_original"
        / "ecm_multi_validation" / "B0005_cycle_20_validation.csv",
}

OUTPUT_DIR = (
    PROJECT_ROOT / "results_nasa_original"
    / "ocv_shape_correction_validation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Train the correction only on the middle of cycle 2.
TRAIN_SOC_MIN = 0.15
TRAIN_SOC_MAX = 0.85

# Evaluate errors separately in the middle and near the voltage knee.
MID_SOC_MIN = 0.15
MID_SOC_MAX = 0.85
KNEE_SOC_MAX = 0.15
ACTIVE_CURRENT_THRESHOLD_A = 0.5


def load_cycle(path, cycle_number):
    if not path.exists():
        raise FileNotFoundError(
            f"Expected cycle {cycle_number} file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "measured_voltage_v",
        "predicted_voltage_v",
        "soc_bounded",
        "current_discharge_positive_a",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Cycle {cycle_number}: missing columns {sorted(missing)}\n"
            f"File: {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=list(required))

    # Use only active discharge for the voltage model comparison.
    df = df[
        df["current_discharge_positive_a"] >= ACTIVE_CURRENT_THRESHOLD_A
    ].copy()

    if df.empty:
        raise ValueError(f"No active-discharge samples in cycle {cycle_number}.")

    df["soc_bounded"] = df["soc_bounded"].clip(0.0, 1.0)
    df["residual_v"] = (
        df["measured_voltage_v"] - df["predicted_voltage_v"]
    )
    df["cycle_number"] = cycle_number

    return df.reset_index(drop=True)


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def calculate_metrics(df, corrected_prediction, region):
    soc = df["soc_bounded"].to_numpy()
    measured = df["measured_voltage_v"].to_numpy()
    baseline = df["predicted_voltage_v"].to_numpy()
    current = df["current_discharge_positive_a"].to_numpy()

    if region == "mid_discharge":
        mask = (
            (soc >= MID_SOC_MIN)
            & (soc <= MID_SOC_MAX)
            & (current >= ACTIVE_CURRENT_THRESHOLD_A)
        )
    elif region == "low_soc_knee":
        mask = (
            (soc <= KNEE_SOC_MAX)
            & (current >= ACTIVE_CURRENT_THRESHOLD_A)
        )
    else:
        mask = current >= ACTIVE_CURRENT_THRESHOLD_A

    if not np.any(mask):
        return {
            "samples": 0,
            "baseline_rmse_v": np.nan,
            "corrected_rmse_v": np.nan,
            "baseline_mae_v": np.nan,
            "corrected_mae_v": np.nan,
            "baseline_bias_v": np.nan,
            "corrected_bias_v": np.nan,
        }

    baseline_error = measured[mask] - baseline[mask]
    corrected_error = measured[mask] - corrected_prediction[mask]

    return {
        "samples": int(mask.sum()),
        "baseline_rmse_v": rmse(baseline_error),
        "corrected_rmse_v": rmse(corrected_error),
        "baseline_mae_v": float(np.mean(np.abs(baseline_error))),
        "corrected_mae_v": float(np.mean(np.abs(corrected_error))),
        "baseline_bias_v": float(np.mean(baseline_error)),
        "corrected_bias_v": float(np.mean(corrected_error)),
    }


def main():
    # -----------------------------------------------------
    # 1. Load training cycle 2
    # -----------------------------------------------------
    train = load_cycle(CYCLE_2_FILE, 2)

    train_mask = (
        (train["soc_bounded"] >= TRAIN_SOC_MIN)
        & (train["soc_bounded"] <= TRAIN_SOC_MAX)
    )

    train_fit = train.loc[train_mask].copy()

    if len(train_fit) < 20:
        raise ValueError(
            f"Only {len(train_fit)} training samples in the selected SOC range."
        )

    # Fit a second-order polynomial to residual versus SOC.
    # Center SOC to improve numerical conditioning.
    soc_center = 0.5
    x_train = train_fit["soc_bounded"].to_numpy() - soc_center
    residual_train = train_fit["residual_v"].to_numpy()

    coefficients = np.polyfit(x_train, residual_train, deg=2)

    # Freeze coefficients: never refit on independent cycles.
    train["correction_v"] = np.polyval(
        coefficients, train["soc_bounded"] - soc_center
    )
    train["corrected_prediction_v"] = (
        train["predicted_voltage_v"] + train["correction_v"]
    )

    # -----------------------------------------------------
    # 2. Evaluate on cycle 2 and independent test cycles
    # -----------------------------------------------------
    all_metrics = []
    all_predictions = []

    cycles = [(2, train)]

    for cycle_number, path in VALIDATION_FILES.items():
        cycles.append((cycle_number, load_cycle(path, cycle_number)))

    for cycle_number, df in cycles:
        df = df.copy()

        if cycle_number != 2:
            df["correction_v"] = np.polyval(
                coefficients, df["soc_bounded"] - soc_center
            )
            df["corrected_prediction_v"] = (
                df["predicted_voltage_v"] + df["correction_v"]
            )

        for region in ["all_active", "mid_discharge", "low_soc_knee"]:
            metrics = calculate_metrics(
                df, df["corrected_prediction_v"].to_numpy(), region
            )
            all_metrics.append({
                "cycle_number": cycle_number,
                "evaluation": (
                    "training_in_sample" if cycle_number == 2
                    else "independent_cycle"
                ),
                "region": region,
                **metrics,
            })

        df["cycle_number"] = cycle_number
        all_predictions.append(df)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_file = OUTPUT_DIR / "ocv_shape_correction_metrics.csv"
    predictions_file = OUTPUT_DIR / "ocv_shape_correction_predictions.csv"
    metrics_df.to_csv(metrics_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)

    # Save the frozen correction coefficients.
    coeff_df = pd.DataFrame([{
        "polynomial_degree": 2,
        "soc_center": soc_center,
        "training_soc_min": TRAIN_SOC_MIN,
        "training_soc_max": TRAIN_SOC_MAX,
        "coefficient_x2": coefficients[0],
        "coefficient_x1": coefficients[1],
        "coefficient_constant": coefficients[2],
        "training_samples": len(train_fit),
    }])
    coeff_file = OUTPUT_DIR / "frozen_correction_coefficients.csv"
    coeff_df.to_csv(coeff_file, index=False)

    # -----------------------------------------------------
    # 3. Plot the frozen correction and independent-cycle results
    # -----------------------------------------------------
    soc_grid = np.linspace(0.0, 1.0, 200)
    correction_grid = np.polyval(
        coefficients, soc_grid - soc_center
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(
        soc_grid * 100.0, correction_grid * 1000.0,
        label="Frozen cycle-2 correction"
    )
    axes[0].axvspan(
        TRAIN_SOC_MIN * 100.0, TRAIN_SOC_MAX * 100.0,
        alpha=0.15, label="Cycle-2 training SOC range"
    )
    axes[0].axhline(0.0, linewidth=1)
    axes[0].set_xlabel("SOC (%)")
    axes[0].set_ylabel("Correction (mV)")
    axes[0].set_title("Frozen SOC-dependent voltage correction")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    test_metrics = metrics_df[
        (metrics_df["evaluation"] == "independent_cycle")
        & (metrics_df["region"] == "all_active")
    ]

    x = np.arange(len(test_metrics))
    width = 0.36

    axes[1].bar(
        x - width / 2,
        test_metrics["baseline_rmse_v"] * 1000.0,
        width,
        label="Original model"
    )
    axes[1].bar(
        x + width / 2,
        test_metrics["corrected_rmse_v"] * 1000.0,
        width,
        label="Corrected model"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"Cycle {c}" for c in test_metrics["cycle_number"]]
    )
    axes[1].set_ylabel("Active-discharge RMSE (mV)")
    axes[1].set_title("Independent-cycle comparison")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plot_file = OUTPUT_DIR / "ocv_shape_correction_validation.png"
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)

    print("\nFROZEN OCV-SHAPED CORRECTION VALIDATION")
    print("======================================")
    print(f"Cycle-2 training samples: {len(train_fit)}")
    print(
        "Training SOC range: "
        f"{TRAIN_SOC_MIN:.0%} to {TRAIN_SOC_MAX:.0%}"
    )
    print("\nFrozen polynomial coefficients:")
    print(f"Quadratic: {coefficients[0]:.8f}")
    print(f"Linear:    {coefficients[1]:.8f}")
    print(f"Constant:  {coefficients[2]:.8f}")

    print("\nMetrics by cycle and region:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print(f"\nMetrics: {metrics_file}")
    print(f"Predictions: {predictions_file}")
    print(f"Coefficients: {coeff_file}")
    print(f"Plot: {plot_file}")

    print(
        "\nCAUTION: The correction is a diagnostic voltage correction, "
        "not a calibrated physical OCV-SOC curve. Cycle 2 results are "
        "in-sample. Cycles 4, 6, 10 and 20 must remain untouched during fitting."
    )


if __name__ == "__main__":
    main()
