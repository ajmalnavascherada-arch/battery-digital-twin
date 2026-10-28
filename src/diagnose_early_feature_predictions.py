
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results_capacity_comparison"
    / "early_features_vs_persistence"
    / "predictions.csv"
)

OUTPUT_DIR = INPUT_FILE.parent


def main():
    df = pd.read_csv(INPUT_FILE)

    required = {
        "target_discharge_index",
        "actual_retention_pct",
        "persistence_prediction_pct",
        "feature_model_prediction_pct",
        "persistence_residual_pp",
        "feature_model_residual_pp",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Multiple rolling windows can predict the same target observation.
    # Average predictions per target index for a readable diagnostic plot.
    grouped = (
        df.groupby("target_discharge_index", as_index=False)
        .agg(
            actual=("actual_retention_pct", "mean"),
            persistence=("persistence_prediction_pct", "mean"),
            feature_model=("feature_model_prediction_pct", "mean"),
        )
        .sort_values("target_discharge_index")
    )

    grouped["persistence_error"] = (
        grouped["actual"] - grouped["persistence"]
    )
    grouped["feature_error"] = (
        grouped["actual"] - grouped["feature_model"]
    )

    print("DIAGNOSTIC SUMMARY")
    print(f"Unique target observations: {len(grouped)}")
    print(
        "Persistence MAE (aggregated targets): "
        f"{mean_absolute_error(grouped['actual'], grouped['persistence']):.4f} pp"
    )
    print(
        "Feature model MAE (aggregated targets): "
        f"{mean_absolute_error(grouped['actual'], grouped['feature_model']):.4f} pp"
    )
    print(
        "Feature-model residual range: "
        f"{grouped['feature_error'].min():.4f} to "
        f"{grouped['feature_error'].max():.4f} pp"
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(
        grouped["target_discharge_index"],
        grouped["actual"],
        label="Actual retention",
    )
    axes[0].plot(
        grouped["target_discharge_index"],
        grouped["persistence"],
        label="Persistence",
    )
    axes[0].plot(
        grouped["target_discharge_index"],
        grouped["feature_model"],
        label="Early voltage + temperature",
    )
    axes[0].set_ylabel("Capacity retention (%)")
    axes[0].set_title("B0005: Actual vs predicted capacity retention")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, linewidth=1)
    axes[1].plot(
        grouped["target_discharge_index"],
        grouped["persistence_error"],
        label="Persistence residual",
    )
    axes[1].plot(
        grouped["target_discharge_index"],
        grouped["feature_error"],
        label="Feature-model residual",
    )
    axes[1].set_xlabel("Discharge index")
    axes[1].set_ylabel("Actual − predicted (pp)")
    axes[1].set_title("Forecast residuals")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "prediction_diagnostics.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"\nSaved plot: {plot_path}")


if __name__ == "__main__":
    main()
