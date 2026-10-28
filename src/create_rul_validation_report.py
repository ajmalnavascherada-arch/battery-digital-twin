
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "leave_one_cell_out"
)

PREDICTIONS_FILE = INPUT_DIR / "loco_predictions.csv"
SUMMARY_FILE = INPUT_DIR / "loco_summary.csv"

OUTPUT_DIR = INPUT_DIR / "final_report"


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(PREDICTIONS_FILE)
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(SUMMARY_FILE)

    predictions = pd.read_csv(PREDICTIONS_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    required = {
        "held_out_cell",
        "forecast_origin",
        "actual_eol_index",
        "method",
        "predicted_eol_index",
        "valid_prediction",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    if predictions["valid_prediction"].dtype != bool:
        predictions["valid_prediction"] = (
            predictions["valid_prediction"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )

    predictions["predicted_eol_index"] = pd.to_numeric(
        predictions["predicted_eol_index"], errors="coerce"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot the two evaluated methods for each held-out cell.
    cells = sorted(predictions["held_out_cell"].unique())
    fig, axes = plt.subplots(
        len(cells), 1,
        figsize=(10, 4 * len(cells)),
        squeeze=False,
    )

    for ax, cell in zip(axes[:, 0], cells):
        subset = predictions[predictions["held_out_cell"] == cell]
        actual_eol = int(subset["actual_eol_index"].iloc[0])

        ax.axhline(
            actual_eol,
            linestyle="--",
            label=f"Observed EOL = {actual_eol}",
        )

        for method, label in [
            ("full_history", "Full-history trend"),
            ("recent_30", "Recent-30 trend"),
        ]:
            method_df = (
                subset[subset["method"] == method]
                .sort_values("forecast_origin")
            )

            ax.plot(
                method_df["forecast_origin"],
                method_df["predicted_eol_index"],
                marker="o",
                label=label,
            )

        ax.set_title(f"Held-out cell: {cell}")
        ax.set_xlabel("Forecast origin (discharge observations available)")
        ax.set_ylabel("Predicted EOL observation index")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Leave-one-cell-out RUL validation", y=1.01)
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "loco_eol_predictions.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Export the summary as a readable CSV.
    summary_path = OUTPUT_DIR / "loco_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Generate a concise Markdown report from the observed results.
    lines = [
        "# Battery Digital Twin — RUL Validation Report",
        "",
        "## Objective",
        "",
        "Evaluate whether recent-30 linear trend extrapolation improves "
        "80% capacity-retention EOL predictions compared with a "
        "full-history linear trend.",
        "",
        "## Method",
        "",
        "- Dataset: existing NASA battery capacity-retention trajectories.",
        "- EOL definition: first observed discharge observation at or below "
        "80% relative capacity retention.",
        "- Validation: leave one cell out; choose the fitting window using "
        "the other cells only.",
        "- Candidate windows: 10, 20 and 30 observations.",
        "- Minimum training coverage for window selection: 80%.",
        "- Recent-30 was selected in all three folds.",
        "- Invalid predictions are excluded from MAE/RMSE but counted in "
        "coverage.",
        "",
        "## Held-out results",
        "",
        "| Held-out cell | Method | Origins | Valid predictions | Coverage (%) "
        "| MAE (observations) | RMSE (observations) | Bias (observations) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['held_out_cell']} "
            f"| {row['method']} "
            f"| {int(row['forecast_origins'])} "
            f"| {int(row['valid_predictions'])} "
            f"| {row['coverage_pct']:.2f} "
            f"| {row['MAE_eol_observations']:.2f} "
            f"| {row['RMSE_eol_observations']:.2f} "
            f"| {row['bias_eol_observations']:.2f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Recent-30 extrapolation substantially reduced EOL error for B0005 "
        "and B0007. Full-history extrapolation performed slightly better "
        "for B0006. The result does not support a universal winner.",
        "",
        "## Limitations",
        "",
        "- Only three battery cells were evaluated.",
        "- Forecasts were made at discrete origins before observed EOL.",
        "- MAE and RMSE exclude invalid predictions; coverage must be "
        "considered alongside error.",
        "- The minimum decline-rate screen of 0.05 percentage points per "
        "observation is exploratory, not a validated physical limit.",
        "- The observed 80% crossing is a retrospective reference and "
        "does not establish exact physical EOL timing.",
        "- Results do not establish generalization to unseen battery "
        "chemistries, cells or operating conditions.",
        "",
        "## Conclusion",
        "",
        "Recent-30 is a promising candidate for further validation, not a "
        "universally validated RUL model. Additional independent cells "
        "and prospective validation are needed before deployment claims.",
        "",
        f"Prediction plot: `{plot_path.name}`",
    ])

    report_path = OUTPUT_DIR / "RUL_validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nFINAL RUL VALIDATION REPORT CREATED")
    print("=" * 60)
    print(f"Plot: {plot_path}")
    print(f"Metrics CSV: {summary_path}")
    print(f"Markdown report: {report_path}")


if __name__ == "__main__":
    main()
