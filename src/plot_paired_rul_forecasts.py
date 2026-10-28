
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "paired_trend_comparison"
    / "paired_rul_predictions.csv"
)

OUTPUT_DIR = INPUT_FILE.parent


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Paired forecast file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "cell_id",
        "forecast_origin",
        "method",
        "observed_eol_index",
        "predicted_eol_index",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    cells = ["B0005", "B0006", "B0007"]

    fig, axes = plt.subplots(
        len(cells), 1, figsize=(11, 12), sharex=False
    )

    for ax, cell_id in zip(axes, cells):
        cell_df = df[df["cell_id"] == cell_id].copy()

        if cell_df.empty:
            ax.set_title(f"{cell_id}: no paired forecasts")
            continue

        # Each method should have one prediction per forecast origin.
        for method, label in [
            ("full_history", "Full-history trend"),
            ("recent_20", "Recent-20 trend"),
        ]:
            method_df = (
                cell_df[cell_df["method"] == method]
                .sort_values("forecast_origin")
            )

            ax.plot(
                method_df["forecast_origin"],
                method_df["predicted_eol_index"],
                marker="o",
                label=label,
            )

        observed_eol = cell_df["observed_eol_index"].iloc[0]

        ax.axhline(
            observed_eol,
            linestyle="--",
            label=f"Observed EOL = {observed_eol}",
        )

        ax.plot(
            cell_df["forecast_origin"].drop_duplicates(),
            cell_df["forecast_origin"].drop_duplicates(),
            linestyle=":",
            label="Forecast origin",
        )

        ax.set_title(cell_id)
        ax.set_xlabel("Forecast origin (discharge index)")
        ax.set_ylabel("Predicted EOL index")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Paired RUL forecasts: full-history vs recent-20 trend",
        y=1.01,
    )
    fig.tight_layout()

    output_path = OUTPUT_DIR / "paired_rul_forecasts.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
