
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "capacity_retention_trajectories.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results_capacity_comparison"
    / "rul_analysis"
    / "forecast_evaluation"
)

EOL_THRESHOLD_PCT = 80.0

# Evaluate predictions at several fractions of each cell's recorded life.
CALIBRATION_FRACTIONS = (0.30, 0.40, 0.50, 0.60)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required = {
        "cell_id",
        "discharge_index",
        "capacity_retention_pct",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values(["cell_id", "discharge_index"]).copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.reset_index(drop=True)

        x = cell_df["discharge_index"].to_numpy(dtype=float)
        y = cell_df["capacity_retention_pct"].to_numpy(dtype=float)

        crossing_positions = np.flatnonzero(y <= EOL_THRESHOLD_PCT)

        if len(crossing_positions) == 0:
            print(
                f"{cell_id}: no observed EOL crossing; skipping evaluation."
            )
            continue

        first_crossing_pos = int(crossing_positions[0])
        observed_eol_index = int(x[first_crossing_pos])

        print(
            f"\n{cell_id}: observed EOL index={observed_eol_index}, "
            f"retention={y[first_crossing_pos]:.2f}%"
        )

        for fraction in CALIBRATION_FRACTIONS:
            # Only use observations strictly before the observed EOL.
            train_end = int(np.floor(len(cell_df) * fraction))
            train_end = min(train_end, first_crossing_pos)

            if train_end < 10:
                continue

            x_train = x[:train_end].reshape(-1, 1)
            y_train = y[:train_end]

            model = LinearRegression()
            model.fit(x_train, y_train)

            slope = float(model.coef_[0])
            intercept = float(model.intercept_)

            if slope >= 0:
                predicted_eol = np.nan
                predicted_rul = np.nan
                status = "No downward degradation trend"
            else:
                predicted_eol_float = (
                    EOL_THRESHOLD_PCT - intercept
                ) / slope

                predicted_eol = int(np.ceil(predicted_eol_float))
                predicted_rul = max(
                    0, predicted_eol - int(x_train[-1, 0])
                )
                status = "Forecast crossing"

            actual_rul = observed_eol_index - int(x_train[-1, 0])

            results.append({
                "cell_id": cell_id,
                "calibration_fraction_of_record": fraction,
                "training_observations": train_end,
                "last_training_discharge_index": int(x_train[-1, 0]),
                "last_training_retention_pct": float(y_train[-1]),
                "observed_eol_index": observed_eol_index,
                "actual_rul_observations": actual_rul,
                "fitted_slope_pp_per_observation": slope,
                "predicted_eol_index": predicted_eol,
                "predicted_rul_observations": predicted_rul,
                "eol_index_error_observations": (
                    predicted_eol - observed_eol_index
                    if np.isfinite(predicted_eol)
                    else np.nan
                ),
                "status": status,
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNo forecasts could be calculated.")
        return

    output_path = OUTPUT_DIR / "rul_forecast_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nRUL FORECAST RESULTS")
    print(results_df.round(3).to_string(index=False))
    print(f"\nSaved results: {output_path}")

    valid = results_df.dropna(subset=["eol_index_error_observations"])
    if not valid.empty:
        print(
            "\nMean absolute EOL-index error: "
            f"{valid['eol_index_error_observations'].abs().mean():.2f} "
            "discharge observations"
        )

    print(
        "\nInterpretation: predictions are made from partial histories, "
        "but the observed EOL index is used as retrospective evaluation "
        "ground truth. These results do not establish performance on "
        "entirely unseen cells."
    )


if __name__ == "__main__":
    main()
