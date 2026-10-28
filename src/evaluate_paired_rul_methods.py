
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
    / "paired_trend_comparison"
)

EOL_THRESHOLD_PCT = 80.0
RECENT_WINDOW = 20
FORECAST_ORIGINS = (30, 40, 50, 60, 70, 80, 90, 100)


def forecast_eol(x_train, y_train):
    model = LinearRegression()
    model.fit(np.asarray(x_train).reshape(-1, 1), y_train)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    if slope >= 0:
        return np.nan, slope

    crossing = (EOL_THRESHOLD_PCT - intercept) / slope

    if not np.isfinite(crossing):
        return np.nan, slope

    return int(np.ceil(crossing)), slope


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

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

    rows = []

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.reset_index(drop=True)

        x = cell_df["discharge_index"].to_numpy(dtype=float)
        y = cell_df["capacity_retention_pct"].to_numpy(dtype=float)

        crossing_positions = np.flatnonzero(y <= EOL_THRESHOLD_PCT)

        if len(crossing_positions) == 0:
            continue

        observed_eol = int(x[crossing_positions[0]])

        for origin in FORECAST_ORIGINS:
            # Require a genuinely pre-EOL forecast origin.
            if origin >= observed_eol:
                continue

            available = x <= origin
            x_train = x[available]
            y_train = y[available]

            if len(x_train) < RECENT_WINDOW:
                continue

            predictions = {
                "full_history": forecast_eol(x_train, y_train),
                "recent_20": forecast_eol(
                    x_train[-RECENT_WINDOW:],
                    y_train[-RECENT_WINDOW:],
                ),
            }

            # Keep the origin only if BOTH methods forecast a crossing.
            if any(
                not np.isfinite(result[0])
                for result in predictions.values()
            ):
                continue

            for method, (predicted_eol, slope) in predictions.items():
                rows.append({
                    "cell_id": cell_id,
                    "forecast_origin": origin,
                    "method": method,
                    "observed_eol_index": observed_eol,
                    "actual_rul_observations": observed_eol - origin,
                    "predicted_eol_index": int(predicted_eol),
                    "predicted_rul_observations": int(predicted_eol) - origin,
                    "eol_error_observations": int(predicted_eol) - observed_eol,
                    "absolute_eol_error": abs(
                        int(predicted_eol) - observed_eol
                    ),
                    "fitted_slope_pp_per_observation": slope,
                })

    results = pd.DataFrame(rows)

    if results.empty:
        print("No shared valid forecast origins were found.")
        return

    # Confirm that both methods are present for every paired origin.
    counts = results.groupby(["cell_id", "forecast_origin"])["method"].nunique()
    paired_keys = counts[counts == 2].index

    paired_index = pd.MultiIndex.from_frame(
        results[["cell_id", "forecast_origin"]]
    )
    results = results.loc[paired_index.isin(paired_keys)].copy()

    detailed_path = OUTPUT_DIR / "paired_rul_predictions.csv"
    results.to_csv(detailed_path, index=False)

    summary_rows = []

    for (cell_id, method), group in results.groupby(["cell_id", "method"]):
        summary_rows.append({
            "cell_id": cell_id,
            "method": method,
            "shared_forecast_origins": group["forecast_origin"].nunique(),
            "MAE_eol_index": group["absolute_eol_error"].mean(),
            "mean_bias_eol_index": group["eol_error_observations"].mean(),
        })

    by_cell = pd.DataFrame(summary_rows)

    overall_rows = []
    for method, group in results.groupby("method"):
        overall_rows.append({
            "cell_id": "ALL_CELLS",
            "method": method,
            "shared_forecast_origins": group[
                ["cell_id", "forecast_origin"]
            ].drop_duplicates().shape[0],
            "MAE_eol_index": group["absolute_eol_error"].mean(),
            "mean_bias_eol_index": group["eol_error_observations"].mean(),
        })

    summary = pd.concat(
        [by_cell, pd.DataFrame(overall_rows)],
        ignore_index=True,
    )

    summary_path = OUTPUT_DIR / "paired_rul_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("PAIRED RUL METHOD COMPARISON")
    print(summary.round(3).to_string(index=False))

    print("\nSHARED FORECAST ORIGINS")
    print(
        results[["cell_id", "forecast_origin"]]
        .drop_duplicates()
        .sort_values(["cell_id", "forecast_origin"])
        .to_string(index=False)
    )

    print(f"\nSaved paired forecasts: {detailed_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
