
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
    / "trend_method_comparison"
)

EOL_THRESHOLD_PCT = 80.0
RECENT_WINDOW = 20

# Fixed discharge-observation indices at which to make predictions.
# Only origins strictly before the observed EOL are evaluated.
FORECAST_ORIGINS = (30, 40, 50, 60, 70, 80, 90, 100)


def predict_eol(train_x, train_y):
    """Fit a linear trend and extrapolate its 80% threshold crossing."""
    model = LinearRegression()
    model.fit(train_x.reshape(-1, 1), train_y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    if slope >= 0:
        return np.nan, slope, "No downward trend"

    crossing = (EOL_THRESHOLD_PCT - intercept) / slope

    if not np.isfinite(crossing):
        return np.nan, slope, "Invalid forecast"

    return int(np.ceil(crossing)), slope, "Forecast crossing"


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

    rows = []

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.reset_index(drop=True)

        x = cell_df["discharge_index"].to_numpy(dtype=float)
        y = cell_df["capacity_retention_pct"].to_numpy(dtype=float)

        crossings = np.flatnonzero(y <= EOL_THRESHOLD_PCT)

        if len(crossings) == 0:
            print(f"{cell_id}: no observed EOL crossing; skipped.")
            continue

        observed_eol = int(x[crossings[0]])

        for origin in FORECAST_ORIGINS:
            # Only observations through this origin are available.
            # Require the origin to precede observed EOL.
            if origin >= observed_eol:
                continue

            train_mask = x <= origin
            x_train = x[train_mask]
            y_train = y[train_mask]

            if len(x_train) < RECENT_WINDOW:
                continue

            full_eol, full_slope, full_status = predict_eol(
                x_train, y_train
            )

            recent_eol, recent_slope, recent_status = predict_eol(
                x_train[-RECENT_WINDOW:],
                y_train[-RECENT_WINDOW:],
            )

            for method, predicted_eol, slope, status in [
                (
                    "full_history",
                    full_eol,
                    full_slope,
                    full_status,
                ),
                (
                    "recent_20_observations",
                    recent_eol,
                    recent_slope,
                    recent_status,
                ),
            ]:
                predicted_rul = (
                    predicted_eol - origin
                    if np.isfinite(predicted_eol)
                    else np.nan
                )

                rows.append({
                    "cell_id": cell_id,
                    "forecast_origin": origin,
                    "history_observations": len(x_train),
                    "observed_eol_index": observed_eol,
                    "actual_rul_observations": observed_eol - origin,
                    "method": method,
                    "fitted_slope_pp_per_observation": slope,
                    "predicted_eol_index": predicted_eol,
                    "predicted_rul_observations": predicted_rul,
                    "eol_index_error": (
                        predicted_eol - observed_eol
                        if np.isfinite(predicted_eol)
                        else np.nan
                    ),
                    "status": status,
                })

    results = pd.DataFrame(rows)

    if results.empty:
        print("No eligible forecasts were produced.")
        return

    results_path = OUTPUT_DIR / "rul_predictions.csv"
    results.to_csv(results_path, index=False)

    print("FIXED-ORIGIN RUL FORECASTS")
    print(results.round(3).to_string(index=False))

    print("\nSUMMARY BY METHOD")
    summary_rows = []

    for method, group in results.groupby("method"):
        valid = group.dropna(subset=["eol_index_error"])

        summary_rows.append({
            "method": method,
            "eligible_forecasts": len(group),
            "valid_forecasts": len(valid),
            "MAE_eol_index": (
                valid["eol_index_error"].abs().mean()
                if len(valid) else np.nan
            ),
            "mean_bias_eol_index": (
                valid["eol_index_error"].mean()
                if len(valid) else np.nan
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "rul_method_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary.round(3).to_string(index=False))
    print(f"\nSaved predictions: {results_path}")
    print(f"Saved summary: {summary_path}")

    print(
        "\nCaution: these are retrospective tests on three recorded cells. "
        "The threshold crossing is known from the complete trajectory for "
        "evaluation only; the fitted forecasts use historical observations "
        "through each forecast origin."
    )


if __name__ == "__main__":
    main()
