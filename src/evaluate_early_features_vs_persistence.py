
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "B0005_fixed_time_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results_capacity_comparison"
    / "early_features_vs_persistence"
)

# Chronological evaluation settings
INITIAL_TRAIN_FRACTION = 0.70
HORIZON = 5
ORIGIN_STRIDE = 5

# Features available from the early part of a discharge.
FEATURES = [
    "first_300s_voltage_drop_V",
    "first_300s_temperature_rise_C",
]



def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    target_col = "relative_capacity_retention_pct"

    required = {
        "discharge_index",
        target_col,
        *FEATURES,
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["discharge_index"] = pd.to_numeric(
        df["discharge_index"], errors="coerce"
    )

    df = df.sort_values("discharge_index").reset_index(drop=True)

    for col in [target_col, *FEATURES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=["discharge_index", target_col, *FEATURES]
    ).reset_index(drop=True)

    if df["discharge_index"].duplicated().any():
        raise ValueError("Duplicate discharge_index values found.")

    # Use the relative retention values already calculated in the CSV.
    df["capacity_retention_pct"] = df[target_col]

    return df


def score(actual, predicted):
    residual = actual - predicted

    return {
        "MAE_pp": mean_absolute_error(actual, predicted),
        "RMSE_pp": np.sqrt(mean_squared_error(actual, predicted)),
        "mean_residual_pp": np.mean(residual),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    y = df["capacity_retention_pct"].to_numpy(dtype=float)
    X = df[FEATURES].to_numpy(dtype=float)

    n = len(df)
    initial_train_size = int(np.floor(n * INITIAL_TRAIN_FRACTION))

    if initial_train_size < 10 or initial_train_size + HORIZON > n:
        raise ValueError(
            f"Insufficient usable rows ({n}) for this evaluation setup."
        )

    print(f"Usable observations: {n}")
    print(f"Initial training history: {initial_train_size}")
    print(f"Forecast horizon: {HORIZON}")
    print(f"Forecast-origin stride: {ORIGIN_STRIDE}")

    predictions = []

    # At each origin, fit using past observations only.
    # The model forecasts the next HORIZON observations.
    for origin in range(
        initial_train_size,
        n - HORIZON + 1,
        ORIGIN_STRIDE,
    ):
        train_y = y[:origin]
        actual = y[origin:origin + HORIZON]

        # Baseline: latest measured retention carried forward.
        persistence_pred = np.full(HORIZON, train_y[-1], dtype=float)

        # Feature model: use early voltage and temperature measurements
        # from the training observations only.
        model = LinearRegression()
        model.fit(X[:origin], train_y)
        feature_pred = model.predict(X[origin:origin + HORIZON])

        for step in range(HORIZON):
            predictions.append({
                "origin_index": origin,
                "forecast_step": step + 1,
                "target_discharge_index": df.iloc[origin + step][
                    "discharge_index"
                ],
                "actual_retention_pct": actual[step],
                "persistence_prediction_pct": persistence_pred[step],
                "feature_model_prediction_pct": feature_pred[step],
                "persistence_residual_pp": (
                    actual[step] - persistence_pred[step]
                ),
                "feature_model_residual_pp": (
                    actual[step] - feature_pred[step]
                ),
            })

    pred_df = pd.DataFrame(predictions)

    # Pooled metrics across forecasted points. Some targets can appear in
    # multiple windows; interpret this as a rolling evaluation, not as
    # independent observations.
    summary_rows = []

    for model_name, pred_col, residual_col in [
        (
            "persistence",
            "persistence_prediction_pct",
            "persistence_residual_pp",
        ),
        (
            "early_voltage_temperature",
            "feature_model_prediction_pct",
            "feature_model_residual_pp",
        ),
    ]:
        actual = pred_df["actual_retention_pct"].to_numpy()
        predicted = pred_df[pred_col].to_numpy()

        metrics = score(actual, predicted)

        summary_rows.append({
            "model": model_name,
            "forecast_points": len(pred_df),
            "forecast_windows": pred_df["origin_index"].nunique(),
            **metrics,
        })

    summary = pd.DataFrame(summary_rows)

    # Compare paired absolute errors for the same forecast points.
    pred_df["persistence_absolute_error_pp"] = (
        pred_df["persistence_residual_pp"].abs()
    )
    pred_df["feature_absolute_error_pp"] = (
        pred_df["feature_model_residual_pp"].abs()
    )

    persistence_mae = pred_df["persistence_absolute_error_pp"].mean()
    feature_mae = pred_df["feature_absolute_error_pp"].mean()

    improvement_pct = (
        100.0 * (persistence_mae - feature_mae) / persistence_mae
        if persistence_mae > 0
        else np.nan
    )

    pred_path = OUTPUT_DIR / "predictions.csv"
    summary_path = OUTPUT_DIR / "summary_metrics.csv"

    pred_df.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nEARLY-FEATURE MODEL VS PERSISTENCE")
    print(summary.round(4).to_string(index=False))
    print(f"\nMAE improvement vs persistence: {improvement_pct:.2f}%")

    if improvement_pct > 0:
        print("The feature model has lower pooled MAE in this evaluation.")
    elif improvement_pct < 0:
        print("Persistence has lower pooled MAE in this evaluation.")
    else:
        print("Both models have equal pooled MAE in this evaluation.")

    print(f"\nSaved predictions: {pred_path}")
    print(f"Saved metrics: {summary_path}")


if __name__ == "__main__":
    main()
