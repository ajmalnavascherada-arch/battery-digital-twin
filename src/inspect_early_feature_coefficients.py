
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "B0005_fixed_time_features.csv"
)

FEATURES = [
    "first_300s_voltage_drop_V",
    "first_300s_temperature_rise_C",
]

TARGET = "relative_capacity_retention_pct"


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values("discharge_index").reset_index(drop=True)

    required = ["discharge_index", TARGET, *FEATURES]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    df = df.reset_index(drop=True)

    y = df[TARGET].to_numpy(dtype=float)
    X = df[FEATURES].to_numpy(dtype=float)

    n = len(df)
    initial_train_size = int(np.floor(0.70 * n))
    horizon = 5
    stride = 5

    rows = []

    for origin in range(initial_train_size, n - horizon + 1, stride):
        model = make_pipeline(
            StandardScaler(),
            LinearRegression(),
        )

        model.fit(X[:origin], y[:origin])

        regression = model.named_steps["linearregression"]
        scaler = model.named_steps["standardscaler"]

        rows.append({
            "origin_index": origin,
            "last_training_discharge_index": df.iloc[origin - 1][
                "discharge_index"
            ],
            "training_rows": origin,
            "intercept": regression.intercept_,
            "voltage_drop_coefficient": regression.coef_[0],
            "temperature_rise_coefficient": regression.coef_[1],
            "voltage_drop_mean": scaler.mean_[0],
            "voltage_drop_scale": scaler.scale_[0],
            "temperature_rise_mean": scaler.mean_[1],
            "temperature_rise_scale": scaler.scale_[1],
        })

    result = pd.DataFrame(rows)

    output_dir = (
        ROOT
        / "results_capacity_comparison"
        / "early_features_vs_persistence"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "feature_coefficients.csv"
    result.to_csv(output_path, index=False)

    print("FEATURE COEFFICIENT STABILITY")
    print(result.round(5).to_string(index=False))

    print("\nCoefficient ranges:")
    for column in [
        "voltage_drop_coefficient",
        "temperature_rise_coefficient",
    ]:
        print(
            f"{column}: "
            f"{result[column].min():.5f} to "
            f"{result[column].max():.5f}"
        )

    print(f"\nSaved coefficients: {output_path}")


if __name__ == "__main__":
    main()
