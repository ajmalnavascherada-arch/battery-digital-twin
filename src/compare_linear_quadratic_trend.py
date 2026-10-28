
"""Compare linear and quadratic time trends on a chronological split."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "B0005_fixed_time_features.csv"
)

OUT_DIR = (
    ROOT / "results_capacity_comparison"
    / "b0005_feature_dataset"
    / "early_soh_baseline"
    / "polynomial_trend_comparison"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "relative_capacity_retention_pct"
TIME = "discharge_index"

df = (
    pd.read_csv(DATA_FILE)
    .sort_values(TIME)
    .reset_index(drop=True)
)

if df[[TARGET, TIME]].isna().any().any():
    raise ValueError("Missing values found in target or index.")

split = int(len(df) * 0.70)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

X_train = train[[TIME]]
X_test = test[[TIME]]
y_train = train[TARGET]
y_test = test[TARGET]

models = {
    "linear_trend": make_pipeline(
        StandardScaler(),
        LinearRegression(),
    ),
    "quadratic_trend": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    ),
}

results = []
predictions = pd.DataFrame({
    TIME: test[TIME].to_numpy(),
    "actual_retention_pct": y_test.to_numpy(),
})

for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    results.append({
        "model": name,
        "MAE_pp": mean_absolute_error(y_test, predicted),
        "RMSE_pp": mean_squared_error(y_test, predicted) ** 0.5,
        "R2": r2_score(y_test, predicted),
        "mean_residual_pp": (y_test.to_numpy() - predicted).mean(),
    })
    predictions[f"{name}_predicted_pct"] = predicted

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "metrics.csv", index=False)
predictions.to_csv(OUT_DIR / "predictions.csv", index=False)

print("Chronological split: first 70% train, last 30% test")
print("Training observations:", len(train))
print("Test observations:", len(test))
print("\nTREND MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    train[TIME], y_train,
    label="Training actual", marker=".", markersize=3
)
ax.plot(
    test[TIME], y_test,
    label="Test actual", marker=".", markersize=3
)
for name in models:
    ax.plot(
        test[TIME], predictions[f"{name}_predicted_pct"],
        label=f"{name} prediction"
    )
ax.axvline(test[TIME].iloc[0], linestyle="--", label="Train/test boundary")
ax.set_xlabel("Discharge observation index")
ax.set_ylabel("Relative capacity retention (%)")
ax.set_title("Linear vs quadratic time-trend prediction")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "trend_comparison.png", dpi=160)
plt.close(fig)

print("\nSaved results to:", OUT_DIR)
