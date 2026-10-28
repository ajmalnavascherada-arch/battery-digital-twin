
"""Diagnose residuals from the chronological time-trend model."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    / "time_trend_diagnostics"
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
    raise ValueError("Missing values found in target or discharge index.")

split = int(len(df) * 0.70)
train = df.iloc[:split]
test = df.iloc[split:]

model = LinearRegression()
model.fit(train[[TIME]], train[TARGET])

df["predicted_retention_pct"] = model.predict(df[[TIME]])
df["residual_pp"] = (
    df[TARGET] - df["predicted_retention_pct"]
)
df["period"] = ["train"] * split + ["test"] * (len(df) - split)

print("Time-trend residual definition: actual minus predicted")
print("Training observations:", len(train))
print("Test observations:", len(test))
print("\nMETRICS BY PERIOD")

for period, group in df.groupby("period", sort=False):
    print(f"\n{period.upper()}")
    print("MAE:", round(
        mean_absolute_error(
            group[TARGET], group["predicted_retention_pct"]
        ), 4
    ))
    print("RMSE:", round(
        mean_squared_error(
            group[TARGET], group["predicted_retention_pct"]
        ) ** 0.5, 4
    ))
    print("Mean residual:", round(group["residual_pp"].mean(), 4))
    print("Residual range:", round(group["residual_pp"].min(), 4),
          "to", round(group["residual_pp"].max(), 4))

df.to_csv(OUT_DIR / "time_trend_residuals.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    df[TIME], df[TARGET],
    marker=".", label="Actual retention"
)
ax.plot(
    df[TIME], df["predicted_retention_pct"],
    label="Time-trend prediction"
)
ax.axvline(
    df.iloc[split][TIME],
    linestyle="--",
    label="Train/test boundary"
)
ax.set_xlabel("Discharge observation index")
ax.set_ylabel("Relative capacity retention (%)")
ax.set_title("Time-trend model: actual vs predicted")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "trajectory.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
ax.axhline(0, linestyle="--")
ax.axvline(
    df.iloc[split][TIME],
    linestyle="--",
    label="Train/test boundary"
)
ax.plot(df[TIME], df["residual_pp"], marker=".")
ax.set_xlabel("Discharge observation index")
ax.set_ylabel("Residual (percentage points)")
ax.set_title("Time-trend residuals")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "residuals.png", dpi=160)
plt.close(fig)

print("\nSaved diagnostics to:", OUT_DIR)
