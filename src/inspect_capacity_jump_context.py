
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results_capacity_comparison" / "combined_capacity_dataset.csv"

TARGETS = {
    "B0005": [313],
    "B0006": [38, 142, 302],
}

df = pd.read_csv(DATA)

for cell, target_cycles in TARGETS.items():
    group = (
        df[df["cell_id"] == cell]
        .sort_values("cycle_number")
        .reset_index(drop=True)
    )

    for cycle in target_cycles:
        matches = group.index[group["cycle_number"] == cycle].tolist()
        if not matches:
            print(f"\n{cell}, cycle {cycle}: not found")
            continue

        i = matches[0]
        start = max(0, i - 10)
        end = min(len(group), i + 11)

        print(f"\n{'=' * 65}")
        print(f"{cell}: context around recorded cycle {cycle}")
        print(
            group.loc[
                start:end - 1,
                ["cycle_number", "capacity_Ah", "ambient_temperature_C"],
            ].to_string(index=False)
        )

        print(
            "\nAmbient temperature values in window:",
            sorted(
                group.loc[start:end - 1, "ambient_temperature_C"]
                .dropna().unique().tolist()
            ),
        )
