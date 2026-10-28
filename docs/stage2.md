# Stage 2 — Capacity fade baseline

Run after Stage 1:
```powershell
python src/analyse_capacity_fade.py
```

Outputs in `results/`:
- `B0005_discharge_capacity.csv`
- `B0005_capacity_fade.png`
- `B0005_capacity_report.txt`

This is a transparent first/last observed-capacity comparison plus a plot. It is not a degradation model or a remaining-useful-life prediction. Inspect the plot, data gaps, and NASA source metadata before making further claims.
