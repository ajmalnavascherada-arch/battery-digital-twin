# Stage 4 — Initial Thevenin ECM baseline

Run from the project root after regenerating Stage 1 outputs from the verified NASA file:
```powershell
python src/thevenin_ecm.py --signals data/processed_nasa_original/B0005_signals_long.csv --cycle 2
```

Outputs in `results_nasa_original/ecm/`:
- predicted/measured voltage CSV
- voltage comparison plot
- model report with fitted parameters and in-sample RMSE/MAE

This first pass selects the longest contiguous active-current segment and fits a constant OCV plus R0/R1/tau. It is an intentionally simplified baseline. Reported errors are training-segment fit metrics, not independent validation. Check the fit, parameters, current sign convention, and identifiability before extending it.
