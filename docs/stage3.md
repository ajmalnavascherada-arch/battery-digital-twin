# Stage 3 — Signal alignment audit

Run from the project root:
```powershell
python src/audit_signal_alignment.py
```

Outputs:
- `results/B0005_signal_alignment_audit.csv`
- `results/B0005_signal_alignment_report.txt`
- `results/B0005_first_discharge_signals_by_sample.png`

This is a prerequisite to voltage prediction. It checks whether voltage, current, and temperature have matching sample counts and trustworthy time vectors. It does not interpolate or truncate signals. Candidate cycles still need physical review before fitting an equivalent-circuit model.
