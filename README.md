# Battery Digital Twin

A reproducible engineering project for lithium-ion battery state estimation, thermal prediction, degradation analysis, and operating-condition optimisation.

## Current milestone: Stage 1 — data inspection

**Do not claim model accuracy or experimental results until they have been computed from the selected dataset.** Keep raw source data unchanged and document preprocessing choices.

### 1. Environment (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Get the dataset

Start with the NASA Li-ion Battery Aging dataset:
https://catalog.data.gov/dataset/li-ion-battery-aging-datasets

Download the dataset from the source page and place the original `.mat` files in `data/raw/`. Do not commit large raw data files to Git.

### 3. Inspect the files

```powershell
python src/inspect_mat.py data/raw/your_file.mat
```

This prints the MATLAB file's top-level variables and nested structure so we can build a parser against the actual file format rather than guessing.

### 4. Project principles

- Preserve original data; processed data belongs in `data/processed/`.
- Record units, assumptions, missing-data handling, and filtering decisions.
- Split train/test data by cell or by time where appropriate to prevent leakage.
- Compare models with simple baselines and report metrics with units.
- Clearly label simulated data; never describe it as experimental data.

## Planned architecture

- `src/inspect_mat.py` — inspect source `.mat` structures
- `src/data_pipeline.py` — next: extract cycles and measurements
- `src/electrical_model.py` — planned equivalent-circuit model
- `src/thermal_model.py` — planned lumped thermal model
- `src/state_estimation.py` — planned SOC estimation
- `src/degradation_model.py` — planned SOH/degradation analysis
- `src/optimisation.py` — planned operating-condition comparisons
- `dashboard/app.py` — planned interactive dashboard
- `tests/` — tests for parsing and model behaviour
