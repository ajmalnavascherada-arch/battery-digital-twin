# Battery Digital Twin: Li-Ion Battery Modelling, State Estimation & Remaining Useful Life Prediction

**A Python-based battery modelling and analytics project using NASA lithium-ion battery aging data**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Domain](https://img.shields.io/badge/Domain-Battery%20Engineering-orange)
![Modelling](https://img.shields.io/badge/Modelling-Electrothermal%20%7C%20ECM-green)
![Status](https://img.shields.io/badge/Status-Research%20%26%20Development-yellow)

## Overview

This project develops a modular, data-driven battery digital twin for analysing lithium-ion battery degradation, electrical behaviour, thermal response, state of charge (SOC), state of health (SOH), and remaining useful life (RUL).

The project uses the NASA lithium-ion battery aging dataset to investigate how battery performance evolves during repeated charge and discharge cycles. Python-based data processing, statistical analysis, mathematical modelling, and validation workflows are used to extract degradation indicators and evaluate predictive methods.

The long-term objective is to establish a reproducible framework that connects battery measurements with physics-based models and data-driven diagnostics.

**Current focus areas:**

- Battery capacity fade and state-of-health analysis
- Equivalent-circuit modelling using the Thevenin model
- State-of-charge estimation through coulomb counting
- Electrothermal behaviour and transient thermal response
- Remaining useful life estimation using capacity-retention trajectories
- Cross-cell evaluation and rolling-origin forecasting
- Automated data processing, diagnostics, and model validation

> **Research scope:** This repository is a battery modelling and analytics prototype. The implemented models have different levels of maturity, and the available results should not be interpreted as proof of a fully validated, real-time battery management system.

## Table of Contents

- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Processing Workflow](#data-processing-workflow)
- [Implemented Models](#implemented-models)
- [Remaining Useful Life Analysis](#remaining-useful-life-analysis)
- [Validation Results](#validation-results)
- [How to Reproduce the Analysis](#how-to-reproduce-the-analysis)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Skills Demonstrated](#skills-demonstrated)
- [References](#references)
- [License](#license)

## Project Objectives

The project addresses several practical questions in battery engineering.

1. **Capacity degradation:** How does measured discharge capacity change over successive cycles?
2. **State of health:** How can capacity retention be used to assess battery degradation?
3. **Electrical dynamics:** How accurately can an equivalent-circuit model reproduce measured terminal voltage?
4. **State estimation:** How can current measurements be integrated to estimate SOC?
5. **Thermal behaviour:** How can transient temperature measurements be analysed and approximated using mathematical models?
6. **Lifetime prediction:** How accurately can future end-of-life crossings be estimated from historical capacity observations?
7. **Model generalisation:** Do forecasting methods remain useful when evaluated on a battery cell excluded from model selection?

These objectives connect experimental data analysis with electrochemical energy storage, battery diagnostics, and predictive maintenance.

## Key Features

| Module | Description |
|---|---|
| Data pipeline | Extracts cycle-level and signal-level information from NASA `.mat` files |
| Capacity analysis | Calculates capacity fade and capacity-retention trajectories |
| Electrical modelling | Implements Thevenin equivalent-circuit models |
| SOC estimation | Uses coulomb counting to estimate battery charge state |
| Thermal analysis | Investigates transient heating, cooling, and temperature response |
| Feature analysis | Examines relationships between measured features and degradation |
| RUL forecasting | Evaluates trend-based capacity forecasting and end-of-life predictions |
| Cross-cell validation | Uses leave-one-cell-out evaluation to examine generalisation |
| Diagnostic tools | Provides scripts for investigating model residuals, parameter sensitivity, and data alignment |
| Automated testing | Includes tests for selected data-analysis and project functionality |

## Technology Stack

- **Programming:** Python
- **Numerical computing:** NumPy, SciPy
- **Data analysis:** pandas
- **Visualisation:** Matplotlib
- **Machine learning:** scikit-learn
- **Data format:** MATLAB `.mat`, CSV
- **Development environment:** PyCharm, PowerShell, Python virtual environments
- **Version control:** Git and GitHub
- **Testing:** pytest-compatible project tests

Dependencies are listed in [`requirements.txt`](requirements.txt).

## Repository Structure

The main components of the repository are organised as follows:

```text
battery-digital-twin/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                    # Local source datasets; not committed
│   ├── processed/              # Local processed data; not committed
│   └── processed_nasa_original/ # Additional local processed data
│
├── src/
│   ├── data_pipeline.py
│   ├── analyse_capacity_fade.py
│   ├── analyse_capacity_fade_generic.py
│   ├── analyse_rul_threshold.py
│   ├── evaluate_loco_rul.py
│   ├── audit_loco_rul.py
│   ├── create_rul_validation_report.py
│   ├── thevenin_ecm.py
│   ├── thevenin_ecm_soc.py
│   ├── thevenin_ecm_soc_voltage.py
│   ├── validate_ecm_cycle.py
│   ├── validate_ecm_multiple_cycles.py
│   └── ...                     # Additional analysis and diagnostic scripts
│
├── tests/
│   ├── test_capacity_analysis.py
│   └── test_project_scaffold.py
│
├── docs/
│   ├── stage2.md
│   ├── stage3.md
│   └── stage4.md
│
└── dashboard/
    └── .gitkeep
```

The structure above highlights the main entry points rather than every script. Consult the `src/` directory for the full collection of analysis, comparison, validation, and diagnostic workflows.

**Data management:** Raw NASA datasets and generated processed data are maintained locally and are not intended to be committed to Git. Some analysis scripts expect particular input and output directories, so check their command-line arguments before running them.

## Dataset

### NASA lithium-ion battery aging dataset

The project uses the NASA lithium-ion battery aging dataset, including battery-cell measurements for cells such as B0005, B0006, and B0007.

The data contains charge, discharge, and impedance records. Depending on the record type, the measurements can include electrical quantities, temperature, time, and capacity-related information.

The dataset supports investigation of:

- Capacity degradation over repeated cycling
- Charge and discharge behaviour
- Terminal voltage and current profiles
- Transient temperature behaviour
- Relationships between cycle history and battery health
- Observed capacity-based end-of-life crossings

### Data preparation

The pipeline extracts cycle metadata and signal measurements from MATLAB files and generates CSV outputs for downstream analysis.

For the original B0005 dataset, the reported processing results were:

| Metric | Result |
|---|---:|
| Extracted records | 616 |
| Impedance records | 278 |
| Charge records | 170 |
| Discharge records | 168 |
| Exported signal samples | 2,957,290 |
| Signals with missing time alignment | 0 |

These figures describe the processed B0005 file used in the reported workflow, not the complete NASA dataset.

### Obtaining the data

The dataset is available from NASA's Prognostics Center of Excellence.

- [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/)
- [NASA Open Data Portal](https://data.nasa.gov/)

Obtain the relevant battery-aging data from the original dataset distribution and place the files in the directory expected by the pipeline.

**Important:** The raw `.mat` files and large processed CSV files are not included in this GitHub repository. Users must obtain the source data separately.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ajmalnavascherada-arch/battery-digital-twin.git
cd battery-digital-twin
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, use an appropriate local execution-policy configuration or run the environment's Python executable directly.

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check the installation

```bash
python --version
pip list
```

Ensure that the Python interpreter selected in PyCharm points to the project's `.venv` environment.

## Data Processing Workflow

### Step 1: Prepare the raw dataset

Place the B0005 MATLAB file at:

```text
data/raw/nasa_original/B0005.mat
```

Confirm that the file exists before starting the pipeline.

### Step 2: Run the data pipeline

From the project root:

```bash
python src/data_pipeline.py data/raw/nasa_original/B0005.mat --out-dir data/processed_nasa_original
```

The pipeline extracts cycle-level information and signal-level measurements into processed outputs.

The precise filenames and optional arguments depend on the implementation of the script. The B0005 workflow has produced files including:

```text
data/processed_nasa_original/
├── B0005_cycle_summary.csv
└── B0005_signals_long.csv
```

The long-format signal CSV can be several hundred megabytes, depending on the source file and extraction process.

### Step 3: Analyse capacity fade

```bash
python src/analyse_capacity_fade.py --help
```

Check the script's supported arguments before running the analysis. The previously used workflow accepts a cycle-summary CSV and an output directory.

For example, the reported analysis used:

```text
Input:
data/processed_nasa_original/B0005_cycle_summary.csv

Output directory:
results_nasa_original/
```

The analysis calculates the first and last valid discharge capacities and the relative capacity change.

### Step 4: Run additional analyses

Examples of available analysis entry points include:

```bash
python src/audit_signal_alignment.py
python src/thevenin_ecm.py --help
python src/thevenin_ecm_soc.py --help
python src/evaluate_loco_rul.py --help
```

Check each script's arguments and expected files first. Some scripts are specialised for particular datasets or directory structures.

## Implemented Models

### 1. Capacity fade and state of health

Capacity fade is assessed using measured discharge capacity across successive discharge observations.

A basic capacity-retention measure is:

**Capacity Retention (%) = (Q<sub>k</sub> / Q<sub>ref</sub>) × 100**

Where:

- Q<sub>k</sub> is the discharge capacity at observation `k`.
- Q<sub>ref</sub> is the reference capacity used for normalisation.

State of health can be expressed using the same capacity ratio when the reference capacity is an appropriate nominal or beginning-of-life value. The chosen reference must be documented because it affects the resulting SOH estimate.

The repository includes capacity-fade analysis and threshold-crossing workflows.

### 2. Thevenin equivalent-circuit model

The Thevenin equivalent-circuit model represents the battery using an open-circuit voltage, an ohmic resistance, and one or more RC branches.

A single-RC model can be represented by:

- Open-circuit voltage, V<sub>OC</sub>
- Ohmic resistance, R<sub>0</sub>
- Polarisation resistance, R<sub>1</sub>
- Polarisation capacitance, C<sub>1</sub>

The terminal-voltage relationship is:

**V<sub>t</sub> = V<sub>OC</sub> − I × R<sub>0</sub> − V<sub>1</sub>**

This equation assumes a discharge-positive current convention, where V<sub>1</sub> represents the polarisation voltage.

The RC dynamics are:

**State of Charge (SOC) Estimation**

SOC is estimated using coulomb counting:

**SOC(t) = SOC(t₀) − (η / Q<sub>usable</sub>) × ∫ I(τ) dτ**

Where:
- `SOC(t)` is the estimated state of charge at time `t`.
- `SOC(t₀)` is the initial state of charge.
- Q<sub>usable</sub> is the usable battery capacity.
- `I(τ)` is the battery current as a function of time.
- `η` is the coulombic efficiency factor.

under the same current convention.

The implementation explores voltage reconstruction and parameter fitting. Experimental model variants also investigate SOC-dependent behaviour and alternative circuit structures.

### 3. State-of-charge estimation

Coulomb counting estimates SOC by integrating current over time:


**SOC(t) = SOC(t₀) − (η / Q<sub>usable</sub>) × ∫ₜ₀ᵗ I(τ) dτ**

Where:
- `SOC(t)` is the state of charge at time `t`.
- `SOC(t₀)` is the initial state of charge.
- `Q<sub>usable</sub>` is the usable battery capacity.
- `I(τ)` is the battery current at time `τ`.
- `η` is the coulombic efficiency factor.

Here, Q<sub>usable</sub> is the selected usable capacity and (τ) represents an efficiency factor. Current and capacity must use consistent units.

The result depends on the initial SOC, current measurement accuracy, sampling intervals, usable-capacity assumptions, and integration convention. Coulomb counting alone does not correct accumulated estimation error.

### 4. Thermal response analysis

Thermal analysis investigates measured temperature profiles, heating and cooling phases, temperature gradients, and differences between measured and modelled behaviour.

Available scripts include:

- `analyse_thermal_phases.py`
- `analyse_thermal_response.py`
- `analyse_post_discharge_cooling.py`
- `fit_thermal_response.py`
- `validate_thermal_response.py`
- `validate_hybrid_thermal_model.py`

These scripts provide analysis and validation tools for the thermal workflows. Not every model is a fully coupled electrochemical-electrothermal digital twin, and the assumptions of individual models should be reviewed before interpreting their outputs.

### 5. Remaining useful life forecasting

The RUL workflow uses capacity-retention trajectories to estimate the future point at which a cell reaches a defined end-of-life threshold.

In the current capacity-based analysis, the threshold is:

**Capacity Retention (%) = (Q<sub>k</sub> / Q<sub>ref</sub>) × 100**

Where:
- Q<sub>k</sub> is the discharge capacity at observation `k`.
- Q<sub>ref</sub> is the reference capacity.

The first observed discharge observation meeting this criterion is treated as the observed end-of-life crossing.

Forecasting methods include full-history and recent-window trend approaches, alongside comparison and validation scripts.

## Remaining Useful Life Analysis

### Leave-one-cell-out validation

To investigate generalisation between battery cells, the RUL evaluation uses a leave-one-cell-out strategy.

The procedure is:

1. Hold out one battery cell for evaluation.
2. Use the other two cells to select the recent-history window.
3. Evaluate forecasts at multiple historical origins.
4. Compare predicted end-of-life crossings with observed crossings.
5. Repeat the process for each of the three cells.
6. Audit the prediction records and reproduce the reported metrics.

The recent-window candidate set includes 10, 20, and 30 discharge observations. The reported selection procedure chose the 30-observation window in all three held-out-cell folds.

This is a small cross-cell experiment. It provides evidence about these cells and this dataset, not a universal conclusion about battery lifetime prediction.

### Evaluation metrics

The main metrics are:

- **MAE:** Mean absolute end-of-life prediction error.
- **RMSE:** Root mean squared end-of-life prediction error.
- **Bias:** Mean signed prediction error.
- **Coverage:** Fraction of evaluated forecast origins producing a valid prediction under the evaluation procedure.

The error metrics are expressed in **discharge-observation units**, not hours, days, or calendar time.

A positive bias in this report means that the predicted end-of-life observation tends to be later than the actual crossing, indicating overestimation of remaining life under the report's sign convention.

## Validation Results

### Capacity degradation

The following results were obtained from the processed NASA cell data:

| Cell | Discharge observations | First observed 80% retention crossing | Final capacity retention |
|---|---:|---:|---:|
| B0005 | 168 | 101 | Approximately 71.38% |
| B0006 | 163 | 59 | Approximately 57.73% |
| B0007 | 163 | 120 | Approximately 75.18% |

The first observed threshold crossing is not necessarily the exact physical end-of-life time. The table reports crossings in the available discharge observations.

### Leave-one-cell-out RUL evaluation

The following metrics were produced by the current evaluation workflow.

| Held-out cell | Method | Valid coverage | MAE (observations) | RMSE (observations) | Bias (observations) |
|---|---|---:|---:|---:|---:|
| B0005 | Full history | 86.67% | 65.92 | 93.16 | +65.92 |
| B0005 | Recent 30 | 80.00% | 17.02 | 27.00 | +11.81 |
| B0006 | Full history | 100.00% | 7.59 | 9.08 | +7.59 |
| B0006 | Recent 30 | 100.00% | 8.05 | 10.02 | +7.35 |
| B0007 | Full history | 88.89% | 43.76 | 78.32 | +39.14 |
| B0007 | Recent 30 | 83.33% | 19.16 | 31.84 | -0.02 |

### Interpretation

- The recent-30-observation method substantially reduced MAE and RMSE for B0005 and B0007 in this evaluation.
- Full-history forecasting performed slightly better on MAE and RMSE for B0006.
- The recent-window approach reduced the positive bias for B0005 and B0007.
- Neither method produced valid predictions at every forecast origin.
- The best method can vary by cell, and the small sample does not establish general performance on unseen battery populations.

### Validation audit

The independent RUL audit checked the saved prediction records and reproduced the summary metrics:

- Observed end-of-life crossings were verified for all three cells.
- 78 prediction rows were checked.
- No prediction-row consistency failures were found.
- The reported summary metrics and coverage values were reproduced.
- The audit completed with a `PASS` status.

This audit establishes consistency between the saved prediction records and the reported calculations. It does not independently establish that the modelling assumptions are physically correct or that the model will generalise to other battery chemistries.

## How to Reproduce the Analysis

The repository contains dedicated scripts for processing, analysis, validation, and reporting.

### RUL validation

Inspect the supported arguments:

```bash
python src/evaluate_loco_rul.py --help
```

Run the evaluation using the input paths and options supported by the script.

The main related scripts are:

```text
src/evaluate_loco_rul.py
src/audit_loco_rul.py
src/compare_loco_selection_weights.py
src/create_rul_validation_report.py
```

The validation workflow has produced a report directory containing:

```text
results_capacity_comparison/
└── rul_analysis/
    └── leave_one_cell_out/
        └── final_report/
            ├── loco_eol_predictions.png
            ├── loco_metrics_summary.csv
            └── RUL_validation_report.md
```

These files are generated analysis outputs and are not included in the initial GitHub commit. To reproduce the report, ensure the required input data and expected intermediate files are available.

### Automated tests

Install the dependencies, then run:

```bash
python -m pytest
```

The repository currently includes tests for selected capacity-analysis and project-scaffold functionality. A successful test run does not imply that every experimental model has been independently validated.

## Limitations

The following limitations are important when interpreting the results.

1. **Limited number of cells:** The reported RUL comparison uses three NASA battery cells. This is insufficient to establish robust generalisation across manufacturers, chemistries, formats, temperatures, and operating conditions.
2. **Observation-based lifetime:** The current end-of-life definition uses a capacity-retention threshold at observed discharge cycles. It does not directly estimate calendar life.
3. **Model dependence:** Forecast accuracy depends on the trend method, historical window, valid prediction coverage, and threshold definition.
4. **ECM parameter identifiability:** Equivalent-circuit parameters may be difficult to identify uniquely from limited or poorly excited data. Parameter bounds and fitted values should be examined before assigning physical meaning.
5. **SOC uncertainty:** Coulomb counting depends on initial SOC, current integration, and the assumed usable capacity.
6. **Thermal model assumptions:** A simplified thermal response model may not capture every heat-generation mechanism or spatial temperature gradient.
7. **Retrospective validation:** The reported results are based on historical data. They do not demonstrate real-time performance in a deployed battery management system.
8. **Reproducibility requirements:** Users need the original dataset, compatible dependencies, and the correct input paths to reproduce all analyses.
9. **Experimental scripts:** Some files are exploratory diagnostics or alternative model implementations. Their presence does not mean that every approach is production-ready.

## Future Work

Planned development directions include:

- Develop and validate SOC-dependent open-circuit-voltage relationships.
- Improve ECM parameter identifiability and evaluate parameters across multiple cycles.
- Compare SOC estimation against appropriate reference measurements.
- Extend RUL validation to additional cells and independent datasets.
- Investigate uncertainty intervals for end-of-life predictions.
- Compare trend-based forecasts with additional data-driven approaches.
- Improve automated tests and configuration of analysis pipelines.
- Add reproducible example outputs and selected figures to the repository.
- Develop a lightweight dashboard for visualising capacity fade, voltage behaviour, thermal response, and RUL predictions.
- Explore integration of electrical and thermal models into a more tightly coupled battery digital twin.

These items describe development directions, not completed capabilities.

## Skills Demonstrated

This project demonstrates practical work in:

- Battery ageing and capacity-degradation analysis
- Lithium-ion battery data processing
- Python scientific computing
- Time-series analysis and forecasting
- Equivalent-circuit modelling
- SOC and SOH estimation concepts
- Transient thermal-response analysis
- Cross-validation and forecast evaluation
- Model diagnostics and parameter sensitivity
- Automated data pipelines and reproducible research
- Git-based version control

The project is relevant to battery testing, battery systems engineering, energy storage, electrochemical engineering, predictive maintenance, and research involving battery state estimation and lifetime prediction.

## References

1. NASA Ames Research Center, *Prognostics Center of Excellence*. [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/)
2. NASA, *Open Data Portal*. [NASA Data](https://data.nasa.gov/)

Consult the original dataset documentation for details about the experimental setup, data fields, test conditions, and permitted usage.

## License

No explicit open-source license is currently specified in this repository.

Until a license is added, do not assume that other users have permission to redistribute or reuse the code. Consider adding an appropriate license if you intend to make the software openly reusable, and check the terms governing the source dataset separately.

---

## Author

**Ajmal Navas Cherada**

M.Sc. Chemical & Energy Engineering  
Otto von Guericke University Magdeburg, Germany

GitHub: [@ajmalnavascherada-arch](https://github.com/ajmalnavascherada-arch)

**Project repository:** [battery-digital-twin](https://github.com/ajmalnavascherada-arch/battery-digital-twin)

*This project is developed for engineering analysis, modelling practice, and research into lithium-ion battery behaviour. It is not intended to replace a safety-qualified battery management system.*
