# PocketPilot

Explainable Machine-Learning System for Smartphone Battery Survival, Mobile-Data Budget, and Screen-Time Overrun Prediction.

## Overview
PocketPilot is a Streamlit-based explainable machine-learning application designed to:
- Predict daily battery drain (mAh) and daily mobile-data usage (MB) from phone usage behavior.
- Estimate total screen time.
- Cluster users into neutral digital usage profiles (K-Means / GMM).
- Estimate budget-overrun risk using threshold rules and probabilistic models (quantile regression, conformal prediction, or bootstrap).
- Run a personalized what-if planner that simulates behavior changes and finds the smallest adjustment meeting user budgets.
- Explain individual and global model predictions using SHAP (SHapley Additive exPlanations).

## Dataset
- **Primary dataset**: *Mobile Device Usage and User Behavior Dataset* (700 user records, 11 columns).
- **Extension dataset (optional)**: *Mobile Usage ScreenTime Dataset* (100+ user records with unlocks, notifications, and category splits).

## System Architecture & Prevention of Data Leakage
- **Forecast Mode**: Uses only information available at prediction time (e.g., current battery %, time elapsed, expected remaining usage). Never uses realized full-day battery drain or end-of-day data usage.
- **Profile-Analysis Mode**: Uses historical metrics for descriptive clustering and profile comparison. Never presented as a forecast.
- **Leakage Prevention**: All budget-overrun labels are derived strictly after the train/test split. Preprocessing pipelines (encoders/scalers) are fit strictly on training splits and applied identically during inference.

## Project Structure
```
PocketPilot/
├── README.md
├── requirements.txt
├── .gitignore
├── streamlit_app.py
├── .streamlit/
│   └── config.toml
├── app/
│   ├── pages/
│   │   ├── 1_Daily_Phone_Planner.py
│   │   ├── 2_Will_I_Make_It.py
│   │   ├── 3_Digital_Use_Profile.py
│   │   └── 4_Model_Evaluation.py
│   └── components/
│       ├── gauges.py
│       ├── input_forms.py
│       └── charts.py
└── src/
    └── pocketpilot/
        ├── __init__.py
        ├── config.py
        ├── data/
        │   ├── __init__.py
        │   ├── loader.py
        │   ├── cleaning.py
        │   └── splitting.py
        ├── features/
        │   ├── __init__.py
        │   ├── engineering.py
        │   ├── preprocessing.py
        │   └── input_modes.py
        ├── models/
        │   ├── __init__.py
        │   ├── battery_model.py
        │   ├── data_usage_model.py
        │   ├── screen_time_model.py
        │   ├── clustering.py
        │   └── model_zoo.py
        ├── risk/
        │   ├── __init__.py
        │   ├── threshold_risk.py
        │   └── probabilistic_risk.py
        ├── planner/
        │   ├── __init__.py
        │   ├── scenarios.py
        │   ├── action_cost.py
        │   └── optimizer.py
        ├── explain/
        │   ├── __init__.py
        │   └── shap_utils.py
        ├── evaluation/
        │   ├── __init__.py
        │   ├── regression_metrics.py
        │   ├── classification_metrics.py
        │   ├── clustering_metrics.py
        │   └── plots.py
        └── inference/
            ├── __init__.py
            └── predictor.py
```

## Setup & Running
```bash
# Clone and enter directory
cd PocketPilot

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit application
streamlit run streamlit_app.py
```
