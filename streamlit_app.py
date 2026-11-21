"""
PocketPilot deployment smoke test (temporary page).

Research and educational prototype for estimating daily smartphone behavior patterns,
battery consumption, mobile data usage, screen time, and budget-overrun risks.
"""

from pathlib import Path
import sys

# Insert src folder into sys.path before importing pocketpilot
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd
import streamlit as st

from pocketpilot import config
from pocketpilot.features.input_modes import get_feature_columns
from pocketpilot.inference.validation import validate_inputs
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP
from pocketpilot.risk.probabilistic_risk import assess_budget


st.set_page_config(
    page_title="PocketPilot Smoke Test",
    page_icon="📱",
    layout="wide",
)

st.title("PocketPilot deployment smoke test (temporary page)")

# Disclaimer banner verbatim
st.info(
    "PocketPilot is a research and educational prototype trained on aggregated public "
    "mobile-usage records. It estimates daily behavior patterns and should not be "
    "interpreted as an operating-system-level battery monitor."
)


@st.cache_resource
def load_all_artifacts() -> dict[str, any]:
    """
    Load all serialized ML pipelines and risk estimation dictionaries.

    Returns
    -------
    dict[str, any]
        Dictionary with keys 'models', 'clustering', 'risk', and 'status'.
    """
    artifacts_status = []

    # 1. Regression models
    models = {}
    for task_name, model_path in config.MODEL_PATHS.items():
        size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0.0
        loaded_obj = joblib.load(model_path)
        models[task_name] = loaded_obj
        artifacts_status.append({
            "Artifact Name": f"{task_name}_model.joblib",
            "Loaded OK": True,
            "File Size (MB)": round(size_mb, 4),
        })

    # 2. Clustering model
    cl_path = config.CLUSTERING_MODEL_PATH
    cl_size = cl_path.stat().st_size / (1024 * 1024) if cl_path.exists() else 0.0
    clustering = joblib.load(cl_path)
    artifacts_status.append({
        "Artifact Name": "clustering_model.joblib",
        "Loaded OK": True,
        "File Size (MB)": round(cl_size, 4),
    })

    # 3. Risk model
    risk_path = config.RISK_MODEL_PATH
    risk_size = risk_path.stat().st_size / (1024 * 1024) if risk_path.exists() else 0.0
    risk = joblib.load(risk_path)
    artifacts_status.append({
        "Artifact Name": "risk_model.joblib",
        "Loaded OK": True,
        "File Size (MB)": round(risk_size, 4),
    })

    return {
        "models": models,
        "clustering": clustering,
        "risk": risk,
        "status": pd.DataFrame(artifacts_status),
    }


# Load artifacts
artifacts = load_all_artifacts()

with st.expander("Model Artifacts Status", expanded=True):
    st.dataframe(artifacts["status"], hide_index=True, use_container_width=True)

st.subheader("User Inputs & Operational Budgets")

col1, col2 = st.columns(2)

with col1:
    device_model = st.selectbox("Device Model", options=config.KNOWN_DEVICES, index=0)
    app_usage_min = st.number_input(
        "App Usage Time (min/day)",
        min_value=1,
        max_value=1440,
        value=240,
        step=10,
        help="Daily total minutes spent actively interacting with apps (1 to 1440 min).",
    )
    screen_on_hours = st.number_input(
        "Expected Screen-On Time (hours/day)",
        min_value=0.1,
        max_value=24.0,
        value=4.5,
        step=0.1,
        format="%.1f",
        help="Anticipated hours of active screen illumination.",
    )
    apps_installed = st.number_input(
        "Number of Apps Installed",
        min_value=0,
        max_value=500,
        value=35,
        step=1,
        help="Total number of applications installed on the device (0 to 500).",
    )

with col2:
    battery_budget = st.number_input(
        "Battery Drain Budget (mAh)",
        min_value=1.0,
        value=2500.0,
        step=100.0,
        format="%.1f",
        help="Maximum acceptable daily battery consumption in mAh.",
    )
    data_budget = st.number_input(
        "Mobile Data Budget (MB)",
        min_value=1.0,
        value=1000.0,
        step=50.0,
        format="%.1f",
        help="Maximum acceptable daily mobile data usage in MB.",
    )
    screen_budget = st.number_input(
        "Screen Time Budget (hours)",
        min_value=0.1,
        value=5.0,
        step=0.5,
        format="%.1f",
        help="Maximum acceptable daily screen-on time in hours.",
    )

st.info(
    "Note: The screen time prediction does not use the screen-on hours input, "
    "it is estimated from app usage, apps installed and device model."
)

if st.button("Run prediction", type="primary"):
    errors = validate_inputs(
        device_model=device_model,
        app_usage_min=app_usage_min,
        screen_on_hours=screen_on_hours,
        apps_installed=apps_installed,
        battery_budget=battery_budget,
        data_budget=data_budget,
        screen_budget=screen_budget,
    )

    if errors:
        for err in errors:
            st.error(err)
    else:
        # Construct raw inputs DataFrames
        # For battery and data: device_model, app_usage_min, screen_on_hours, apps_installed
        raw_full = pd.DataFrame([{
            "device_model": device_model,
            "app_usage_min": float(app_usage_min),
            "screen_on_hours": float(screen_on_hours),
            "apps_installed": int(apps_installed),
        }])

        # For screen_time: device_model, app_usage_min, apps_installed (no screen_on_hours)
        raw_screen = pd.DataFrame([{
            "device_model": device_model,
            "app_usage_min": float(app_usage_min),
            "apps_installed": int(apps_installed),
        }])

        # Predict battery
        battery_pipe = artifacts["models"]["battery"]
        battery_pred = float(battery_pipe.predict(raw_full[TASK_RAW_INPUTS_MAP["battery"]])[0])
        battery_risk = assess_budget(artifacts["risk"]["battery"], battery_pred, float(battery_budget))

        # Predict data
        data_pipe = artifacts["models"]["data"]
        data_pred = float(data_pipe.predict(raw_full[TASK_RAW_INPUTS_MAP["data"]])[0])
        data_risk = assess_budget(artifacts["risk"]["data"], data_pred, float(data_budget))

        # Predict screen time
        screen_pipe = artifacts["models"]["screen_time"]
        screen_pred = float(screen_pipe.predict(raw_screen[TASK_RAW_INPUTS_MAP["screen_time"]])[0])
        screen_risk = assess_budget(artifacts["risk"]["screen_time"], screen_pred, float(screen_budget))

        st.subheader("Prediction Results & Risk Assessment")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(label="Predicted Battery Drain", value=f"{battery_pred:.1f} mAh")
            st.write(f"**Budget:** {battery_budget:.1f} mAh")
            st.write(f"**Risk Level:** `{battery_risk['level']}`")
            st.write(f"**P(overrun):** {battery_risk['p_overrun'] * 100:.1f}%")
            int80_b = battery_risk["interval_80"]
            st.write(f"**80% Interval:** [{int80_b[0]:.1f}, {int80_b[1]:.1f}] mAh")

        with res_col2:
            st.metric(label="Predicted Data Usage", value=f"{data_pred:.1f} MB")
            st.write(f"**Budget:** {data_budget:.1f} MB")
            st.write(f"**Risk Level:** `{data_risk['level']}`")
            st.write(f"**P(overrun):** {data_risk['p_overrun'] * 100:.1f}%")
            int80_d = data_risk["interval_80"]
            st.write(f"**80% Interval:** [{int80_d[0]:.1f}, {int80_d[1]:.1f}] MB")

        with res_col3:
            st.metric(label="Predicted Screen Time", value=f"{screen_pred:.2f} hours")
            st.write(f"**Budget:** {screen_budget:.1f} hours")
            st.write(f"**Risk Level:** `{screen_risk['level']}`")
            st.write(f"**P(overrun):** {screen_risk['p_overrun'] * 100:.1f}%")
            int80_s = screen_risk["interval_80"]
            st.write(f"**80% Interval:** [{int80_s[0]:.2f}, {int80_s[1]:.2f}] hours")
