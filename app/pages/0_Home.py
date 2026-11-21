"""
Home overview page for PocketPilot.
"""

from pathlib import Path
import sys

# Bootstrap sys.path for standalone or router execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

from app.components.state import get_artifacts
from pocketpilot import config

DISCLAIMER_TEXT = (
    "PocketPilot is a research and educational prototype trained on aggregated public "
    "mobile-usage records. It estimates daily behavior patterns and should not be "
    "interpreted as an operating-system-level battery monitor."
)


def render_home_page() -> None:
    st.title("📱 PocketPilot")
    st.subheader("Intelligent Daily Smartphone Resource Planner & Survival Estimator")

    st.markdown(
        "**PocketPilot** helps smartphone users forecast daily battery drain, mobile data consumption, "
        "and screen time before they happen. By combining predictive machine learning pipelines, "
        "calibrated risk models, and an action-cost what-if optimizer, PocketPilot provides actionable, "
        "interpretable recommendations to prevent unexpected end-of-day battery depletion."
    )

    st.divider()

    st.markdown("### 🧭 Application Modules")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "- **Daily Phone Planner**: Full daily resource forecasting, calibrated overrun risk gauges, "
            "battery survival time estimation, and minimal-disruption what-if behavior remediation."
        )
        st.markdown(
            "- **Will I Make It?**: Goal-directed survival assistant assessing whether your remaining charge "
            "will last until a specified target hour, with actionable behavior change alternatives."
        )
    with col2:
        st.markdown(
            "- **Digital Use Profile**: Retrospective behavioral persona clustering and percentile benchmarking "
            "evaluating observed daily averages or multi-day CSV usage histories."
        )
        st.markdown(
            "- **Model Evaluation**: Transparent research reports, cross-validation metrics, SHAP beeswarm "
            "attributions, risk calibration curves, and testing inventories."
        )

    st.divider()

    st.markdown("### 🔬 Methodological Separation: Forecast vs. Profile-Analysis")
    st.markdown(
        "PocketPilot strictly decouples **forward-looking forecasts** from **retrospective profile analysis**. "
        "In **Forecast Mode** (Pages 1 & 2), models consume only pre-day known inputs and strictly exclude realized "
        "end-of-day metrics (`battery_mah`, `data_mb`, `behavior_class`) to prevent target leakage. "
        "In **Profile-Analysis Mode** (Page 3), retrospective observed usage metrics are clustered into neutral, "
        "non-judgmental personas for historical self-reflection. Profile classifications are never generated "
        "from forecast predictions."
    )

    st.divider()

    st.markdown("### ⚙️ System & Artifact Status")
    try:
        artifacts = get_artifacts()
        status_rows = []
        for task_key, path in config.MODEL_PATHS.items():
            sz = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
            model_type = type(artifacts["pipelines"][task_key].named_steps["model"]).__name__
            status_rows.append({
                "Artifact": f"{task_key.title()} Regression Pipeline",
                "Model Architecture": model_type,
                "Status": "Loaded",
                "File Size": f"{sz:.2f} MB",
            })

        c_sz = config.CLUSTERING_MODEL_PATH.stat().st_size / 1024 if config.CLUSTERING_MODEL_PATH.exists() else 0.0
        status_rows.append({
            "Artifact": "Clustering Persona Bundle",
            "Model Architecture": f"K-Means (k={artifacts['clustering']['k']})",
            "Status": "Loaded",
            "File Size": f"{c_sz:.1f} KB",
        })

        r_sz = config.RISK_MODEL_PATH.stat().st_size / 1024 if config.RISK_MODEL_PATH.exists() else 0.0
        status_rows.append({
            "Artifact": "Calibrated Risk Layer",
            "Model Architecture": "Empirical Heteroscedastic Conformal Residuals",
            "Status": "Loaded",
            "File Size": f"{r_sz:.1f} KB",
        })

        st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not load artifact status table: {exc}")

    st.divider()

    st.markdown("### ⚠️ Known Research & Prototype Limitations")
    st.markdown(
        "- **Synthetic/Simulated Data**: The underlying dataset represents synthetic user behavior patterns with clean separations and high collinearity.\n"
        "- **Feature Collinearity**: Usage columns correlate closely (0.93 - 0.98), meaning SHAP attributions describe the mathematical models rather than causal mechanisms.\n"
        "- **Behavioral Assumptions**: Wi-Fi data offloading and operating-system power-saving multipliers are configurable heuristics, not empirical estimates learned from data.\n"
        "- **Constant Drain Assumption**: Battery survival estimation assumes daily active consumption is spread evenly across waking hours.\n"
        "- **Educational Scope**: PocketPilot does not interact directly with smartphone kernel battery statistics or hardware sensors."
    )

    st.divider()
    st.info(DISCLAIMER_TEXT)


if __name__ == "__main__":
    render_home_page()
