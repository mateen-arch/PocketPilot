"""PocketPilot - Main Streamlit Application Router.

Defines navigation and launches multi-page interactive modules:
- Home: Overview, system status, limitations, and prototype disclaimer
- Daily Phone Planner: Forecast-driven resource management & minimal interventions
- Will I Make It: Goal-oriented battery survival & what-if scenario comparisons
- Digital Use Profile: Retrospective observed usage clustering & historical tracking
- Model Evaluation: Empirical benchmark tables, residual analysis, SHAP, and test audits
"""

from pathlib import Path
import sys
import streamlit as st

# Bootstrap sys.path with project root and src directory
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
for p in (ROOT_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="PocketPilot", layout="wide")

pages = [
    st.Page("app/pages/0_Home.py", title="Home", icon=":material/home:", default=True),
    st.Page(
        "app/pages/1_Daily_Phone_Planner.py",
        title="Daily Phone Planner",
        icon=":material/schedule:",
    ),
    st.Page(
        "app/pages/2_Will_I_Make_It.py",
        title="Will I Make It",
        icon=":material/battery_charging_full:",
    ),
    st.Page(
        "app/pages/3_Digital_Use_Profile.py",
        title="Digital Use Profile",
        icon=":material/person:",
    ),
    st.Page(
        "app/pages/4_Model_Evaluation.py",
        title="Model Evaluation",
        icon=":material/analytics:",
    ),
]

nav = st.navigation(pages)
nav.run()
