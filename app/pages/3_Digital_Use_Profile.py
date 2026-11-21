"""
Page 3: Digital Use Profile - Retrospective behavioral persona clustering and historical percentile analysis.
"""

import io
from pathlib import Path
import logging
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

from app.components.charts import render_history_trend_chart
from app.components.state import get_artifacts
from pocketpilot.inference.predictor import profile, profile_history
from pocketpilot.inference.validation import parse_history_csv


def render_profile_page() -> None:
    st.title("📊 Digital Use Profile")
    st.markdown(
        "Analyze your historical smartphone usage patterns against reference behavior personas."
    )
    st.info(
        "⚠️ **Retrospective Analysis Only**: This page evaluates **OBSERVED historical averages** or multi-day logs. "
        "It operates independently from the forecasting tools on Pages 1 and 2 to strictly prevent target leakage."
    )

    artifacts = get_artifacts()

    tab_manual, tab_csv = st.tabs(["📝 Manual Entry (Daily Averages)", "📁 Upload Multi-Day History CSV"])

    # TAB 1: Manual Entry
    with tab_manual:
        st.subheader("Enter Observed Historical Daily Averages")
        col1, col2 = st.columns(2)

        with col1:
            obs_app = st.number_input(
                "Observed Average App Usage (Minutes/Day)",
                value=250.0,
                step=15.0,
                key="p3_obs_app",
            )
            obs_screen = st.number_input(
                "Observed Average Screen Time (Hours/Day)",
                value=5.0,
                step=0.5,
                key="p3_obs_screen",
            )
        with col2:
            obs_bat = st.number_input(
                "Observed Average Battery Drain (mAh/Day)",
                value=1600.0,
                step=100.0,
                key="p3_obs_bat",
            )
            obs_data = st.number_input(
                "Observed Average Mobile Data (MB/Day)",
                value=850.0,
                step=100.0,
                key="p3_obs_data",
            )

        if st.button("🔎 Classify Usage Persona", type="primary", key="p3_classify_btn"):
            if obs_app <= 0 or obs_screen <= 0 or obs_bat <= 0 or obs_data <= 0:
                st.error("Observed metrics must be strictly positive non-zero numbers.")
                st.session_state["p3_manual_result"] = None
            else:
                observed_dict = {
                    "app_usage_min": float(obs_app),
                    "screen_on_hours": float(obs_screen),
                    "battery_mah": float(obs_bat),
                    "data_mb": float(obs_data),
                }

                try:
                    res = profile(artifacts, observed_dict)
                    st.session_state["p3_manual_result"] = res
                    st.session_state["p3_manual_input"] = observed_dict
                except ValueError as ve:
                    st.error(str(ve))
                    st.session_state["p3_manual_result"] = None
                except Exception:
                    logging.exception("Error during profile evaluation")
                    st.error("Something went wrong. Please check your inputs and try again.")
                    st.session_state["p3_manual_result"] = None

        m_res = st.session_state.get("p3_manual_result")
        if m_res is not None:
            st.divider()
            persona_label = m_res["profile_label"]
            st.success(f"### Assigned Behavioral Persona: **{persona_label}**")
            st.caption("ℹ️ Profiles describe usage intensity in simulated reference data and are not a health or addiction assessment.")

            # Percentile report explanation sentence
            pct_df = m_res["percentile_report"]
            scr_pct = pct_df["screen_on_hours"].iloc[0]
            app_pct = pct_df["app_usage_min"].iloc[0]
            st.markdown(
                f"**Benchmarking Insight:** Your daily screen time is higher than **{scr_pct:.1f}%** of reference users, "
                f"and your app engagement exceeds **{app_pct:.1f}%** of the baseline population."
            )

            # Centroid comparison table
            st.markdown("#### Comparison with Persona Cluster Centroids (Original Units)")
            centroids = m_res["centroid_table"].copy()
            user_in = st.session_state.get("p3_manual_input", {})
            user_row = pd.DataFrame([user_in], index=["Your Observed Average"])
            comp_table = pd.concat([user_row, centroids])
            st.dataframe(
                comp_table.style.format({
                    "app_usage_min": "{:.1f} min",
                    "screen_on_hours": "{:.2f} h",
                    "battery_mah": "{:.0f} mAh",
                    "data_mb": "{:.0f} MB",
                }),
                use_container_width=True,
            )

    # TAB 2: Multi-day CSV Upload
    with tab_csv:
        st.subheader("Upload Multi-Day Historical Log")
        st.markdown(
            "Upload a CSV recording multi-day usage history. Required columns: "
            "`app_usage_min`, `screen_on_hours`, `battery_mah`, `data_mb` (optional `date`)."
        )

        # In-memory template download
        template_csv = "date,app_usage_min,screen_on_hours,battery_mah,data_mb\n2026-11-01,180,4.2,1450,650\n2026-11-02,240,5.1,1700,900\n2026-11-03,320,6.5,2100,1200\n"
        st.download_button(
            "📥 Download CSV Log Template",
            data=template_csv.encode("utf-8"),
            file_name="pocketpilot_history_template.csv",
            mime="text/csv",
            key="p3_dl_template",
        )

        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], key="p3_file_upload")

        if uploaded_file is not None:
            try:
                history_clean = parse_history_csv(uploaded_file)
                st.success(f"Successfully loaded and validated {len(history_clean)} records.")

                hist_res = profile_history(artifacts, history_clean)

                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    avg_p = hist_res["average_profile"]
                    st.metric("Multi-Day Average Persona", avg_p["profile_label"])
                with col_h2:
                    st.markdown("**Share of Days per Persona:**")
                    for lbl, share in hist_res["label_shares"].items():
                        st.markdown(f"- **{lbl}**: `{share*100:.1f}%` of logged days")

                st.markdown("#### Historical Usage Timeline & Persona Evolution")
                render_history_trend_chart(history_clean, hist_res["history_predictions"], key_prefix="p3")

            except ValueError as ve:
                st.error(f"Validation error: {ve}")
            except Exception as exc:
                logging.exception("Error processing uploaded CSV history")
                st.error(f"Failed to process CSV file: {exc}")


if __name__ == "__main__":
    render_profile_page()
