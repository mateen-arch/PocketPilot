"""
Page 1: Daily Phone Planner - Full daily resource forecasting, risk evaluation, and what-if planning.
"""

from datetime import time
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

from app.components.charts import render_shap_waterfall_bar
from app.components.gauges import render_resource_gauge
from app.components.input_forms import render_preset_selector, validate_user_inputs
from app.components.state import get_artifacts
from pocketpilot import config
from pocketpilot.explain.shap_utils import explain_one
from pocketpilot.inference.formatting import format_clock, format_hours, format_probability
from pocketpilot.inference.predictor import will_i_make_it

DISCLAIMER_TEXT = (
    "PocketPilot is a research and educational prototype trained on aggregated public "
    "mobile-usage records. It estimates daily behavior patterns and should not be "
    "interpreted as an operating-system-level battery monitor."
)


def render_daily_planner_page() -> None:
    st.title("🔋 Daily Phone Planner")
    st.markdown(
        "Forecast your daily smartphone resource consumption, evaluate budget overrun risks, "
        "and discover minimal-disruption behavioral plans to make your battery last all day."
    )
    st.caption("ℹ️ Note: All entered usage values represent expected DAILY TOTALS for today, as the underlying models are trained on daily aggregates.")

    artifacts = get_artifacts()

    # Preset selector
    render_preset_selector(page_prefix="p1")

    st.subheader("1. Your Smartphone & Usage Inputs")
    col1, col2, col3 = st.columns(3)

    with col1:
        dev_default = st.session_state.get("p1_device_model", "Samsung Galaxy S21")
        dev_idx = config.KNOWN_DEVICES.index(dev_default) if dev_default in config.KNOWN_DEVICES else 0
        device_model = st.selectbox("Device Model", config.KNOWN_DEVICES, index=dev_idx, key="p1_device_model")

        app_usage_min = st.number_input(
            "Expected Total App Minutes Today",
            min_value=1,
            max_value=1440,
            value=int(st.session_state.get("p1_app_usage_min", 250)),
            step=15,
            key="p1_app_usage_min",
        )

        estimate_screen = st.checkbox(
            "I do not know, estimate screen time for me",
            value=False,
            key="p1_estimate_screen",
        )

        if not estimate_screen:
            screen_on_hours = st.number_input(
                "Expected Screen-On Hours Today",
                min_value=0.1,
                max_value=24.0,
                value=float(st.session_state.get("p1_screen_on_hours", 5.0)),
                step=0.5,
                key="p1_screen_on_hours",
            )
        else:
            screen_on_hours = 5.0  # Placeholder, ignored by predictor when screen_known=False
            st.info("Screen time will be estimated directly by the screen-time machine learning model.")

        apps_installed = st.number_input(
            "Apps Installed on Phone",
            min_value=0,
            max_value=500,
            value=int(st.session_state.get("p1_apps_installed", 50)),
            step=5,
            key="p1_apps_installed",
        )

    with col2:
        st.markdown("**Battery State & Daily Schedule**")
        battery_capacity_mah = st.number_input(
            "Battery Capacity (mAh)",
            min_value=config.CAPACITY_RANGE_MAH[0],
            max_value=config.CAPACITY_RANGE_MAH[1],
            value=int(st.session_state.get("p1_battery_capacity_mah", 4000)),
            step=250,
            key="p1_battery_capacity_mah",
        )

        current_battery_pct = st.number_input(
            "Current Battery State (%)",
            min_value=1,
            max_value=100,
            value=int(st.session_state.get("p1_current_battery_pct", 38)),
            step=1,
            key="p1_current_battery_pct",
        )

        min_battery_pct = st.number_input(
            "Minimum Reserve Level Required (%)",
            min_value=0,
            max_value=99,
            value=int(st.session_state.get("p1_min_battery_pct", 15)),
            step=5,
            key="p1_min_battery_pct",
        )

        current_time = st.time_input(
            "Current Time",
            value=time(16, 0),
            key="p1_current_time",
        )

        hours_needed = st.number_input(
            "Hours Phone is Needed from Now",
            min_value=0.5,
            max_value=24.0,
            value=float(st.session_state.get("p1_hours_needed", 6.0)),
            step=0.5,
            key="p1_hours_needed",
        )

    with col3:
        st.markdown("**Resource Target Budgets**")
        data_budget_mb = st.number_input(
            "Mobile Data Budget (MB)",
            min_value=1.0,
            max_value=100000.0,
            value=1000.0,
            step=250.0,
            key="p1_data_budget",
        )

        screen_budget_hours = st.number_input(
            "Screen-Time Target (Hours)",
            min_value=0.5,
            max_value=24.0,
            value=6.0,
            step=0.5,
            key="p1_screen_budget",
        )

        allow_assumptions = st.toggle(
            "Allow assumption-based actions (Wi-Fi shift, power saving)",
            value=st.session_state.get("p1_allow_assumptions", True),
            key="p1_allow_assumptions",
            help="If disabled, optimizer will only explore learned behavioral cuts (app minutes and screen time).",
        )

    # Action button
    if st.button("🚀 Forecast & Evaluate Plan", type="primary", key="p1_run_btn"):
        # Validation checks
        errs = validate_user_inputs(
            device_model=device_model,
            app_usage_min=app_usage_min,
            screen_on_hours=screen_on_hours,
            apps_installed=apps_installed,
            battery_budget=1800.0,
            data_budget=data_budget_mb,
            screen_budget=screen_budget_hours,
            screen_known=not estimate_screen,
        )

        # Boundary check: battery already depleted
        if current_battery_pct <= min_battery_pct:
            errs.append(
                f"Current battery level ({current_battery_pct}%) is already at or below your minimum reserve limit ({min_battery_pct}%)."
            )

        if errs:
            for e in errs:
                st.error(e)
            st.session_state["p1_result"] = None
        else:
            hour_now = current_time.hour + current_time.minute / 60.0
            need_until = hour_now + hours_needed

            inputs_dict = {
                "device_model": device_model,
                "app_usage_min": float(app_usage_min),
                "apps_installed": int(apps_installed),
                "screen_known": not estimate_screen,
            }
            if not estimate_screen:
                inputs_dict["screen_on_hours"] = float(screen_on_hours)

            survival_inputs = {
                "battery_capacity_mah": float(battery_capacity_mah),
                "current_battery_pct": float(current_battery_pct),
                "current_hour": hour_now,
                "need_until_hour": need_until,
                "min_battery_pct": float(min_battery_pct),
            }

            extra_budgets = {
                "data_mb": float(data_budget_mb),
                "screen_hours": float(screen_budget_hours),
            }

            try:
                result = will_i_make_it(
                    artifacts=artifacts,
                    inputs=inputs_dict,
                    survival_inputs=survival_inputs,
                    extra_budgets=extra_budgets,
                    allow_assumptions=allow_assumptions,
                )
                st.session_state["p1_result"] = result
                st.session_state["p1_inputs"] = inputs_dict
            except ValueError as ve:
                st.error(str(ve))
                st.session_state["p1_result"] = None
            except Exception as exc:
                logging.exception("Error during Daily Planner execution")
                st.error("Something went wrong. Please check your inputs and try again.")
                st.session_state["p1_result"] = None

    # Render persisted result
    result = st.session_state.get("p1_result")
    if result is not None:
        st.divider()
        st.subheader("2. Forecast & Survival Assessment")

        verdict = result["verdict"]
        survival_info = result["survival"]
        forecast_info = result["forecast"]
        b_used = result["budgets_used"]

        # Big Verdict banner
        if survival_info["expected_to_make_it"]:
            st.success(f"**Verdict:** {verdict}")
        else:
            st.warning(f"**Verdict:** {verdict}")

        # Metrics overview cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(
                "Expected Survival Time",
                format_hours(survival_info["hours_until_threshold"]),
                f"Needed: {format_hours(survival_info['hours_needed'])}",
            )
            st.caption(f"80% Interval: `{survival_info['hours_until_lower']:.1f}` to `{survival_info['hours_until_upper']:.1f} h`")
        with col_m2:
            st.metric(
                "P(Make It to Target)",
                format_probability(survival_info["p_make_it"]),
                delta="Likely safe" if survival_info["expected_to_make_it"] else "At risk",
            )
        with col_m3:
            st.metric(
                "Predicted Battery Drain",
                f"{forecast_info['predictions']['battery_mah']:.0f} mAh",
                f"Rate: {survival_info['hourly_rate']:.1f} mAh/h",
            )
        with col_m4:
            src_str = "You entered" if forecast_info["screen_source"] == "user" else "Model estimate"
            st.metric(
                "Predicted Screen Time",
                f"{forecast_info['predictions']['screen_hours']:.1f} h",
                f"Source: {src_str}",
            )

        st.markdown("#### Operational Resource Risk Gauges")
        col_g1, col_g2, col_g3 = st.columns(3)

        with col_g1:
            r_bat = forecast_info["risk"]["battery"]
            render_resource_gauge(
                title="Battery Consumption",
                predicted=forecast_info["predictions"]["battery_mah"],
                budget=b_used["battery_mah"],
                unit="mAh",
                risk_level=r_bat["level"],
                p_overrun_str=format_probability(r_bat["p_overrun"]),
                interval_80=r_bat.get("interval_80"),
                key_prefix="p1_bat",
            )
            st.caption(f"Evaluated against equivalent daily budget: `{b_used['battery_mah']:.0f} mAh`.")

        with col_g2:
            r_data = forecast_info["risk"]["data"]
            render_resource_gauge(
                title="Mobile Data Usage",
                predicted=forecast_info["predictions"]["data_mb"],
                budget=b_used["data_mb"],
                unit="MB",
                risk_level=r_data["level"],
                p_overrun_str=format_probability(r_data["p_overrun"]),
                interval_80=r_data.get("interval_80"),
                key_prefix="p1_data",
            )

        with col_g3:
            r_scr = forecast_info["risk"]["screen_time"]
            render_resource_gauge(
                title="Screen-On Time",
                predicted=forecast_info["predictions"]["screen_hours"],
                budget=b_used["screen_hours"],
                unit="hours",
                risk_level=r_scr["level"],
                p_overrun_str=format_probability(r_scr["p_overrun"]),
                interval_80=r_scr.get("interval_80"),
                key_prefix="p1_scr",
            )

        # What-If Plan remediation section (if goal is not met)
        plan_info = result.get("plan")
        if plan_info is not None:
            st.divider()
            st.subheader("3. Recommended Remediation Plan")
            if plan_info["status"] == "plan_found":
                best = plan_info["recommended_plan"]
                st.success(
                    f"**Feasible Plan Found!** (Disruption Cost: `{best['action_cost']:.1f}`)\n\n"
                    "Applying these targeted adjustments allows your phone to satisfy all resource constraints:"
                )

                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.markdown("**Action Levers:**")
                    if best["app_cut_min"] > 0:
                        st.markdown(f"- ⏱️ Reduce app usage by **{int(best['app_cut_min'])} minutes**")
                    if best["screen_cut_hours"] > 0:
                        st.markdown(f"- 📱 Cut extra screen time by **{best['screen_cut_hours']:.1f} hours**")
                    if best["wifi_share"] > 0:
                        st.markdown(f"- 📶 Shift **{int(best['wifi_share']*100)}%** mobile data to Wi-Fi *(assumption)*")
                    if best["power_saving"]:
                        st.markdown("- 🔋 Turn on **Power-Saving Mode** *(assumption: reduces battery by 15%)*")
                with col_p2:
                    st.markdown("**Expected Resource Reductions:**")
                    st.markdown(f"- Battery Drain: `{forecast_info['predictions']['battery_mah']:.0f}` → **`{best['battery_pred']:.0f} mAh`**")
                    st.markdown(f"- Mobile Data: `{forecast_info['predictions']['data_mb']:.0f}` → **`{best['data_pred']:.0f} MB`**")
                    st.markdown(f"- Screen Time: `{forecast_info['predictions']['screen_hours']:.1f}` → **`{best['screen_pred']:.1f} h`**")
                with col_p3:
                    st.markdown("**Post-Plan Risk State:**")
                    st.markdown(f"- Battery Risk: `{best['risk']['battery']['level']}` (P(overrun)={format_probability(best['risk']['battery']['p_overrun'])})")
                    st.markdown(f"- Data Risk: `{best['risk']['data']['level']}` (P(overrun)={format_probability(best['risk']['data']['p_overrun'])})")
                    st.markdown(f"- Screen Risk: `{best['risk']['screen_time']['level']}` (P(overrun)={format_probability(best['risk']['screen_time']['p_overrun'])})")
            else:
                st.error(
                    "**Goal Not Achievable within Safe Operational Bounds.**\n\n"
                    "Even with maximum allowable reductions, the target duration cannot be met without recharging. "
                    "A best-effort scenario minimizing maximum overrun has been calculated."
                )

        # Transparent Explanations Expander
        with st.expander("🔍 Why this prediction? (SHAP Feature Contributions)"):
            st.markdown(
                "SHAP (SHapley Additive exPlanations) decomposes the prediction into additive contributions "
                "from each feature relative to the training baseline."
            )
            inputs_for_shap = st.session_state.get("p1_inputs", {})
            row_raw = pd.DataFrame([{
                "device_model": inputs_for_shap.get("device_model", device_model),
                "app_usage_min": inputs_for_shap.get("app_usage_min", app_usage_min),
                "screen_on_hours": forecast_info["screen_used"],
                "apps_installed": inputs_for_shap.get("apps_installed", apps_installed),
            }])

            tab_bat, tab_data, tab_scr = st.tabs(["Battery SHAP", "Data Usage SHAP", "Screen Time SHAP"])
            with tab_bat:
                try:
                    bat_exp = explain_one(artifacts["pipelines"]["battery"], row_raw, top_k=5)
                    render_shap_waterfall_bar(
                        bat_exp["contributions"],
                        bat_exp["prediction"],
                        bat_exp["base_value"],
                        "Battery Drain",
                        "mAh",
                        key_prefix="p1_shap_bat",
                    )
                except ValueError as ve:
                    st.info(f"SHAP explanation note: {ve}")
            with tab_data:
                try:
                    data_exp = explain_one(artifacts["pipelines"]["data"], row_raw, top_k=5)
                    render_shap_waterfall_bar(
                        data_exp["contributions"],
                        data_exp["prediction"],
                        data_exp["base_value"],
                        "Data Consumption",
                        "MB",
                        key_prefix="p1_shap_data",
                    )
                except ValueError as ve:
                    st.info(f"SHAP explanation note: {ve}")
            with tab_scr:
                try:
                    row_scr = row_raw[["device_model", "app_usage_min", "apps_installed"]]
                    scr_exp = explain_one(artifacts["pipelines"]["screen_time"], row_scr, top_k=3)
                    render_shap_waterfall_bar(
                        scr_exp["contributions"],
                        scr_exp["prediction"],
                        scr_exp["base_value"],
                        "Screen-On Time",
                        "hours",
                        key_prefix="p1_shap_scr",
                    )
                except ValueError as ve:
                    st.info(f"SHAP explanation note: {ve}")

        # Assumptions Expander
        with st.expander("📋 Theoretical & Modeling Assumptions"):
            for a in survival_info.get("assumptions", []):
                st.markdown(f"- {a}")
            if plan_info and plan_info.get("assumptions_used"):
                st.markdown("**Planner Lever Assumptions Employed:**")
                for pu in plan_info["assumptions_used"]:
                    st.markdown(f"- *{pu}*")
            st.caption("ℹ️ To see your retrospective digital persona classification, please visit the **Digital Use Profile** page.")

    st.divider()
    st.info(DISCLAIMER_TEXT)


if __name__ == "__main__":
    render_daily_planner_page()
