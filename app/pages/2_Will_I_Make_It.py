"""
Page 2: Will I Make It? - Goal-oriented battery survival estimation and remedial plan recommendations.
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

from app.components.charts import render_alternatives_table
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


def render_will_i_make_it_page() -> None:
    st.title("🎯 Will I Make It?")
    st.markdown(
        "Define your personal survival target and find out if your remaining charge will get you there."
    )

    artifacts = get_artifacts()

    # Preset selector
    render_preset_selector(page_prefix="p2")

    st.subheader("1. Usage Baseline & Smartphone Status")
    col1, col2 = st.columns(2)

    with col1:
        dev_default = st.session_state.get("p2_device_model", "Samsung Galaxy S21")
        dev_idx = config.KNOWN_DEVICES.index(dev_default) if dev_default in config.KNOWN_DEVICES else 0
        device_model = st.selectbox("Device Model", config.KNOWN_DEVICES, index=dev_idx, key="p2_device_model")

        app_usage_min = st.number_input(
            "Expected Total App Minutes Today",
            min_value=1,
            max_value=1440,
            value=int(st.session_state.get("p2_app_usage_min", 250)),
            step=15,
            key="p2_app_usage_min",
        )

        estimate_screen = st.checkbox(
            "Estimate screen time for me",
            value=False,
            key="p2_estimate_screen",
        )

        if not estimate_screen:
            screen_on_hours = st.number_input(
                "Expected Screen-On Hours Today",
                min_value=0.1,
                max_value=24.0,
                value=float(st.session_state.get("p2_screen_on_hours", 5.0)),
                step=0.5,
                key="p2_screen_on_hours",
            )
        else:
            screen_on_hours = 5.0

        apps_installed = st.number_input(
            "Apps Installed",
            min_value=0,
            max_value=500,
            value=int(st.session_state.get("p2_apps_installed", 50)),
            step=5,
            key="p2_apps_installed",
        )

    with col2:
        battery_capacity_mah = st.number_input(
            "Battery Capacity (mAh)",
            min_value=config.CAPACITY_RANGE_MAH[0],
            max_value=config.CAPACITY_RANGE_MAH[1],
            value=int(st.session_state.get("p2_battery_capacity_mah", 4000)),
            step=250,
            key="p2_battery_capacity_mah",
        )

        current_battery_pct = st.number_input(
            "Current Battery (%)",
            min_value=1,
            max_value=100,
            value=int(st.session_state.get("p2_current_battery_pct", 40)),
            step=1,
            key="p2_current_battery_pct",
        )

        current_time = st.time_input(
            "Current Time",
            value=time(16, 0),
            key="p2_current_time",
        )

    st.subheader("2. Your Battery Survival Target")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        target_min_pct = st.number_input(
            "I want at least this much battery left (%)",
            min_value=0,
            max_value=99,
            value=20,
            step=5,
            key="p2_target_min_pct",
        )
    with col_t2:
        target_duration_hours = st.number_input(
            "In this many hours from now",
            min_value=0.5,
            max_value=24.0,
            value=5.0,
            step=0.5,
            key="p2_target_duration_hours",
        )

    allow_assumptions = st.toggle(
        "Allow assumption-based actions (Wi-Fi shift, power saving)",
        value=True,
        key="p2_allow_assumptions",
    )

    if st.button("🔍 Evaluate My Target", type="primary", key="p2_eval_btn"):
        # Validate that current battery is strictly above the target reserve
        if current_battery_pct <= target_min_pct:
            st.error(
                f"Goal already violated! Your current battery level ({current_battery_pct}%) is already "
                f"at or below your desired target reserve of {target_min_pct}%."
            )
            st.session_state["p2_result"] = None
        else:
            errs = validate_user_inputs(
                device_model=device_model,
                app_usage_min=app_usage_min,
                screen_on_hours=screen_on_hours,
                apps_installed=apps_installed,
                battery_budget=1800.0,
                data_budget=1000.0,
                screen_budget=6.0,
                screen_known=not estimate_screen,
            )

            if errs:
                for e in errs:
                    st.error(e)
                st.session_state["p2_result"] = None
            else:
                hour_now = current_time.hour + current_time.minute / 60.0
                need_until = hour_now + target_duration_hours

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
                    "min_battery_pct": float(target_min_pct),
                }

                try:
                    res = will_i_make_it(
                        artifacts=artifacts,
                        inputs=inputs_dict,
                        survival_inputs=survival_inputs,
                        extra_budgets=None,
                        allow_assumptions=allow_assumptions,
                    )
                    st.session_state["p2_result"] = res
                    st.session_state["p2_inputs"] = inputs_dict
                except ValueError as ve:
                    st.error(str(ve))
                    st.session_state["p2_result"] = None
                except Exception as exc:
                    logging.exception("Error during Will I Make It evaluation")
                    st.error("Something went wrong. Please check your inputs and try again.")
                    st.session_state["p2_result"] = None

    # Render persisted result
    res = st.session_state.get("p2_result")
    if res is not None:
        st.divider()
        s_res = res["survival"]
        f_res = res["forecast"]
        p_res = res.get("plan")
        verdict = res["verdict"]

        if s_res["expected_to_make_it"]:
            st.success(f"### 🎉 Goal Achievable!\n\n**Verdict:** {verdict}")
        else:
            st.warning(f"### ⚠️ Goal at Risk!\n\n**Verdict:** {verdict}")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Estimated Survival",
                format_hours(s_res["hours_until_threshold"]),
                f"Target: {format_hours(s_res['hours_needed'])}",
            )
            st.caption(f"80% Interval: `{s_res['hours_until_lower']:.1f}` to `{s_res['hours_until_upper']:.1f} h`")
        with col_b:
            st.metric("P(Make It)", format_probability(s_res["p_make_it"]))
            st.caption(f"Usable capacity: `{s_res['usable_mah']:.0f} mAh`")
        with col_c:
            st.metric("Hourly Drain", f"{s_res['hourly_rate']:.1f} mAh/h")
            st.caption(f"Equivalent daily budget: `{s_res['equivalent_daily_budget_mah']:.0f} mAh`")

        # Top Drain Contributor (SHAP attribution)
        st.markdown("#### 🔍 Biggest Driver of Battery Drain")
        try:
            inputs_for_shap = st.session_state.get("p2_inputs", {})
            row_raw = pd.DataFrame([{
                "device_model": inputs_for_shap.get("device_model", device_model),
                "app_usage_min": inputs_for_shap.get("app_usage_min", app_usage_min),
                "screen_on_hours": f_res["screen_used"],
                "apps_installed": inputs_for_shap.get("apps_installed", apps_installed),
            }])
            bat_exp = explain_one(artifacts["pipelines"]["battery"], row_raw, top_k=1)
            top_feature = bat_exp["contributions"][0]
            st.info(
                f"**Top Contributor to Battery Drain:** `{top_feature[0]}` (Value: `{top_feature[1]}`) "
                f"adding `+{top_feature[2]:.1f} mAh` relative to the average baseline.\n\n"
                "*Note: Features are closely correlated; SHAP attributions describe the statistical model's "
                "attributions rather than isolated real-world causal drivers.*"
            )
        except Exception:
            pass

        # Remediation & Alternatives Section if goal is not met
        if not s_res["expected_to_make_it"]:
            st.divider()
            st.subheader("💡 Behavior Remediation Options")

            if p_res and p_res.get("status") == "plan_found":
                best = p_res["recommended_plan"]
                st.markdown(
                    f"**Recommended Minimal Disruption Plan** (Disruption Cost: `{best['action_cost']:.1f}`):"
                )

                st.markdown("**Learned vs. Assumption Effects:**")
                st.markdown(
                    f"- **Learned Effect**: {p_res.get('learned_effects_note', 'Cutting app minutes lowers screen time through the screen model.')}"
                )
                if p_res.get("assumptions_used"):
                    st.markdown("- **Assumption Effects**:")
                    for a in p_res["assumptions_used"]:
                        st.markdown(f"  - *{a}*")

                st.markdown("#### Alternative Planner Scenarios")
                alternatives = p_res.get("alternatives")
                equiv_budget = s_res["equivalent_daily_budget_mah"]
                render_alternatives_table(
                    alternatives_df=alternatives,
                    battery_budget=equiv_budget,
                    battery_risk_bundle=artifacts["risk"]["battery"],
                )
            else:
                st.error(
                    "**Goal is Not Achievable.**\n\n"
                    "No viable combination of behavioral reductions within training bounds can extend "
                    f"battery life to {format_hours(s_res['hours_needed'])} without recharging. "
                    "Please consider connecting your phone to a charger."
                )

    st.divider()
    st.info(DISCLAIMER_TEXT)


if __name__ == "__main__":
    render_will_i_make_it_page()
