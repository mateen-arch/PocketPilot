"""
Reusable input widgets, persona preset loaders, and validation callers for PocketPilot pages.
"""

from typing import Any
import streamlit as st

from pocketpilot import config
from pocketpilot.inference.validation import validate_inputs


def apply_preset_callback(page_prefix: str) -> None:
    """Callback triggered before widget re-render to safely update session_state from preset."""
    selected_key = st.session_state.get(f"{page_prefix}_preset_select")
    if selected_key and selected_key in config.PERSONAS:
        p = config.PERSONAS[selected_key]
        st.session_state[f"{page_prefix}_device_model"] = p["device_model"]
        st.session_state[f"{page_prefix}_app_usage_min"] = int(p["app_usage_min"])
        st.session_state[f"{page_prefix}_screen_on_hours"] = float(p["screen_on_hours"])
        st.session_state[f"{page_prefix}_apps_installed"] = int(p["apps_installed"])
        st.session_state[f"{page_prefix}_battery_capacity_mah"] = int(p["battery_capacity_mah"])


def render_preset_selector(page_prefix: str) -> None:
    """Render the 'Load an example user' dropdown and Apply button."""
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        preset_options = {"": "-- Select an example persona --"}
        preset_options.update({k: v["label"] for k, v in config.PERSONAS.items()})
        st.selectbox(
            "Load an example user",
            options=list(preset_options.keys()),
            format_func=lambda k: preset_options[k],
            key=f"{page_prefix}_preset_select",
        )
    with col_btn:
        st.write("")  # Alignment spacing
        st.write("")
        st.button(
            "Apply preset",
            key=f"{page_prefix}_apply_preset_btn",
            on_click=apply_preset_callback,
            args=(page_prefix,),
        )


def validate_user_inputs(
    device_model: Any,
    app_usage_min: Any,
    screen_on_hours: Any,
    apps_installed: Any,
    battery_budget: Any = 1800.0,
    data_budget: Any = 1000.0,
    screen_budget: Any = 6.0,
    screen_known: bool = True,
) -> list[str]:
    """
    Validate usage form inputs using central validation logic.

    Returns empty list if valid, or a list of user-readable error strings.
    """
    # If screen is estimated, pass a dummy valid value to validate_inputs
    val_screen = screen_on_hours if screen_known else 5.0
    return validate_inputs(
        device_model=device_model,
        app_usage_min=app_usage_min,
        screen_on_hours=val_screen,
        apps_installed=apps_installed,
        battery_budget=battery_budget,
        data_budget=data_budget,
        screen_budget=screen_budget,
    )
