"""
Input validation utilities for PocketPilot inference and deployment apps.
"""

from typing import Any
import numpy as np

from pocketpilot.config import KNOWN_DEVICES


def validate_inputs(
    device_model: Any,
    app_usage_min: Any,
    screen_on_hours: Any,
    apps_installed: Any,
    battery_budget: Any,
    data_budget: Any,
    screen_budget: Any,
) -> list[str]:
    """
    Validate user inputs for PocketPilot forecast and risk estimation.

    Parameters
    ----------
    device_model : Any
        Selected smartphone device model. Must be in KNOWN_DEVICES.
    app_usage_min : Any
        Daily app usage in minutes. Range: [1, 1440].
    screen_on_hours : Any
        Expected screen-on time in hours. Range: [0.1, 24.0].
    apps_installed : Any
        Number of installed applications. Range: [0, 500].
    battery_budget : Any
        User battery budget in mAh. Must be strictly positive (> 0).
    data_budget : Any
        User data budget in MB. Must be strictly positive (> 0).
    screen_budget : Any
        User screen time budget in hours. Must be strictly positive (> 0).

    Returns
    -------
    list[str]
        List of human-readable error messages. Empty list if all inputs are valid.
    """
    errors: list[str] = []

    # 1. Device model validation
    if device_model is None or not isinstance(device_model, str) or not device_model.strip():
        errors.append("Device model must be provided.")
    elif device_model not in KNOWN_DEVICES:
        errors.append(f"Device model '{device_model}' is not recognized. Expected one of: {KNOWN_DEVICES}.")

    # 2. App usage minutes validation: [1, 1440]
    if app_usage_min is None:
        errors.append("App usage minutes must be provided.")
    else:
        try:
            val_app = float(app_usage_min)
            if not np.isfinite(val_app) or val_app < 1.0 or val_app > 1440.0:
                errors.append("App usage must be between 1 and 1440 minutes.")
        except (ValueError, TypeError):
            errors.append("App usage minutes must be a valid number.")

    # 3. Expected screen-on hours validation: [0.1, 24.0]
    if screen_on_hours is None:
        errors.append("Expected screen-on hours must be provided.")
    else:
        try:
            val_screen = float(screen_on_hours)
            if not np.isfinite(val_screen) or val_screen < 0.1 or val_screen > 24.0:
                errors.append("Expected screen-on hours must be between 0.1 and 24.0 hours.")
        except (ValueError, TypeError):
            errors.append("Expected screen-on hours must be a valid number.")

    # 4. Apps installed validation: [0, 500]
    if apps_installed is None:
        errors.append("Number of apps installed must be provided.")
    else:
        try:
            val_apps = float(apps_installed)
            if not np.isfinite(val_apps) or val_apps < 0.0 or val_apps > 500.0:
                errors.append("Number of apps installed must be between 0 and 500.")
        except (ValueError, TypeError):
            errors.append("Number of apps installed must be a valid number.")

    # 5. Battery budget validation: > 0 mAh
    if battery_budget is None:
        errors.append("Battery budget must be provided.")
    else:
        try:
            val_bat_b = float(battery_budget)
            if not np.isfinite(val_bat_b) or val_bat_b <= 0.0:
                errors.append("Battery budget must be strictly positive (> 0 mAh).")
        except (ValueError, TypeError):
            errors.append("Battery budget must be a valid number.")

    # 6. Data budget validation: > 0 MB
    if data_budget is None:
        errors.append("Data budget must be provided.")
    else:
        try:
            val_data_b = float(data_budget)
            if not np.isfinite(val_data_b) or val_data_b <= 0.0:
                errors.append("Data budget must be strictly positive (> 0 MB).")
        except (ValueError, TypeError):
            errors.append("Data budget must be a valid number.")

    # 7. Screen time budget validation: > 0 hours
    if screen_budget is None:
        errors.append("Screen time budget must be provided.")
    else:
        try:
            val_scr_b = float(screen_budget)
            if not np.isfinite(val_scr_b) or val_scr_b <= 0.0:
                errors.append("Screen time budget must be strictly positive (> 0 hours).")
        except (ValueError, TypeError):
            errors.append("Screen time budget must be a valid number.")

    return errors
