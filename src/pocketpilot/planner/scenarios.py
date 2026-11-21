"""
What-if behavior perturbation scenario definitions and vector evaluation.
"""

from itertools import product
from typing import Any
import numpy as np
import pandas as pd

from pocketpilot.config import (
    PLANNER_APP_CUTS,
    PLANNER_SCREEN_CUTS,
    PLANNER_WIFI_SHARES,
    POWER_SAVING_BATTERY_MULTIPLIER,
)
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP


def build_scenario_grid() -> pd.DataFrame:
    """
    Construct the full combinatorial Cartesian grid of behavior modification scenarios.

    Returns
    -------
    pd.DataFrame
        DataFrame of 224 scenarios with levers, non-zero lever counts, and assumption flags.
    """
    rows = []
    # Cartesian product: 7 app cuts x 4 screen cuts x 4 wifi shares x 2 power saving states = 224 combinations
    for app_cut, screen_cut, wifi_share, power_saving in product(
        PLANNER_APP_CUTS, PLANNER_SCREEN_CUTS, PLANNER_WIFI_SHARES, (False, True)
    ):
        n_levers = (
            int(app_cut > 0)
            + int(screen_cut > 0.0)
            + int(wifi_share > 0.0)
            + int(power_saving is True)
        )
        is_assumption = (wifi_share > 0.0) or (power_saving is True)
        rows.append({
            "app_cut_min": int(app_cut),
            "screen_cut_hours": float(screen_cut),
            "wifi_share": float(wifi_share),
            "power_saving": bool(power_saving),
            "n_levers": int(n_levers),
            "is_assumption_based": bool(is_assumption),
        })

    return pd.DataFrame(rows)


def training_bounds(train_df: pd.DataFrame) -> dict[str, float]:
    """
    Extract lower training data domain bounds to prevent extrapolation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Cleaned training split partition.

    Returns
    -------
    dict[str, float]
        Dictionary with minimum observed training values for app usage and screen time.
    """
    app_min = float(train_df["app_usage_min"].min())
    screen_min = float(train_df["screen_on_hours"].min())

    return {
        "app_usage_min": app_min,
        "screen_on_hours": screen_min,
        "app_min": app_min,
        "screen_min": screen_min,
    }


def evaluate_scenarios(
    base: dict[str, Any],
    grid: pd.DataFrame,
    pipelines: dict[str, Any],
    bounds: dict[str, Any],
) -> pd.DataFrame:
    """
    Evaluate candidate behavior scenarios using vectorized pipeline batch inference.

    Methodology:
    1. Adjust app minutes: new_app = base_app - app_cut_min.
    2. Screen time follow-through:
       - Cutting app minutes must lower screen time using the trained screen_time model.
       - delta_screen = screen_pred(new_app) - screen_pred(base_app).
       - When app_cut_min == 0, delta_screen is exactly 0.0.
    3. Total new screen time: new_screen = base_screen - screen_cut_hours + delta_screen.
    4. Domain boundary check:
       - The planner never extrapolates below the training range.
       - within_bounds = (new_app >= bounds_app) & (new_screen >= bounds_screen) OR no-op scenario.
       - Scenarios with within_bounds False must never be recommended.
    5. Resource predictions:
       - Run battery and data pipelines with (device_model, new_app, new_screen, apps_installed).
       - Apply assumption levers POST-PREDICTION:
         * Power saving: battery_pred * POWER_SAVING_BATTERY_MULTIPLIER.
         * Wi-Fi shift: data_pred * (1 - wifi_share).
         * Power saving does NOT modify data, and Wi-Fi shift does NOT modify battery.
    6. screen_pred = new_screen (evaluated against the screen-time budget).

    Parameters
    ----------
    base : dict[str, Any]
        Baseline user record with device_model, app_usage_min, screen_on_hours, apps_installed.
    grid : pd.DataFrame
        Combinatorial scenario grid from build_scenario_grid.
    pipelines : dict[str, Any]
        Dictionary of trained model pipelines: 'battery', 'data', 'screen_time'.
    bounds : dict[str, Any]
        Training domain bounds from training_bounds.

    Returns
    -------
    pd.DataFrame
        Evaluated scenarios with levers, resulting inputs, predictions, and feasibility bounds.
    """
    # Clone grid to avoid mutating inputs
    df = grid.copy()
    n_scenarios = len(df)

    device_model = base["device_model"]
    base_app = float(base["app_usage_min"])
    base_screen = float(base["screen_on_hours"])
    apps_installed = int(base["apps_installed"])

    bounds_app = float(bounds.get("app_min", bounds.get("app_usage_min", 0.0)))
    bounds_screen = float(bounds.get("screen_min", bounds.get("screen_on_hours", 0.0)))

    # 1. New app usage minutes
    new_app = base_app - df["app_cut_min"].values

    # 2. Vectorized screen-time follow-through
    # Evaluate screen time model at base app usage
    base_screen_input = pd.DataFrame([{
        "device_model": device_model,
        "app_usage_min": base_app,
        "apps_installed": apps_installed,
    }])[TASK_RAW_INPUTS_MAP["screen_time"]]
    base_screen_pred = float(pipelines["screen_time"].predict(base_screen_input)[0])

    # Evaluate screen time model for all new_app values in batch
    # Use bounds-clipped inputs for model evaluation to ensure safe feature generation
    safe_app_for_screen = np.maximum(new_app, bounds_app)
    screen_inputs_batch = pd.DataFrame({
        "device_model": [device_model] * n_scenarios,
        "app_usage_min": safe_app_for_screen,
        "apps_installed": [apps_installed] * n_scenarios,
    })[TASK_RAW_INPUTS_MAP["screen_time"]]

    batch_screen_preds = pipelines["screen_time"].predict(screen_inputs_batch)
    delta_screen = np.where(df["app_cut_min"].values == 0, 0.0, batch_screen_preds - base_screen_pred)

    # 3. New screen-on hours
    new_screen = base_screen - df["screen_cut_hours"].values + delta_screen

    # 4. Domain validity / extrapolation prevention
    # The planner never extrapolates below the training range
    is_noop = (
        (df["app_cut_min"].values == 0)
        & (df["screen_cut_hours"].values == 0.0)
        & (df["wifi_share"].values == 0.0)
        & (~df["power_saving"].values)
    )
    within_bounds = (
        ((new_app >= bounds_app) & (new_screen >= bounds_screen))
        | is_noop
    )

    # 5. Batch inference for battery and data pipelines
    # Clip inputs to training domain minimum for pipeline transformation safety
    safe_screen = np.maximum(new_screen, bounds_screen)
    safe_app = np.maximum(new_app, bounds_app)
    full_inputs_batch = pd.DataFrame({
        "device_model": [device_model] * n_scenarios,
        "app_usage_min": safe_app,
        "screen_on_hours": safe_screen,
        "apps_installed": [apps_installed] * n_scenarios,
    })

    raw_bat_preds = pipelines["battery"].predict(full_inputs_batch[TASK_RAW_INPUTS_MAP["battery"]])
    raw_data_preds = pipelines["data"].predict(full_inputs_batch[TASK_RAW_INPUTS_MAP["data"]])

    # Apply ASSUMPTION levers post-prediction
    # Power saving modifies battery only; Wi-Fi shift modifies data only
    battery_pred = np.where(
        df["power_saving"].values,
        raw_bat_preds * POWER_SAVING_BATTERY_MULTIPLIER,
        raw_bat_preds,
    )
    data_pred = raw_data_preds * (1.0 - df["wifi_share"].values)

    # 6. Screen time prediction is new_screen
    screen_pred = new_screen

    df["new_app"] = new_app
    df["new_screen"] = new_screen
    df["battery_pred"] = battery_pred
    df["data_pred"] = data_pred
    df["screen_pred"] = screen_pred
    df["within_bounds"] = within_bounds

    return df
