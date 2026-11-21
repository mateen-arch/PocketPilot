"""
Input modes and feature leakage prevention rules for PocketPilot.

Rationale for Input Modes:
1. Forecast Mode:
   - Used when predicting battery drain, data consumption, or total screen time.
   - Only inputs known BEFORE or DURING the day are permitted.
   - Realized end-of-day metrics (battery_mah, data_mb) cannot be used as features to forecast each other.
   - behavior_class cannot be used because it is a direct synthetic banding of app usage, screen time,
     battery drain, and data usage. Using it would constitute direct target leakage.
   - User ID has no causal power. Operating System is 100% collinear with Device Model (only iPhone 12 is iOS;
     the four others are Android). Age and gender are reserved strictly for subgroup residual analysis.
   - For screen time prediction, screen_on_hours (the target) and app_share_of_screen (derived from screen_on_hours)
     must be strictly excluded.

2. Profile-Analysis Mode:
   - Used for retrospective clustering (K-Means/GMM) into neutral behavioral profiles.
   - Historical aggregate metrics (app_usage_min, screen_on_hours, battery_mah, data_mb) are clustered.
   - These profiles are descriptive and never presented as forward-looking forecasts.
"""

from pocketpilot.config import TARGET_BATTERY, TARGET_DATA, TARGET_SCREEN

# Allowed feature lists by task in Forecast mode
FORECAST_FEATURES: dict[str, list[str]] = {
    "battery": [
        "device_model",
        "app_usage_min",
        "screen_on_hours",
        "apps_installed",
        "app_share_of_screen",
    ],
    "data": [
        "device_model",
        "app_usage_min",
        "screen_on_hours",
        "apps_installed",
        "app_share_of_screen",
    ],
    "screen_time": [
        "device_model",
        "app_usage_min",
        "apps_installed",
    ],
}

# Allowed feature list for Profile-analysis mode (retrospective clustering)
CLUSTERING_FEATURES: list[str] = [
    "app_usage_min",
    "screen_on_hours",
    "battery_mah",
    "data_mb",
]

# Columns strictly forbidden in Forecast mode due to target leakage or artificial correlations
FORBIDDEN_IN_FORECAST: set[str] = {
    "battery_mah",              # Realized end-of-day target
    "data_mb",                  # Realized end-of-day target
    "behavior_class",           # Discretized banding of outcome metrics
    "data_per_screen_hour",     # Derived by dividing target data_mb by screen_on_hours
    "battery_per_screen_hour",  # Derived by dividing target battery_mah by screen_on_hours
    "user_id",                  # Non-predictive unique record identifier
}

# Columns never used as model inputs across any mode
# - user_id: non-causal identifier
# - os: perfectly redundant with device_model
# - age, gender: excluded from models to prevent demographic bias, retained for subgroup error analysis
# - behavior_class: synthetic label
UNUSED_COLUMNS: set[str] = {
    "user_id",
    "os",
    "age",
    "gender",
    "behavior_class",
}


def get_feature_columns(task: str, mode: str) -> list[str]:
    """
    Retrieve the authorized list of feature columns for a specified task and mode.

    Parameters
    ----------
    task : str
        Target modeling task: {"battery", "data", "screen_time", "clustering"}.
    mode : str
        Operating mode: {"forecast", "profile"}.

    Returns
    -------
    list[str]
        Copy of the authorized list of feature column names.

    Raises
    ------
    ValueError
        If task or mode is unrecognized, or if an invalid combination is requested
        (e.g., forecasting tasks in profile mode or clustering in forecast mode).
    """
    valid_tasks = {"battery", "data", "screen_time", "clustering"}
    valid_modes = {"forecast", "profile"}

    if task not in valid_tasks:
        raise ValueError(
            f"Invalid task '{task}'. Expected one of: {sorted(valid_tasks)}"
        )

    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Expected one of: {sorted(valid_modes)}"
        )

    if mode == "forecast":
        if task == "clustering":
            raise ValueError(
                "Clustering task is for Profile-analysis mode only; not allowed in 'forecast' mode."
            )
        return list(FORECAST_FEATURES[task])

    if mode == "profile":
        if task != "clustering":
            raise ValueError(
                f"Task '{task}' is a forecast task and must be requested with mode='forecast', not 'profile'."
            )
        return list(CLUSTERING_FEATURES)

    raise ValueError(f"Unsupported task/mode pairing: task='{task}', mode='{mode}'")


def assert_no_leakage(feature_cols: list[str], task: str, mode: str) -> None:
    """
    Validate that a given list of feature columns contains no data leakage.

    Parameters
    ----------
    feature_cols : list[str]
        List of feature column names to be checked.
    task : str
        Modeling task: {"battery", "data", "screen_time", "clustering"}.
    mode : str
        Operating mode: {"forecast", "profile"}.

    Raises
    ------
    ValueError
        If any forbidden column or target-derived column is detected in feature_cols.
    """
    if mode == "forecast":
        forbidden = set(FORBIDDEN_IN_FORECAST)

        # Add task-specific targets to forbidden set
        if task == "battery":
            forbidden.add(TARGET_BATTERY)
        elif task == "data":
            forbidden.add(TARGET_DATA)
        elif task == "screen_time":
            # For screen_time estimation, screen_on_hours is the target,
            # and app_share_of_screen is directly computed from screen_on_hours.
            forbidden.add(TARGET_SCREEN)
            forbidden.add("app_share_of_screen")

        leaked_cols = set(feature_cols) & forbidden
        if leaked_cols:
            raise ValueError(
                f"Data leakage detected for task '{task}' in mode '{mode}'! "
                f"Forbidden columns present: {sorted(leaked_cols)}"
            )
