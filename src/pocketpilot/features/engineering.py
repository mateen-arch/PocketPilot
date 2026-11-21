"""
Feature engineering module for PocketPilot.

Calculates derived domain features for Forecast and Profile-analysis modes.
"""

import pandas as pd


def add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute input features permissible in Forecast mode.

    Generates:
    - app_share_of_screen = app_usage_min / (screen_on_hours * 60)

    Note on app_share_of_screen:
    This metric represents the ratio of active foreground app usage time to total
    screen-on time. It is an intensity metric, NOT a probability. It is deliberately
    not clipped to [0, 1] because multitasking, background media services, or split-screen
    app sessions can legally record app usage exceeding single-app foreground screen time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'app_usage_min' and 'screen_on_hours'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the added 'app_share_of_screen' feature.

    Raises
    ------
    ValueError
        If screen_on_hours <= 0 anywhere in the dataset.
    """
    out = df.copy()

    if (out["screen_on_hours"] <= 0).any():
        non_positive_count = (out["screen_on_hours"] <= 0).sum()
        raise ValueError(
            f"screen_on_hours must be strictly positive (> 0) to compute app_share_of_screen. "
            f"Found {non_positive_count} non-positive values."
        )

    out["app_share_of_screen"] = out["app_usage_min"] / (out["screen_on_hours"] * 60.0)
    return out


def add_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain features permissible ONLY in Profile-analysis mode.

    Generates:
    - data_per_screen_hour = data_mb / screen_on_hours
    - battery_per_screen_hour = battery_mah / screen_on_hours

    Leakage Prevention Notice:
    Because these features divide realized end-of-day outcomes (data_mb and battery_mah)
    by screen_on_hours, they are intended exclusively for descriptive clustering and profile
    analysis. They must NEVER appear in any Forecast mode feature pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'data_mb', 'battery_mah', and 'screen_on_hours'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing 'data_per_screen_hour' and 'battery_per_screen_hour'.

    Raises
    ------
    ValueError
        If screen_on_hours <= 0 anywhere in the dataset.
    """
    out = df.copy()

    if (out["screen_on_hours"] <= 0).any():
        non_positive_count = (out["screen_on_hours"] <= 0).sum()
        raise ValueError(
            f"screen_on_hours must be strictly positive (> 0) to compute per-hour profile metrics. "
            f"Found {non_positive_count} non-positive values."
        )

    out["data_per_screen_hour"] = out["data_mb"] / out["screen_on_hours"]
    out["battery_per_screen_hour"] = out["battery_mah"] / out["screen_on_hours"]
    return out
