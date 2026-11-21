"""
Action-cost objective evaluation quantifying behavioral disruption.
"""

import pandas as pd

from pocketpilot.config import W1_MINUTES, W2_POWER_SAVING, W3_WIFI_PER_PCT_POINT


def action_cost(grid_or_scenarios: pd.DataFrame) -> pd.Series:
    """
    Compute behavioral disruption penalty for candidate modification plans.

    This objective replaces the initial proposal's category penalty with a power-saving
    convenience penalty because Dataset 1 has no app categories; the weights are design
    choices reflecting human friction and device convenience loss, not learned parameters.

    Formula:
        Cost = W1 * (app_cut_min + 60 * screen_cut_hours)
             + W2 * power_saving
             + W3 * (100 * wifi_share)

    Parameters
    ----------
    grid_or_scenarios : pd.DataFrame
        DataFrame containing columns: 'app_cut_min', 'screen_cut_hours',
        'wifi_share', 'power_saving'.

    Returns
    -------
    pd.Series
        Calculated action costs. The no-op scenario costs exactly 0.0.
    """
    app_cut = grid_or_scenarios["app_cut_min"].astype(float)
    screen_cut = grid_or_scenarios["screen_cut_hours"].astype(float)
    wifi_share = grid_or_scenarios["wifi_share"].astype(float)
    power_saving = grid_or_scenarios["power_saving"].astype(float)

    cost = (
        W1_MINUTES * (app_cut + 60.0 * screen_cut)
        + W2_POWER_SAVING * power_saving
        + W3_WIFI_PER_PCT_POINT * (100.0 * wifi_share)
    )

    return cost
