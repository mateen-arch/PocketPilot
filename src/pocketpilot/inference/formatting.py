"""
Display and string formatting helpers for PocketPilot inference outputs (Feature F8).

Provides pure, safe formatting functions for clock times, probabilities, and hours.
"""

from typing import Any
import numpy as np


def format_clock(hour: float) -> str:
    """
    Format a decimal hour into standard 24-hour clock string (HH:MM),
    annotating next-day offsets if hour >= 24.0.

    Parameters
    ----------
    hour : float
        Time in decimal hours (e.g. 16.0 -> '16:00', 22.5 -> '22:30', 25.5 -> '01:30 (+1 day)').

    Returns
    -------
    str
        Formatted clock string.

    Raises
    ------
    ValueError
        If hour is negative, infinite, or NaN.
    """
    try:
        val = float(hour)
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"Hour must be non-negative and finite, got: {hour}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Hour must be a valid non-negative number, got: {hour}") from exc

    total_minutes = int(round(val * 60.0))
    days = total_minutes // (24 * 60)
    rem_minutes = total_minutes % (24 * 60)
    h = rem_minutes // 60
    m = rem_minutes % 60

    clock_str = f"{h:02d}:{m:02d}"
    if days == 0:
        return clock_str
    if days == 1:
        return f"{clock_str} (+1 day)"
    return f"{clock_str} (+{days} days)"


def format_probability(p: float) -> str:
    """
    Format a probability value in [0, 1] as a clean percentage string.

    Probabilities are clipped at 1/(n+1) and n/(n+1) in the risk layer,
    so the app must not show false precision.
    - p < 0.01 -> 'under 1%'
    - p > 0.99 -> 'over 99%'
    - otherwise -> rounded whole percentage (e.g. 0.146 -> '15%')

    Parameters
    ----------
    p : float
        Probability in [0, 1].

    Returns
    -------
    str
        Formatted percentage string.

    Raises
    ------
    ValueError
        If p is outside [0, 1] or not finite.
    """
    try:
        val = float(p)
        if not np.isfinite(val) or val < 0.0 or val > 1.0:
            raise ValueError(f"Probability must be in [0, 1] and finite, got: {p}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Probability must be a valid number in [0, 1], got: {p}") from exc

    if val < 0.01:
        return "under 1%"
    if val > 0.99:
        return "over 99%"
    pct = int(round(val * 100.0))
    return f"{pct}%"


def format_hours(h: float) -> str:
    """
    Format a duration in decimal hours with 1 decimal place (e.g. 9.65 -> '9.7 h', 0.0 -> '0.0 h').

    Parameters
    ----------
    h : float
        Hours duration.

    Returns
    -------
    str
        Formatted hours string.

    Raises
    ------
    ValueError
        If h is non-finite or negative.
    """
    try:
        val = float(h)
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"Hours must be non-negative and finite, got: {h}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Hours must be a valid non-negative number, got: {h}") from exc

    return f"{val:.1f} h"
