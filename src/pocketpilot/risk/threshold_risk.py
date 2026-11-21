"""
Deterministic threshold-based budget risk classification (Safe / Watch / Likely exceed).
"""

import numpy as np

from pocketpilot.config import WATCH_FRACTION


def risk_level(
    y_hat: float,
    budget: float,
    watch_fraction: float = WATCH_FRACTION,
) -> str:
    """
    Classify predicted usage against a user-defined resource budget using deterministic threshold rules.

    Threshold Decision Boundary:
    - 'Safe': y_hat < watch_fraction * budget
    - 'Watch': watch_fraction * budget <= y_hat < budget
    - 'Likely exceed': y_hat >= budget

    Boundary Conditions:
    - y_hat == 0.8 * budget -> 'Watch'
    - y_hat == budget -> 'Likely exceed'

    Parameters
    ----------
    y_hat : float
        Predicted resource consumption (battery mAh, data MB, or screen hours).
    budget : float
        User-defined operational budget (> 0).
    watch_fraction : float, default=WATCH_FRACTION (0.8)
        Fraction of budget that triggers the 'Watch' threshold alert.

    Returns
    -------
    str
        One of {'Safe', 'Watch', 'Likely exceed'}.

    Raises
    ------
    ValueError
        If budget is non-positive or non-finite, or if y_hat is non-finite.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError(f"Budget must be a finite, strictly positive number (> 0), got: {budget}")
    if not np.isfinite(y_hat):
        raise ValueError(f"Predicted value (y_hat) must be finite, got: {y_hat}")

    if y_hat < watch_fraction * budget:
        return "Safe"
    if y_hat < budget:
        return "Watch"
    return "Likely exceed"


def risk_levels(
    y_hat_array: np.ndarray,
    budget: float,
    watch_fraction: float = WATCH_FRACTION,
) -> np.ndarray:
    """
    Vectorized threshold risk classification across an array of predictions.

    Parameters
    ----------
    y_hat_array : np.ndarray
        1D array of predicted resource values.
    budget : float
        User-defined operational budget (> 0).
    watch_fraction : float, default=WATCH_FRACTION (0.8)
        Fraction of budget that triggers the 'Watch' warning.

    Returns
    -------
    np.ndarray
        Array of string risk status labels.

    Raises
    ------
    ValueError
        If budget is non-positive or non-finite, or any prediction is non-finite.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError(f"Budget must be a finite, strictly positive number (> 0), got: {budget}")

    arr = np.asarray(y_hat_array, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Predicted array contains non-finite values (NaN or Inf)")

    watch_threshold = watch_fraction * budget

    conditions = [
        arr < watch_threshold,
        (arr >= watch_threshold) & (arr < budget),
        arr >= budget,
    ]
    choices = ["Safe", "Watch", "Likely exceed"]

    return np.select(conditions, choices, default="Likely exceed")
