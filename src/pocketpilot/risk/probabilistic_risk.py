"""
Probabilistic budget overrun estimation and central prediction intervals.
"""

from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from pocketpilot.config import (
    HETEROSCEDASTICITY_RHO_THRESHOLD,
    RANDOM_SEED,
    TARGET_BATTERY,
    TARGET_DATA,
    TARGET_SCREEN,
)
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP, TASK_TARGET_MAP
from pocketpilot.risk.threshold_risk import risk_level


def compute_oof_residuals(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute out-of-fold (OOF) residuals using 5-fold cross-validation on the training split.

    # Out-of-fold residuals are used because in-sample residuals would be too small
    # and make the risk estimates overconfident.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted or cloned model pipeline.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : array-like
        Ground truth target values for the training partition.
    seed : int, default=RANDOM_SEED
        Random state for K-Fold shuffling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (oof_residuals, oof_predictions)
    """
    # Clone drops fitted state while preserving tuned hyperparameters and architecture
    cloned_pipe = clone(pipeline)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    y_true = np.asarray(y_train, dtype=float)
    oof_pred = cross_val_predict(cloned_pipe, X_train, y_true, cv=cv, n_jobs=-1)
    oof_resid = y_true - oof_pred

    return oof_resid, oof_pred


def choose_residual_variant(
    oof_pred: np.ndarray,
    oof_resid: np.ndarray,
) -> tuple[str, float]:
    """
    Determine whether to model residuals additively or relatively based on heteroscedasticity.

    # This rule uses training data only; evaluates whether error magnitude scales with predicted usage.

    Parameters
    ----------
    oof_pred : np.ndarray
        Out-of-fold predicted values.
    oof_resid : np.ndarray
        Out-of-fold residuals (y_true - oof_pred).

    Returns
    -------
    tuple[str, float]
        ('relative' or 'additive', spearman_rho)
    """
    res = spearmanr(oof_pred, np.abs(oof_resid))
    rho = float(res.statistic if hasattr(res, "statistic") else res[0])

    if rho > HETEROSCEDASTICITY_RHO_THRESHOLD:
        return "relative", rho
    return "additive", rho


def fit_risk_model(
    task: str,
    pipeline: Pipeline,
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Fit empirical residual distribution for budget overrun risk estimation.

    Parameters
    ----------
    task : str
        Modeling task: 'battery', 'data', or 'screen_time'.
    pipeline : Pipeline
        Trained model pipeline for the task.
    train_df : pd.DataFrame
        Training data split.

    Returns
    -------
    dict[str, Any]
        Risk bundle dictionary with keys:
        ['task', 'variant', 'rho', 'residuals', 'n_residuals', 'floor_value', 'seed'].
    """
    if task not in TASK_TARGET_MAP:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {list(TASK_TARGET_MAP.keys())}")

    target_col = TASK_TARGET_MAP[task]
    raw_inputs = TASK_RAW_INPUTS_MAP[task]

    X_train = train_df[raw_inputs]
    y_train = np.asarray(train_df[target_col], dtype=float)

    oof_resid, oof_pred = compute_oof_residuals(pipeline, X_train, y_train, seed=RANDOM_SEED)
    variant, rho = choose_residual_variant(oof_pred, oof_resid)

    floor_val = float(0.01 * np.mean(y_train))

    if variant == "relative":
        # Relative residual = (y_true - y_hat) / max(y_hat, floor)
        safe_denom = np.maximum(oof_pred, floor_val)
        residuals = oof_resid / safe_denom
    else:
        # Additive residual = y_true - y_hat
        residuals = oof_resid

    return {
        "task": task,
        "variant": variant,
        "rho": rho,
        "residuals": np.asarray(residuals, dtype=float),
        "n_residuals": len(residuals),
        "floor_value": floor_val,
        "seed": RANDOM_SEED,
    }


def overrun_probability(
    bundle: dict[str, Any],
    y_hat: float | np.ndarray,
    budget: float,
) -> float | np.ndarray:
    """
    Calculate P(Y > budget | y_hat) using empirical residual sampling.

    # Clip the result to the range 1/(n+1) to n/(n+1) because a finite sample
    # cannot justify probabilities of exactly 0 or 1.

    Parameters
    ----------
    bundle : dict[str, Any]
        Fitted risk bundle from fit_risk_model.
    y_hat : float or np.ndarray
        Predicted resource consumption value(s).
    budget : float
        User-defined operational budget (> 0).

    Returns
    -------
    float or np.ndarray
        Estimated overrun probability, clipped to [1/(n+1), n/(n+1)].

    Raises
    ------
    ValueError
        If budget is non-positive or non-finite, or if y_hat contains non-finite values.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError(f"Budget must be finite and strictly positive (> 0), got: {budget}")

    residuals = bundle["residuals"]
    n = bundle["n_residuals"]
    variant = bundle["variant"]

    is_scalar = np.isscalar(y_hat) or (isinstance(y_hat, np.ndarray) and y_hat.ndim == 0)
    arr_hat = np.asarray(y_hat, dtype=float)

    if not np.all(np.isfinite(arr_hat)):
        raise ValueError("y_hat contains non-finite values (NaN or Inf)")

    lower_clip = 1.0 / (n + 1.0)
    upper_clip = float(n) / (n + 1.0)

    if variant == "relative":
        if is_scalar:
            simulated = float(arr_hat) * (1.0 + residuals)
            prob = float(np.mean(simulated > budget))
        else:
            simulated = arr_hat[:, None] * (1.0 + residuals[None, :])
            prob = np.mean(simulated > budget, axis=1)
    else:
        if is_scalar:
            simulated = float(arr_hat) + residuals
            prob = float(np.mean(simulated > budget))
        else:
            simulated = arr_hat[:, None] + residuals[None, :]
            prob = np.mean(simulated > budget, axis=1)

    clipped = np.clip(prob, lower_clip, upper_clip)
    return float(clipped) if is_scalar else clipped


def predict_interval(
    bundle: dict[str, Any],
    y_hat: float | np.ndarray,
    level: float,
) -> tuple[Any, Any]:
    """
    Compute central empirical prediction intervals (e.g. 80% or 90%) for resource consumption.

    Parameters
    ----------
    bundle : dict[str, Any]
        Fitted risk bundle from fit_risk_model.
    y_hat : float or np.ndarray
        Predicted resource consumption value(s).
    level : float
        Nominal interval coverage level (e.g. 0.8 or 0.9).

    Returns
    -------
    tuple[Any, Any]
        (lower_bound, upper_bound), with lower bound floored at 0.
    """
    if not (0 < level < 1):
        raise ValueError(f"Interval level must be in (0, 1), got: {level}")

    residuals = bundle["residuals"]
    variant = bundle["variant"]

    is_scalar = np.isscalar(y_hat) or (isinstance(y_hat, np.ndarray) and y_hat.ndim == 0)
    arr_hat = np.asarray(y_hat, dtype=float)

    alpha = (1.0 - level) / 2.0
    q_low = float(np.quantile(residuals, alpha))
    q_high = float(np.quantile(residuals, 1.0 - alpha))

    if variant == "relative":
        lower = arr_hat * (1.0 + q_low)
        upper = arr_hat * (1.0 + q_high)
    else:
        lower = arr_hat + q_low
        upper = arr_hat + q_high

    lower = np.maximum(0.0, lower)

    if is_scalar:
        return (float(lower), float(upper))
    return (lower, upper)


def assess_budget(
    bundle: dict[str, Any],
    y_hat: float,
    budget: float,
) -> dict[str, Any]:
    """
    Unified budget risk evaluation combining deterministic thresholds, overrun probability, and intervals.

    Parameters
    ----------
    bundle : dict[str, Any]
        Fitted risk bundle from fit_risk_model.
    y_hat : float
        Predicted resource consumption.
    budget : float
        User-defined operational budget.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - 'predicted': float y_hat
        - 'budget': float budget
        - 'level': deterministic status ('Safe', 'Watch', 'Likely exceed')
        - 'p_overrun': empirical overrun probability
        - 'interval_80': (lower, upper) 80% prediction interval
        - 'interval_90': (lower, upper) 90% prediction interval
    """
    level = risk_level(y_hat, budget)
    p_ov = float(overrun_probability(bundle, y_hat, budget))
    int80 = predict_interval(bundle, y_hat, level=0.8)
    int90 = predict_interval(bundle, y_hat, level=0.9)

    return {
        "predicted": float(y_hat),
        "budget": float(budget),
        "level": level,
        "p_overrun": p_ov,
        "interval_80": int80,
        "interval_90": int90,
    }
