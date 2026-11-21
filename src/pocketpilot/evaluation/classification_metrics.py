"""
Classification and calibration evaluation metrics for budget-overrun risk.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def overrun_classification_metrics(
    y_true_exceeds: np.ndarray | pd.Series | list[int],
    y_pred_exceeds: np.ndarray | pd.Series | list[int],
) -> dict[str, Any]:
    """
    Calculate classification performance metrics (precision, recall, F1) for budget exceed events.

    Parameters
    ----------
    y_true_exceeds : array-like
        Binary indicator of realized budget overruns (1 if y_true > budget else 0).
    y_pred_exceeds : array-like
        Binary predictions of budget overruns (1 if alert triggered else 0).

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'precision', 'recall', 'f1', 'support_positive', and 'support_total'.
    """
    y_t = np.asarray(y_true_exceeds, dtype=int)
    y_p = np.asarray(y_pred_exceeds, dtype=int)

    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    support_pos = int(np.sum(y_t))
    support_tot = int(len(y_t))

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support_positive": support_pos,
        "support_total": support_tot,
    }


def brier_score(
    y_true_exceeds: np.ndarray | pd.Series | list[int],
    p: np.ndarray | pd.Series | list[float],
) -> float:
    """
    Compute Brier probability score: Mean squared error between observed binary outcomes and probabilities.

    Lower Brier score indicates superior probabilistic accuracy.

    Parameters
    ----------
    y_true_exceeds : array-like
        Binary ground truth labels (0 or 1).
    p : array-like
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
        Brier score loss.
    """
    y_t = np.asarray(y_true_exceeds, dtype=float)
    probs = np.asarray(p, dtype=float)
    return float(np.mean((y_t - probs) ** 2))


def reliability_table(
    y_true_exceeds: np.ndarray | pd.Series | list[int],
    p: np.ndarray | pd.Series | list[float],
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Construct a calibration reliability table partitioning predicted probabilities into equal-width bins.

    Parameters
    ----------
    y_true_exceeds : array-like
        Binary true outcomes (1 if overrun occurred else 0).
    p : array-like
        Predicted overrun probabilities.
    n_bins : int, default=5
        Number of equal-width probability bins between 0.0 and 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['bin_range', 'count', 'mean_predicted_probability', 'observed_exceed_rate'].
    """
    y_t = np.asarray(y_true_exceeds, dtype=float)
    probs = np.asarray(p, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    records: list[dict[str, Any]] = []
    for b in range(n_bins):
        low, high = bins[b], bins[b + 1]
        mask = bin_indices == b
        cnt = int(np.sum(mask))

        if cnt > 0:
            mean_prob = float(np.mean(probs[mask]))
            obs_rate = float(np.mean(y_t[mask]))
        else:
            mean_prob = float((low + high) / 2.0)
            obs_rate = 0.0

        records.append({
            "bin_range": f"[{low:.1f}, {high:.1f}]",
            "count": cnt,
            "mean_predicted_probability": mean_prob,
            "observed_exceed_rate": obs_rate,
        })

    return pd.DataFrame(records)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """
    Calculate Expected Calibration Error (ECE) as count-weighted mean absolute gap.

    Parameters
    ----------
    table : pd.DataFrame
        Output of reliability_table.

    Returns
    -------
    float
        ECE score in range [0, 1].
    """
    total = table["count"].sum()
    if total == 0:
        return 0.0

    gap = np.abs(table["mean_predicted_probability"] - table["observed_exceed_rate"])
    weighted_gap = table["count"] * gap
    return float(weighted_gap.sum() / total)


def interval_coverage(
    y_true: np.ndarray | pd.Series | list[float],
    lower: np.ndarray | pd.Series | list[float],
    upper: np.ndarray | pd.Series | list[float],
) -> float:
    """
    Compute empirical coverage fraction of prediction intervals: share of true y in [lower, upper].

    Parameters
    ----------
    y_true : array-like
        Ground truth realized values.
    lower : array-like
        Lower bounds of prediction intervals.
    upper : array-like
        Upper bounds of prediction intervals.

    Returns
    -------
    float
        Empirical coverage fraction in [0, 1].
    """
    y_t = np.asarray(y_true, dtype=float)
    low = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)

    is_covered = (y_t >= low) & (y_t <= up)
    return float(np.mean(is_covered))
