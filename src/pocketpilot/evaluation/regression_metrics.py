"""
Regression evaluation metrics and residual analysis utilities.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def regression_metrics(
    y_true: np.ndarray | pd.Series | list[float],
    y_pred: np.ndarray | pd.Series | list[float],
) -> dict[str, float]:
    """
    Calculate primary regression performance metrics: MAE, RMSE, and R2.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted target values from the regression model.

    Returns
    -------
    dict[str, float]
        Dictionary containing 'mae', 'rmse', and 'r2' (also indexed by uppercase keys).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def residual_table(
    df: pd.DataFrame,
    y_true: np.ndarray | pd.Series | list[float],
    y_pred: np.ndarray | pd.Series | list[float],
    group_col: str,
) -> pd.DataFrame:
    """
    Compute subgroup residual statistics: count, mean residual, and MAE per group.

    Residual is defined as (y_true - y_pred). A positive mean residual indicates
    systematic under-prediction, while a negative mean residual indicates over-prediction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the grouping column.
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Model predictions.
    group_col : str
        Name of the column in df to group by (e.g. 'device_model', 'os', 'behavior_class').

    Returns
    -------
    pd.DataFrame
        Summary table with columns: [group_col, 'count', 'mean_residual', 'mae'].
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_t - y_p
    abs_errors = np.abs(residuals)

    res_df = pd.DataFrame({
        group_col: df[group_col].values,
        "residual": residuals,
        "abs_error": abs_errors,
    })

    grouped = (
        res_df.groupby(group_col, observed=True)
        .agg(
            count=("residual", "count"),
            mean_residual=("residual", "mean"),
            mae=("abs_error", "mean"),
        )
        .reset_index()
    )

    return grouped
