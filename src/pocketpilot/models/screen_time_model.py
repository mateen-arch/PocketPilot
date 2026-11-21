"""
Screen-on time regression / estimation model wrapper (predicting hours/day).
"""

from typing import Any
import pandas as pd
from pocketpilot.models.trainer import train_task


def train_screen_time_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fast: bool = False,
) -> dict[str, Any]:
    """
    Train and benchmark screen-on time estimation candidate models.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data split.
    test_df : pd.DataFrame
        Held-out test data split.
    fast : bool, default=False
        If True, executes fast hyperparameter search for testing.

    Returns
    -------
    dict[str, Any]
        Dictionary with fitted candidates, evaluation results, and selected model pipeline.
    """
    return train_task(task="screen_time", train_df=train_df, test_df=test_df, fast=fast)
