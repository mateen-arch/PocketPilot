"""
Daily mobile data consumption regression model wrapper (predicting MB).
"""

from typing import Any
import pandas as pd
from pocketpilot.models.trainer import train_task


def train_data_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fast: bool = False,
) -> dict[str, Any]:
    """
    Train and benchmark mobile data usage regression candidate models.

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
    return train_task(task="data", train_df=train_df, test_df=test_df, fast=fast)
