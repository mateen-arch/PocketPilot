"""
Clustering evaluation metrics computation: Silhouette Score, Davies-Bouldin Index, and cluster profiling.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from pocketpilot.features.input_modes import get_feature_columns


def clustering_quality(
    X_scaled: np.ndarray,
    labels: np.ndarray | pd.Series | list[int],
) -> dict[str, float]:
    """
    Compute cluster separation and cohesion metrics on standardized features.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix.
    labels : array-like
        Cluster assignments.

    Returns
    -------
    dict[str, float]
        Dictionary with 'silhouette' and 'davies_bouldin' scores.
    """
    labels_arr = np.asarray(labels)
    sil = float(silhouette_score(X_scaled, labels_arr))
    db = float(davies_bouldin_score(X_scaled, labels_arr))
    return {
        "silhouette": sil,
        "davies_bouldin": db,
    }


def cluster_summary(
    df: pd.DataFrame,
    labels_series: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Compute demographic and usage statistics per profile label in original physical units.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the original clustering feature columns.
    labels_series : array-like
        Assigned profile labels for each row in df.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: ['profile_label', 'count', 'share', ...features_in_original_units].
    """
    feature_cols = get_feature_columns("clustering", mode="profile")
    temp = df[feature_cols].copy()
    temp["profile_label"] = np.asarray(labels_series)

    total_rows = len(temp)
    grouped = temp.groupby("profile_label", observed=True)

    agg_dict: dict[str, tuple[str, Any]] = {
        "count": ("profile_label", "count"),
        "share": ("profile_label", lambda s: len(s) / total_rows),
    }
    for col in feature_cols:
        agg_dict[col] = (col, "mean")

    summary_df = grouped.agg(**agg_dict).reset_index()
    return summary_df


def cluster_vs_behavior_class(
    df: pd.DataFrame,
    labels_series: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """
    Compare unsupervised usage clusters against the dataset's synthetic 'behavior_class' label.

    Note on behavior_class:
    This column is used strictly for retrospective validation and comparative analysis;
    it is NEVER provided as an input to clustering or forecasting models.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'behavior_class'.
    labels_series : array-like
        Assigned profile labels.

    Returns
    -------
    tuple[pd.DataFrame, float]
        A tuple of (crosstab_df, adjusted_rand_score).
    """
    labels_arr = np.asarray(labels_series)
    ct = pd.crosstab(
        labels_arr,
        df["behavior_class"],
        rownames=["profile_label"],
        colnames=["behavior_class"],
    )
    ari = float(adjusted_rand_score(df["behavior_class"], labels_arr))
    return ct, ari
