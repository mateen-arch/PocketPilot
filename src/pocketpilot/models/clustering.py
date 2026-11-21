"""
Usage profile clustering module (K-Means / GMM) generating neutral, non-judgmental usage personas.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pocketpilot.config import CLUSTER_K_SELECT_RANGE, RANDOM_SEED
from pocketpilot.features.input_modes import get_feature_columns

# Neutral, objective behavioral labels ordered ascending from lowest to highest overall usage.
# Hard rule: Never use stigmatizing or clinical terminology (e.g., addicted, addiction, problematic, unhealthy).
NEUTRAL_LABELS: dict[int, list[str]] = {
    3: ["Light use", "Moderate use", "Heavy use"],
    4: ["Light use", "Moderate use", "Heavy use", "Very heavy use"],
    5: ["Light use", "Lower-moderate use", "Moderate use", "Heavy use", "Very heavy use"],
    6: ["Light use", "Lower-moderate use", "Moderate use", "Upper-moderate use", "Heavy use", "Very heavy use"],
}


def build_kmeans_pipeline(k: int) -> Pipeline:
    """
    Construct an sklearn Pipeline comprising StandardScaler and KMeans.

    Parameters
    ----------
    k : int
        Number of clusters.

    Returns
    -------
    Pipeline
        Unfitted pipeline with StandardScaler and KMeans.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)),
    ])


def evaluate_k_range(train_df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    """
    Evaluate clustering quality metrics across a range of k values for K-Means and GMM.

    Methodology:
    - Features: app_usage_min, screen_on_hours, battery_mah, data_mb (behavior_class is strictly excluded).
    - Features are standardized using StandardScaler fitted strictly on the training partition.
    - Silhouette and Davies-Bouldin metrics are computed on the scaled training features using hard assignments.
    - Inertia is reported for K-Means; BIC is reported for Gaussian Mixture Models.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data partition.
    ks : list[int]
        List of cluster counts k to evaluate (e.g. 2 to 8).

    Returns
    -------
    pd.DataFrame
        Evaluation results with columns: ['method', 'k', 'silhouette', 'davies_bouldin', 'inertia', 'bic'].
    """
    feature_cols = get_feature_columns("clustering", mode="profile")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[feature_cols])

    records: list[dict[str, Any]] = []

    for k in ks:
        # (a) K-Means pipeline
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit(X_scaled)
        km_labels = km.labels_
        km_sil = float(silhouette_score(X_scaled, km_labels))
        km_db = float(davies_bouldin_score(X_scaled, km_labels))
        records.append({
            "method": "kmeans",
            "k": k,
            "silhouette": km_sil,
            "davies_bouldin": km_db,
            "inertia": float(km.inertia_),
            "bic": np.nan,
        })

        # (b) Gaussian Mixture Model on scaled features
        gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=RANDOM_SEED).fit(X_scaled)
        gmm_labels = gmm.predict(X_scaled)
        gmm_sil = float(silhouette_score(X_scaled, gmm_labels))
        gmm_db = float(davies_bouldin_score(X_scaled, gmm_labels))
        records.append({
            "method": "gmm",
            "k": k,
            "silhouette": gmm_sil,
            "davies_bouldin": gmm_db,
            "inertia": np.nan,
            "bic": float(gmm.bic(X_scaled)),
        })

    return pd.DataFrame(records)


def select_k(results_df: pd.DataFrame) -> int:
    """
    Select the optimal number of clusters k based on the training data.

    # selection uses training data only; K-Means ships because its centroids are directly
    # interpretable and prediction is a nearest-centroid lookup; the GMM rows are for comparison only.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of evaluate_k_range.

    Returns
    -------
    int
        Optimal cluster count k maximizing silhouette score within CLUSTER_K_SELECT_RANGE (3 to 6).
    """
    min_k, max_k = CLUSTER_K_SELECT_RANGE
    km_candidates = results_df[
        (results_df["method"] == "kmeans")
        & (results_df["k"] >= min_k)
        & (results_df["k"] <= max_k)
    ]
    if km_candidates.empty:
        raise ValueError(f"No KMeans results found within selection range {CLUSTER_K_SELECT_RANGE}.")

    best_row = km_candidates.loc[km_candidates["silhouette"].idxmax()]
    return int(best_row["k"])


def label_clusters(pipeline: Pipeline, feature_cols: list[str]) -> dict[int, str]:
    """
    Assign neutral persona labels to clusters ordered ascending by overall usage intensity.

    Intensity is computed as the mean of each cluster's scaled centroid coordinates.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted K-Means pipeline containing 'scaler' and 'kmeans' steps.
    feature_cols : list[str]
        Clustering feature column names.

    Returns
    -------
    dict[int, str]
        Dictionary mapping cluster_id (int) -> neutral persona label.

    Raises
    ------
    ValueError
        If k is not supported in NEUTRAL_LABELS.
    """
    kmeans: KMeans = pipeline.named_steps["kmeans"]
    k = kmeans.n_clusters
    if k not in NEUTRAL_LABELS:
        raise ValueError(
            f"No neutral label mapping defined for k={k}. Supported k: {sorted(NEUTRAL_LABELS.keys())}"
        )

    # Compute mean scaled intensity per cluster
    scaled_centers = kmeans.cluster_centers_
    intensities = scaled_centers.mean(axis=1)

    # Order cluster IDs ascending by intensity
    cluster_order = np.argsort(intensities)

    label_map: dict[int, str] = {}
    for rank, cluster_id in enumerate(cluster_order):
        label_map[int(cluster_id)] = NEUTRAL_LABELS[k][rank]

    return label_map


def fit_profile_model(train_df: pd.DataFrame, k: int) -> dict[str, Any]:
    """
    Fit the K-Means usage profiling model bundle on the training split.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data split.
    k : int
        Number of clusters to fit.

    Returns
    -------
    dict[str, Any]
        Plain dictionary bundle containing:
        - 'pipeline': fitted sklearn Pipeline (StandardScaler + KMeans)
        - 'label_map': dict mapping cluster_id -> persona label string
        - 'feature_cols': list of feature names used
        - 'k': integer number of clusters
        - 'centroids_original_units': DataFrame with centroids mapped back to real-world units
    """
    feature_cols = get_feature_columns("clustering", mode="profile")
    pipeline = build_kmeans_pipeline(k)
    pipeline.fit(train_df[feature_cols])

    label_map = label_clusters(pipeline, feature_cols)

    scaler: StandardScaler = pipeline.named_steps["scaler"]
    kmeans: KMeans = pipeline.named_steps["kmeans"]

    # Invert centroids back to original physical units (minutes, hours, mAh, MB)
    orig_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    df_centers = pd.DataFrame(orig_centers, columns=feature_cols)
    df_centers.index = [label_map[i] for i in range(k)]

    return {
        "pipeline": pipeline,
        "label_map": label_map,
        "feature_cols": feature_cols,
        "k": k,
        "centroids_original_units": df_centers,
    }


def predict_profile(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign user records to usage profile clusters and neutral persona labels.

    Parameters
    ----------
    bundle : dict[str, Any]
        Fitted clustering bundle from fit_profile_model.
    df : pd.DataFrame
        Input data containing the required clustering features.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['cluster_id', 'profile_label'] aligned with input index.

    Raises
    ------
    ValueError
        If any required feature is missing, non-numeric, non-finite, or negative.
    """
    feature_cols = bundle["feature_cols"]

    # 1. Missing columns check
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required clustering column(s): {missing}")

    # 2. Validation of values
    for col in feature_cols:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column '{col}' must be numeric, but has dtype {series.dtype}")
        if series.isna().any() or np.isinf(series).any():
            raise ValueError(f"Column '{col}' contains NaN or infinite values")
        if (series < 0).any():
            raise ValueError(f"Column '{col}' contains negative values violating physical bounds")

    # Predict cluster indices
    pipeline: Pipeline = bundle["pipeline"]
    cluster_ids = pipeline.predict(df[feature_cols])
    labels = [bundle["label_map"][int(cid)] for cid in cluster_ids]

    return pd.DataFrame({
        "cluster_id": cluster_ids,
        "profile_label": labels,
    }, index=df.index)


def percentile_report(
    reference_df: pd.DataFrame,
    row_df: pd.DataFrame | pd.Series,
    columns: list[str],
) -> pd.DataFrame:
    """
    Calculate the empirical percentile (0 to 100) of user values within a reference distribution.

    For each metric, percentile = (count of reference values <= user value) / N * 100.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference distribution (typically training split).
    row_df : pd.DataFrame or pd.Series
        User row(s) to evaluate.
    columns : list[str]
        List of column names for which to compute percentiles.

    Returns
    -------
    pd.DataFrame
        DataFrame with requested columns containing empirical percentiles in range [0, 100].
    """
    if isinstance(row_df, pd.Series):
        row_df = row_df.to_frame().T

    records: list[dict[str, float]] = []
    for _, row in row_df.iterrows():
        row_pcts: dict[str, float] = {}
        for col in columns:
            if col not in reference_df.columns:
                raise ValueError(f"Column '{col}' not found in reference DataFrame")
            if col not in row:
                raise ValueError(f"Column '{col}' not found in target row")

            ref_vals = np.asarray(reference_df[col], dtype=float)
            val = float(row[col])
            pct = float(np.mean(ref_vals <= val) * 100.0)
            row_pcts[col] = pct
        records.append(row_pcts)

    return pd.DataFrame(records, index=row_df.index)
