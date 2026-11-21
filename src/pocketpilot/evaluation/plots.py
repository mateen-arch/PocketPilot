"""
Diagnostic and evaluation plotting routines for PocketPilot models and clustering.
"""

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pocketpilot.config import RANDOM_SEED
from pocketpilot.models.clustering import predict_profile, select_k


def plot_k_selection(results_df: pd.DataFrame, path: Path | str) -> None:
    """
    Plot Silhouette score and Davies-Bouldin index across k for K-Means and GMM.

    Parameters
    ----------
    results_df : pd.DataFrame
        Evaluation results from evaluate_k_range with columns:
        ['method', 'k', 'silhouette', 'davies_bouldin', 'inertia', 'bic'].
    path : Path or str
        Destination image path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    selected_k = select_k(results_df)

    km_df = results_df[results_df["method"] == "kmeans"].sort_values("k")
    gmm_df = results_df[results_df["method"] == "gmm"].sort_values("k")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Silhouette Score Plot (higher is better)
    ax1.plot(km_df["k"], km_df["silhouette"], marker="o", linewidth=2, color="#2563EB", label="K-Means")
    ax1.plot(gmm_df["k"], gmm_df["silhouette"], marker="s", linewidth=2, linestyle="--", color="#10B981", label="GMM")
    ax1.axvline(selected_k, color="#EF4444", linestyle=":", linewidth=2, label=f"Selected k = {selected_k}")
    ax1.set_title("Cluster Cohesion & Separation: Silhouette Score\n(Higher is Better)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax1.set_ylabel("Silhouette Score", fontsize=10)
    ax1.set_xticks(km_df["k"])
    ax1.legend(loc="best")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Davies-Bouldin Index Plot (lower is better)
    ax2.plot(km_df["k"], km_df["davies_bouldin"], marker="o", linewidth=2, color="#2563EB", label="K-Means")
    ax2.plot(gmm_df["k"], gmm_df["davies_bouldin"], marker="s", linewidth=2, linestyle="--", color="#10B981", label="GMM")
    ax2.axvline(selected_k, color="#EF4444", linestyle=":", linewidth=2, label=f"Selected k = {selected_k}")
    ax2.set_title("Cluster Separation Index: Davies-Bouldin\n(Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax2.set_ylabel("Davies-Bouldin Index", fontsize=10)
    ax2.set_xticks(km_df["k"])
    ax2.legend(loc="best")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pca_clusters(
    bundle: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    path: Path | str,
) -> None:
    """
    Project usage profiles onto 2D PCA space fitted on scaled training data only.

    Parameters
    ----------
    bundle : dict[str, Any]
        Fitted clustering bundle from fit_profile_model.
    train_df : pd.DataFrame
        Training data partition.
    test_df : pd.DataFrame
        Held-out test data partition.
    path : Path or str
        Destination image path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = bundle["feature_cols"]
    scaler = bundle["pipeline"].named_steps["scaler"]

    # Transform features with training scaler
    X_train_scaled = scaler.transform(train_df[feature_cols])
    X_test_scaled = scaler.transform(test_df[feature_cols])

    # Fit PCA strictly on scaled training data
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    train_preds = predict_profile(bundle, train_df)
    test_preds = predict_profile(bundle, test_df)

    # Determine unique labels and ordered color map
    labels_order = list(bundle["centroids_original_units"].index)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels_order)))
    label_to_color = {lbl: colors[i] for i, lbl in enumerate(labels_order)}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot train points
    for lbl in labels_order:
        mask = train_preds["profile_label"] == lbl
        ax.scatter(
            X_train_pca[mask, 0],
            X_train_pca[mask, 1],
            color=label_to_color[lbl],
            marker="o",
            alpha=0.6,
            s=45,
            label=f"{lbl} (Train)",
        )

    # Plot test points with distinct marker
    for lbl in labels_order:
        mask = test_preds["profile_label"] == lbl
        ax.scatter(
            X_test_pca[mask, 0],
            X_test_pca[mask, 1],
            color=label_to_color[lbl],
            marker="^",
            edgecolors="black",
            linewidth=1.2,
            alpha=0.9,
            s=70,
            label=f"{lbl} (Test)",
        )

    ev1 = pca.explained_variance_ratio_[0]
    ev2 = pca.explained_variance_ratio_[1]
    ax.set_title("PCA Projection of Usage Profiles (2D Latent Space)", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Principal Component 1 ({ev1:.1%} Explained Variance)", fontsize=10)
    ax.set_ylabel(f"Principal Component 2 ({ev2:.1%} Explained Variance)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_centroids(bundle: dict[str, Any], path: Path | str) -> None:
    """
    Generate grouped bar chart of standardized centroid coordinates per profile and feature.

    Parameters
    ----------
    bundle : dict[str, Any]
        Fitted clustering bundle from fit_profile_model.
    path : Path or str
        Destination image path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kmeans = bundle["pipeline"].named_steps["kmeans"]
    scaled_centers = kmeans.cluster_centers_  # shape: (k, n_features)
    feature_cols = bundle["feature_cols"]

    # Order profiles from lowest to highest overall intensity
    labels_order = list(bundle["centroids_original_units"].index)

    # Map labels to cluster_ids
    lbl_to_cid = {v: k for k, v in bundle["label_map"].items()}
    ordered_cids = [lbl_to_cid[lbl] for lbl in labels_order]

    x = np.arange(len(labels_order))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(feature_cols))
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, (feat, offset, color) in enumerate(zip(feature_cols, offsets, colors)):
        feat_vals = [scaled_centers[cid, idx] for cid in ordered_cids]
        ax.bar(x + offset, feat_vals, width=width, label=feat, color=color, edgecolor="black", alpha=0.85)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title("Standardized Cluster Centroids by Usage Persona (Z-Scores)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Usage Profile Persona", fontsize=11)
    ax.set_ylabel("Standardized Value (Z-Score)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_order, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
