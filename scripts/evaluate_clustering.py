"""
Evaluation and reporting script for usage-profile clustering.

Generates metric tables and diagnostic visual artifacts:
- reports/metrics/clustering_k_selection.csv
- reports/metrics/clustering_quality.csv
- reports/metrics/clustering_profiles_train.csv
- reports/metrics/clustering_profiles_test.csv
- reports/metrics/clustering_vs_behavior_class.csv
- reports/figures/clustering_k_selection.png
- reports/figures/clustering_pca.png
- reports/figures/clustering_centroids.png
"""

from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.metrics import adjusted_rand_score

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pocketpilot.config import (
    CLUSTER_K_REPORT_RANGE,
    CLUSTERING_MODEL_PATH,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
)
from pocketpilot.evaluation.clustering_metrics import (
    cluster_summary,
    cluster_vs_behavior_class,
    clustering_quality,
)
from pocketpilot.evaluation.plots import (
    plot_centroids,
    plot_k_selection,
    plot_pca_clusters,
)
from pocketpilot.models.clustering import (
    evaluate_k_range,
    predict_profile,
)


def main() -> None:
    print("=" * 80)
    print("PocketPilot: Evaluating Usage-Profile Clustering on Train & Test Splits")
    print("=" * 80)

    reports_metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    reports_figures_dir = PROJECT_ROOT / "reports" / "figures"
    reports_metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_figures_dir.mkdir(parents=True, exist_ok=True)

    if not CLUSTERING_MODEL_PATH.exists():
        raise FileNotFoundError(f"Clustering model not found at {CLUSTERING_MODEL_PATH}. Run scripts/train_clustering.py first.")

    bundle = joblib.load(CLUSTERING_MODEL_PATH)
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    # 1. Full k-selection metrics table
    print("[1/5] Generating k-selection evaluation table across CLUSTER_K_REPORT_RANGE...")
    k_eval_df = evaluate_k_range(train_df, CLUSTER_K_REPORT_RANGE)
    k_eval_path = reports_metrics_dir / "clustering_k_selection.csv"
    k_eval_df.to_csv(k_eval_path, index=False)
    print(f"      Saved k-selection table -> {k_eval_path}")

    # 2. Cluster assignments
    print("[2/5] Assigning records to usage profiles for train and test splits...")
    train_preds = predict_profile(bundle, train_df)
    test_preds = predict_profile(bundle, test_df)

    # 3. Clustering quality metrics (Silhouette, Davies-Bouldin, ARI)
    print("[3/5] Computing separation metrics and Adjusted Rand Index...")
    feature_cols = bundle["feature_cols"]
    scaler = bundle["pipeline"].named_steps["scaler"]
    X_train_scaled = scaler.transform(train_df[feature_cols])
    X_test_scaled = scaler.transform(test_df[feature_cols])

    train_qual = clustering_quality(X_train_scaled, train_preds["cluster_id"])
    test_qual = clustering_quality(X_test_scaled, test_preds["cluster_id"])

    ari_train = float(adjusted_rand_score(train_df["behavior_class"], train_preds["profile_label"]))
    ari_test = float(adjusted_rand_score(test_df["behavior_class"], test_preds["profile_label"]))

    quality_df = pd.DataFrame([
        {
            "split": "train",
            "silhouette": train_qual["silhouette"],
            "davies_bouldin": train_qual["davies_bouldin"],
            "adjusted_rand_index": ari_train,
        },
        {
            "split": "test",
            "silhouette": test_qual["silhouette"],
            "davies_bouldin": test_qual["davies_bouldin"],
            "adjusted_rand_index": ari_test,
        },
    ])
    quality_path = reports_metrics_dir / "clustering_quality.csv"
    quality_df.to_csv(quality_path, index=False)
    print(f"      Saved clustering quality table -> {quality_path}")
    print("\nClustering Quality Summary:")
    print(quality_df.to_string(index=False))

    # 4. Profile summaries in original physical units
    print("\n[4/5] Generating profile persona summaries in original units...")
    train_summary = cluster_summary(train_df, train_preds["profile_label"])
    test_summary = cluster_summary(test_df, test_preds["profile_label"])

    train_sum_path = reports_metrics_dir / "clustering_profiles_train.csv"
    test_sum_path = reports_metrics_dir / "clustering_profiles_test.csv"
    train_summary.to_csv(train_sum_path, index=False)
    test_summary.to_csv(test_sum_path, index=False)
    print(f"      Saved train profile summary -> {train_sum_path}")
    print(f"      Saved test profile summary  -> {test_sum_path}")

    # 5. Crosstab against synthetic behavior_class
    print("[5/5] Generating crosstabs against synthetic behavior_class...")
    ct_train, _ = cluster_vs_behavior_class(train_df, train_preds["profile_label"])
    ct_test, _ = cluster_vs_behavior_class(test_df, test_preds["profile_label"])

    ct_train_reset = ct_train.reset_index()
    ct_train_reset.insert(0, "split", "train")
    ct_test_reset = ct_test.reset_index()
    ct_test_reset.insert(0, "split", "test")

    ct_combined = pd.concat([ct_train_reset, ct_test_reset], ignore_index=True)
    crosstab_path = reports_metrics_dir / "clustering_vs_behavior_class.csv"
    ct_combined.to_csv(crosstab_path, index=False)
    print(f"      Saved crosstab comparison -> {crosstab_path}")

    # 6. Figures
    print("\nGenerating diagnostic clustering figures...")
    fig_k_path = reports_figures_dir / "clustering_k_selection.png"
    fig_pca_path = reports_figures_dir / "clustering_pca.png"
    fig_cent_path = reports_figures_dir / "clustering_centroids.png"

    plot_k_selection(k_eval_df, fig_k_path)
    print(f"      Saved k-selection plot  -> {fig_k_path}")

    plot_pca_clusters(bundle, train_df, test_df, fig_pca_path)
    print(f"      Saved PCA clusters plot -> {fig_pca_path}")

    plot_centroids(bundle, fig_cent_path)
    print(f"      Saved centroids plot    -> {fig_cent_path}")

    print("=" * 80)
    print("Clustering evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
