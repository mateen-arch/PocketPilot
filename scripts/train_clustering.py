"""
Training and serialization script for usage-profile clustering (K-Means / GMM).

Executes k-sweep evaluation across CLUSTER_K_REPORT_RANGE, selects optimal k based on
highest silhouette score within CLUSTER_K_SELECT_RANGE (3 to 6) on training split,
fits the K-Means profiling bundle, and updates models/metadata.json.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import joblib
import pandas as pd
import sklearn

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pocketpilot.config import (
    CLUSTER_K_REPORT_RANGE,
    CLUSTERING_MODEL_PATH,
    MODELS_DIR,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    RANDOM_SEED,
)
from pocketpilot.models.clustering import (
    evaluate_k_range,
    fit_profile_model,
    select_k,
)


def main() -> None:
    print("=" * 80)
    print("PocketPilot: Usage-Profile Clustering (Profile-Analysis Mode)")
    print("=" * 80)

    if not PROCESSED_TRAIN_PATH.exists() or not PROCESSED_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Processed datasets not found at {PROCESSED_TRAIN_PATH} and {PROCESSED_TEST_PATH}. "
            "Run scripts/prepare_data.py first."
        )

    print(f"Loading training data from {PROCESSED_TRAIN_PATH}...")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    print(f"Training partition loaded: {len(train_df)} rows.\n")

    # 1. Evaluate k range on training split
    print(f"Evaluating k range {CLUSTER_K_REPORT_RANGE} for K-Means and GMM...")
    k_eval_df = evaluate_k_range(train_df, CLUSTER_K_REPORT_RANGE)

    print("\nCluster Selection Benchmark Table:")
    print("-" * 80)
    formatted_eval = k_eval_df.copy()
    formatted_eval["silhouette"] = formatted_eval["silhouette"].apply(lambda v: f"{v:.4f}")
    formatted_eval["davies_bouldin"] = formatted_eval["davies_bouldin"].apply(lambda v: f"{v:.4f}")
    formatted_eval["inertia"] = formatted_eval["inertia"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "-")
    formatted_eval["bic"] = formatted_eval["bic"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "-")
    print(formatted_eval.to_string(index=False))
    print("-" * 80)

    # 2. Select optimal k
    chosen_k = select_k(k_eval_df)
    km_match = k_eval_df[(k_eval_df["method"] == "kmeans") & (k_eval_df["k"] == chosen_k)].iloc[0]
    sil_val = float(km_match["silhouette"])
    db_val = float(km_match["davies_bouldin"])

    print(f"\nOptimal cluster count selected: k = {chosen_k}")
    print(f"  Training Silhouette Score:    {sil_val:.4f}")
    print(f"  Training Davies-Bouldin Index: {db_val:.4f}")

    # 3. Fit profiling bundle on training split
    print(f"\nFitting production clustering bundle with k = {chosen_k}...")
    bundle = fit_profile_model(train_df, chosen_k)

    print("\nAssigned Neutral Persona Labels:")
    for cid, label in sorted(bundle["label_map"].items()):
        print(f"  Cluster ID {cid}: '{label}'")

    print("\nCluster Centroids in Original Real-World Units:")
    print("-" * 80)
    print(bundle["centroids_original_units"].to_string())
    print("-" * 80)

    # 4. Serialize clustering bundle
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, CLUSTERING_MODEL_PATH)
    print(f"\nSaved clustering bundle -> {CLUSTERING_MODEL_PATH}")

    # 5. Update models/metadata.json without touching regression entries
    metadata_path = MODELS_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["clustering"] = {
        "selected_k": chosen_k,
        "selection_rule": "highest silhouette score within CLUSTER_K_SELECT_RANGE (3 to 6) on training split",
        "silhouette": sil_val,
        "davies_bouldin": db_val,
        "label_map": {str(cid): lbl for cid, lbl in bundle["label_map"].items()},
        "feature_list": bundle["feature_cols"],
        "sklearn_version": sklearn.__version__,
        "random_seed": RANDOM_SEED,
        "training_date": datetime.now(timezone.utc).isoformat(),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated models metadata -> {metadata_path}")
    print("=" * 80)
    print("Clustering training complete successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
