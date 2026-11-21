"""
Training and serialization script for the budget-overrun risk layer.

Fits empirical out-of-fold residual distributions for each regression task on the training split,
selects between additive and relative residual modeling based on heteroscedasticity,
and serializes the unified risk model to models/risk_model.joblib.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pocketpilot.config import (
    MODEL_PATHS,
    MODELS_DIR,
    PROCESSED_TRAIN_PATH,
    RANDOM_SEED,
    RISK_MODEL_PATH,
)
from pocketpilot.risk.probabilistic_risk import fit_risk_model

TASKS = ["battery", "data", "screen_time"]


def main() -> None:
    print("=" * 80)
    print("PocketPilot: Training Budget-Overrun Risk Layer (Feature F5)")
    print("=" * 80)

    if not PROCESSED_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {PROCESSED_TRAIN_PATH}.")

    print(f"Loading training data from {PROCESSED_TRAIN_PATH}...")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    print(f"Training partition loaded: {len(train_df)} rows.\n")

    risk_bundles: dict[str, dict] = {}
    risk_metadata: dict[str, dict] = {}

    for task in TASKS:
        model_path = MODEL_PATHS[task]
        if not model_path.exists():
            raise FileNotFoundError(f"Selected model for task '{task}' not found at {model_path}. Run train_all.py first.")

        print(f"Fitting risk model for task: {task.upper()}...")
        pipeline = joblib.load(model_path)
        bundle = fit_risk_model(task, pipeline, train_df)
        risk_bundles[task] = bundle

        res = bundle["residuals"]
        q05 = float(np.percentile(res, 5))
        q10 = float(np.percentile(res, 10))
        q50 = float(np.percentile(res, 50))
        q90 = float(np.percentile(res, 90))
        q95 = float(np.percentile(res, 95))

        print(f"  Chosen Variant: {bundle['variant'].upper()} (Spearman rho = {bundle['rho']:.4f})")
        print(f"  Sample Count:   {bundle['n_residuals']} out-of-fold residuals")
        print(f"  Quantiles:      q05={q05:.4f}, q10={q10:.4f}, q50={q50:.4f}, q90={q90:.4f}, q95={q95:.4f}\n")

        risk_metadata[task] = {
            "variant": bundle["variant"],
            "rho": bundle["rho"],
            "n_residuals": bundle["n_residuals"],
            "residual_quantiles": {
                "q05": q05,
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "q95": q95,
            },
            "method_description": (
                "Empirical residual sampling using 5-fold cross-validated out-of-fold "
                "predictions on the training partition."
            ),
            "sklearn_version": sklearn.__version__,
            "seed": RANDOM_SEED,
            "training_date": datetime.now(timezone.utc).isoformat(),
        }

    # Save risk bundles
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(risk_bundles, RISK_MODEL_PATH)
    print(f"Saved risk bundles to {RISK_MODEL_PATH}")

    # Update models/metadata.json
    metadata_path = MODELS_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["risk"] = risk_metadata

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated metadata in {metadata_path}")
    print("=" * 80)
    print("Risk model training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
