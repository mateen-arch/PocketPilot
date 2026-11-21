"""
Training script for all PocketPilot regression models (battery, data, screen_time).

Loads processed train and test datasets, trains candidate regressors via K-fold cross-validation,
evaluates on test data, selects the best model per task by lowest CV MAE on training data,
and persists selected models, candidate pipelines, and metadata.
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
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    RANDOM_SEED,
)
from pocketpilot.features.input_modes import get_feature_columns
from pocketpilot.models.trainer import train_task

TASKS = ["battery", "data", "screen_time"]
MODEL_FILENAME_MAP = {
    "battery": "battery_model.joblib",
    "data": "data_usage_model.joblib",
    "screen_time": "screen_time_model.joblib",
}


def main() -> None:
    print("=" * 80)
    print("PocketPilot: Training Regression Tasks (Battery, Data, Screen Time)")
    print("=" * 80)

    # Load preprocessed partitions
    if not PROCESSED_TRAIN_PATH.exists() or not PROCESSED_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Processed data splits not found at {PROCESSED_TRAIN_PATH} and {PROCESSED_TEST_PATH}. "
            "Please run scripts/prepare_data.py first."
        )

    print(f"Loading training data from {PROCESSED_TRAIN_PATH}...")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    print(f"Loading test data from {PROCESSED_TEST_PATH}...")
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    print(f"Dataset partitions loaded: {len(train_df)} train rows, {len(test_df)} test rows.\n")

    models_dir = PROJECT_ROOT / "models"
    candidates_dir = models_dir / "candidates"
    models_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict] = {}

    for task in TASKS:
        print("-" * 80)
        print(f"TASK: {task.upper()} REGRESSION")
        print("-" * 80)

        # Train candidates using 5-fold CV on train split
        task_res = train_task(task=task, train_df=train_df, test_df=test_df, fast=False)
        results_df = task_res["results"]
        selected_name = task_res["selected_model_name"]
        selected_pipe = task_res["selected_model"]

        # Display results table
        print("\nModel Comparison Table:")
        display_df = results_df[[
            "model",
            "cv_mae_mean",
            "cv_mae_std",
            "test_mae",
            "test_rmse",
            "test_r2",
            "is_selected",
        ]].copy()

        display_df["CV MAE"] = display_df.apply(
            lambda r: f"{r['cv_mae_mean']:.3f} +/- {r['cv_mae_std']:.3f}", axis=1
        )
        display_df["Test MAE"] = display_df["test_mae"].apply(lambda v: f"{v:.3f}")
        display_df["Test RMSE"] = display_df["test_rmse"].apply(lambda v: f"{v:.3f}")
        display_df["Test R2"] = display_df["test_r2"].apply(lambda v: f"{v:.4f}")
        display_df["Status"] = display_df["is_selected"].apply(
            lambda s: "[SELECTED]" if s else ""
        )

        cols_to_print = ["model", "CV MAE", "Test MAE", "Test RMSE", "Test R2", "Status"]
        print(display_df[cols_to_print].to_string(index=False))

        # 1. Save selected model
        selected_save_path = models_dir / MODEL_FILENAME_MAP[task]
        joblib.dump(selected_pipe, selected_save_path)
        print(f"\nSaved selected pipeline ({selected_name}) -> {selected_save_path}")

        # 2. Save all candidate pipelines
        for cand_name, grid_search in task_res["candidates"].items():
            cand_save_path = candidates_dir / f"{task}_{cand_name}.joblib"
            joblib.dump(grid_search.best_estimator_, cand_save_path)

        # 3. Compile metadata for selected model
        sel_row = results_df[results_df["is_selected"]].iloc[0]
        metadata[task] = {
            "selected_model": selected_name,
            "best_params": sel_row["best_params"],
            "raw_features": task_res["raw_features"],
            "forecast_features": get_feature_columns(task, mode="forecast"),
            "cv_mae_mean": float(sel_row["cv_mae_mean"]),
            "cv_mae_std": float(sel_row["cv_mae_std"]),
            "test_metrics": {
                "mae": float(sel_row["test_mae"]),
                "rmse": float(sel_row["test_rmse"]),
                "r2": float(sel_row["test_r2"]),
            },
            "sklearn_version": sklearn.__version__,
            "random_seed": RANDOM_SEED,
            "training_date": datetime.now(timezone.utc).isoformat(),
        }

    # Save metadata.json
    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved training metadata -> {metadata_path}")
    print("=" * 80)
    print("All tasks successfully trained and serialized!")
    print("=" * 80)


if __name__ == "__main__":
    main()
