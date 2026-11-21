"""
Training and evaluation engine for PocketPilot regression tasks.
"""

from typing import Any
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from pocketpilot.config import RANDOM_SEED, TARGET_BATTERY, TARGET_DATA, TARGET_SCREEN
from pocketpilot.evaluation.regression_metrics import regression_metrics
from pocketpilot.features.input_modes import assert_no_leakage, get_feature_columns
from pocketpilot.models.model_zoo import get_candidate_models

TASK_TARGET_MAP: dict[str, str] = {
    "battery": TARGET_BATTERY,
    "data": TARGET_DATA,
    "screen_time": TARGET_SCREEN,
}

TASK_RAW_INPUTS_MAP: dict[str, list[str]] = {
    "battery": ["device_model", "app_usage_min", "screen_on_hours", "apps_installed"],
    "data": ["device_model", "app_usage_min", "screen_on_hours", "apps_installed"],
    "screen_time": ["device_model", "app_usage_min", "apps_installed"],
}


def train_task(
    task: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fast: bool = False,
) -> dict[str, Any]:
    """
    Train and benchmark candidate regression models for a single task using cross-validation.

    Methodology & Leakage Prevention:
    1. Feature leakage assertion: Verifies that no forbidden or target-derived columns are present.
    2. Training strictly on train_df: GridSearchCV performs K-Fold CV exclusively on the training partition.
    3. Model selection rule:
       # SELECT the final model by LOWEST CV MAE on the training split. The test split must NEVER influence selection.
       The test split is evaluated only once on the refitted models for reporting purposes.
    4. Baseline policy: 'mean_baseline' is recorded for reference/benchmarking but is excluded from selection.

    Parameters
    ----------
    task : str
        Modeling task: 'battery', 'data', or 'screen_time'.
    train_df : pd.DataFrame
        Cleaned training partition.
    test_df : pd.DataFrame
        Cleaned held-out test partition.
    fast : bool, default=False
        If True, uses 3-fold CV and small search grids for fast testing.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'task': task name
        - 'target': target column name
        - 'raw_features': list of raw input columns
        - 'candidates': dict mapping model_name -> fitted GridSearchCV instance
        - 'results': DataFrame summarizing CV and test metrics for every candidate
        - 'selected_model_name': name of the winning model
        - 'selected_model': refitted best Pipeline of the winning model
    """
    if task not in TASK_TARGET_MAP:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {list(TASK_TARGET_MAP.keys())}")

    target_col = TASK_TARGET_MAP[task]
    raw_inputs = TASK_RAW_INPUTS_MAP[task]

    # Verify leakage rules on the official forecast feature columns
    official_features = get_feature_columns(task, mode="forecast")
    assert_no_leakage(official_features, task, mode="forecast")

    X_train = train_df[raw_inputs]
    y_train = train_df[target_col]
    X_test = test_df[raw_inputs]
    y_test = test_df[target_col]

    candidates = get_candidate_models(task, fast=fast)
    n_splits = 3 if fast else 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fitted_candidates: dict[str, GridSearchCV] = {}
    results_rows: list[dict[str, Any]] = []

    for name, (pipe, grid_params) in candidates.items():
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=grid_params,
            scoring="neg_mean_absolute_error",
            cv=cv,
            refit=True,
            n_jobs=-1,
        )

        # Fit strictly on training split
        grid_search.fit(X_train, y_train)
        fitted_candidates[name] = grid_search

        # CV metrics on training split
        cv_mae_mean = float(-grid_search.best_score_)
        cv_mae_std = float(grid_search.cv_results_["std_test_score"][grid_search.best_index_])

        # Test set evaluation (computed once after fitting)
        y_test_pred = grid_search.predict(X_test)
        test_metrics = regression_metrics(y_test, y_test_pred)

        results_rows.append({
            "task": task,
            "model": name,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "best_params": grid_search.best_params_,
        })

    results_df = pd.DataFrame(results_rows)

    # SELECT the final model by LOWEST CV MAE on the training split. The test split must NEVER influence selection.
    # Exclude reference 'mean_baseline' from being selected
    eligible = results_df[results_df["model"] != "mean_baseline"]
    best_row = eligible.loc[eligible["cv_mae_mean"].idxmin()]
    selected_model_name = str(best_row["model"])

    results_df["is_selected"] = results_df["model"] == selected_model_name
    selected_pipeline = fitted_candidates[selected_model_name].best_estimator_

    return {
        "task": task,
        "target": target_col,
        "raw_features": raw_inputs,
        "candidates": fitted_candidates,
        "results": results_df,
        "selected_model_name": selected_model_name,
        "selected_model": selected_pipeline,
    }
