"""
Tests for model training, candidate selection, persistence, and regression validation.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from pocketpilot.config import PROCESSED_TEST_PATH, PROCESSED_TRAIN_PATH
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP, train_task

TASKS = ["battery", "data", "screen_time"]


@pytest.fixture(scope="module")
def dataset_splits():
    """Load train and test partitions from processed storage."""
    assert PROCESSED_TRAIN_PATH.exists() and PROCESSED_TEST_PATH.exists(), (
        "Processed datasets missing. Run scripts/prepare_data.py first."
    )
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    return train_df, test_df


@pytest.mark.parametrize("task", TASKS)
def test_pipeline_raw_inputs_and_predictions(task, dataset_splits):
    """Verify that each trained pipeline accepts ONLY raw forecast columns and returns finite predictions."""
    train_df, test_df = dataset_splits
    res = train_task(task, train_df, test_df, fast=True)

    raw_inputs = TASK_RAW_INPUTS_MAP[task]
    selected_pipe = res["selected_model"]

    # Provide a DataFrame containing ONLY the raw forecast input columns
    X_test_raw = test_df[raw_inputs]
    preds = selected_pipe.predict(X_test_raw)

    assert len(preds) == len(test_df), f"Expected {len(test_df)} predictions, got {len(preds)}"
    assert np.all(np.isfinite(preds)), "Predictions contained non-finite (NaN or Inf) values"


@pytest.mark.parametrize("task", TASKS)
def test_pipeline_serialization_parity(task, dataset_splits, tmp_path):
    """Verify that a serialized-then-loaded pipeline produces identical predictions."""
    train_df, test_df = dataset_splits
    res = train_task(task, train_df, test_df, fast=True)
    raw_inputs = TASK_RAW_INPUTS_MAP[task]
    selected_pipe = res["selected_model"]

    X_test_raw = test_df[raw_inputs]
    in_memory_preds = selected_pipe.predict(X_test_raw)

    # Save to temporary path (never into models/ or reports/)
    dump_path = tmp_path / f"{task}_model.joblib"
    joblib.dump(selected_pipe, dump_path)

    loaded_pipe = joblib.load(dump_path)
    loaded_preds = loaded_pipe.predict(X_test_raw)

    np.testing.assert_allclose(
        in_memory_preds,
        loaded_preds,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Discrepancy detected between in-memory and loaded pipeline for task '{task}'",
    )


@pytest.mark.parametrize("task", TASKS)
def test_predictions_deterministic_for_same_seed(task, dataset_splits):
    """Verify that training with the same random seed produces deterministic predictions."""
    train_df, test_df = dataset_splits
    res1 = train_task(task, train_df, test_df, fast=True)
    res2 = train_task(task, train_df, test_df, fast=True)

    raw_inputs = TASK_RAW_INPUTS_MAP[task]
    preds1 = res1["selected_model"].predict(test_df[raw_inputs])
    preds2 = res2["selected_model"].predict(test_df[raw_inputs])

    np.testing.assert_allclose(
        preds1,
        preds2,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Non-deterministic predictions detected across runs for task '{task}'",
    )


@pytest.mark.parametrize("task", TASKS)
def test_selected_model_beats_mean_baseline(task, dataset_splits):
    """Verify that the selected model beats the mean_baseline and achieves R2 > 0.5."""
    train_df, test_df = dataset_splits
    res = train_task(task, train_df, test_df, fast=True)
    results_df = res["results"]

    sel_row = results_df[results_df["is_selected"]].iloc[0]
    base_row = results_df[results_df["model"] == "mean_baseline"].iloc[0]

    sel_r2 = float(sel_row["test_r2"])
    base_r2 = float(base_row["test_r2"])

    assert sel_r2 > 0.5, f"Task '{task}': Selected model R2 {sel_r2:.4f} did not exceed threshold of 0.5"
    assert sel_r2 > base_r2, f"Task '{task}': Selected model R2 {sel_r2:.4f} failed to beat mean_baseline {base_r2:.4f}"
    assert float(sel_row["test_mae"]) < float(base_row["test_mae"]), (
        f"Task '{task}': Selected model MAE was not lower than mean_baseline"
    )


@pytest.mark.parametrize("task", TASKS)
def test_selected_model_matches_lowest_cv_mae(task, dataset_splits):
    """Verify that selection is governed strictly by lowest CV MAE on train split, proving zero test leakage."""
    train_df, test_df = dataset_splits
    res = train_task(task, train_df, test_df, fast=True)
    results_df = res["results"]

    # Exclude reference baseline
    eligible = results_df[results_df["model"] != "mean_baseline"]
    lowest_cv_mae_name = eligible.loc[eligible["cv_mae_mean"].idxmin()]["model"]

    assert res["selected_model_name"] == lowest_cv_mae_name, (
        f"Selection mismatch for task '{task}': chosen '{res['selected_model_name']}', "
        f"but lowest CV MAE model was '{lowest_cv_mae_name}'"
    )
