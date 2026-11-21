"""Tests for pipeline execution tracing and stage inspection.

Verifies:
- trace_pipeline produces exactly one entry per pipeline step.
- Traced prediction strictly equals direct pipeline.predict.
- Feature names after preprocessor match get_feature_names_out().
- Input DataFrame is never mutated by the tracing routine.
- Screen-time trace strictly excludes screen_on_hours and app_share_of_screen.
"""

import numpy as np
import pandas as pd
import pytest

from pocketpilot.config import PROCESSED_TEST_PATH, PROCESSED_TRAIN_PATH
from pocketpilot.inference.trace import trace_pipeline
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP, train_task


@pytest.fixture(scope="module")
def dataset_splits():
    """Load train and test partitions from processed storage."""
    assert PROCESSED_TRAIN_PATH.exists() and PROCESSED_TEST_PATH.exists(), (
        "Processed datasets missing. Run scripts/prepare_data.py first."
    )
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    return train_df, test_df


def test_trace_battery_pipeline(dataset_splits):
    """Verify trace_pipeline structure, feature names, parity, and immutability on battery pipeline."""
    train_df, test_df = dataset_splits
    res = train_task("battery", train_df, test_df, fast=True)
    pipe = res["selected_model"]

    raw_cols = TASK_RAW_INPUTS_MAP["battery"]
    X_input = test_df[raw_cols].iloc[0:1].copy()
    X_original = X_input.copy(deep=True)

    trace = trace_pipeline(pipe, X_input)

    # 1. One entry per pipeline step
    assert len(trace) == len(pipe.steps), f"Expected {len(pipe.steps)} trace entries, got {len(trace)}"

    # 2. Final prediction parity (output_row is rounded to 4 decimals)
    direct_pred = float(pipe.predict(X_input)[0])
    traced_pred = float(trace[-1]["output_row"][0]["prediction"])
    assert np.isclose(traced_pred, direct_pred, atol=1e-4)

    # 3. Preprocessor feature names match get_feature_names_out
    preproc_step = pipe.named_steps["preprocessor"]
    expected_feats = list(preproc_step.get_feature_names_out())
    # Preprocessor is index 1 in battery pipeline (engineer -> preprocessor -> model)
    preproc_trace = [t for t in trace if t["step_name"] == "preprocessor"][0]
    assert preproc_trace["feature_names"] == expected_feats

    # 4. Input DataFrame is not mutated
    pd.testing.assert_frame_equal(X_input, X_original)


def test_trace_screen_time_pipeline_no_screen_or_share(dataset_splits):
    """Verify screen-time pipeline trace parity and strict exclusion of screen_on_hours and app_share_of_screen."""
    train_df, test_df = dataset_splits
    res = train_task("screen_time", train_df, test_df, fast=True)
    pipe = res["selected_model"]

    raw_cols = TASK_RAW_INPUTS_MAP["screen_time"]
    X_input = test_df[raw_cols].iloc[0:1].copy()
    X_original = X_input.copy(deep=True)

    trace = trace_pipeline(pipe, X_input)

    # 1. One entry per step
    assert len(trace) == len(pipe.steps)

    # 2. Final prediction parity (output_row is rounded to 4 decimals)
    direct_pred = float(pipe.predict(X_input)[0])
    traced_pred = float(trace[-1]["output_row"][0]["prediction"])
    assert np.isclose(traced_pred, direct_pred, atol=1e-4)

    # 3. Input DataFrame immutability
    pd.testing.assert_frame_equal(X_input, X_original)

    # 4. Strict absence of forbidden columns across all stages
    forbidden = {"screen_on_hours", "app_share_of_screen"}
    for step_trace in trace:
        step_feats = step_trace.get("feature_names") or []
        for feat in step_feats:
            assert feat not in forbidden, (
                f"Forbidden column '{feat}' found in screen_time trace step '{step_trace['step_name']}'"
            )
        step_row = step_trace.get("output_row", [{}])[0]
        for key in step_row.keys():
            assert key not in forbidden, (
                f"Forbidden column '{key}' found in screen_time output values"
            )
