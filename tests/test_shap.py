"""
Unit and integration tests for SHAP explainability utilities.

All tests operate in-memory on fast-trained candidate models without modifying
disk artifacts or depending on files in models/ or reports/.
"""

import numpy as np
import pandas as pd
import pytest

from pocketpilot.data.cleaning import clean_ds1
from pocketpilot.data.loader import load_ds1
from pocketpilot.data.splitting import split_ds1
from pocketpilot.explain.shap_utils import (
    compute_shap,
    explain_one,
    get_final_estimator,
    group_device_features,
    mean_abs_importance,
    transform_for_explanation,
)
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP, TASK_TARGET_MAP, train_task


@pytest.fixture(scope="module")
def shared_data_and_models():
    """
    Build clean split datasets and fast candidate pipelines in-memory.
    """
    raw_df = load_ds1()
    clean_df = clean_ds1(raw_df)
    train_df, test_df = split_ds1(clean_df)

    res_battery = train_task("battery", train_df, test_df, fast=True)
    res_data = train_task("data", train_df, test_df, fast=True)
    res_screen = train_task("screen_time", train_df, test_df, fast=True)

    # Use XGBoost for battery and screen_time, and RandomForest for data
    # (fast mode may select linear model as winning model, which is not tree-based)
    xgb_battery = res_battery["candidates"]["xgboost"].best_estimator_
    rf_data = res_data["candidates"]["random_forest"].best_estimator_
    xgb_screen = res_screen["candidates"]["xgboost"].best_estimator_
    linear_battery = res_battery["candidates"]["linear"].best_estimator_

    return {
        "train_df": train_df,
        "test_df": test_df,
        "battery_pipe": xgb_battery,
        "data_pipe": rf_data,
        "screen_pipe": xgb_screen,
        "linear_pipe": linear_battery,
    }


def test_shap_additivity(shared_data_and_models):
    """
    Verify SHAP additivity: base_value + row sum of SHAP values equals pipeline prediction
    within 1e-3 * mean target.
    """
    test_df = shared_data_and_models["test_df"]

    # Test battery (XGBoost)
    bat_pipe = shared_data_and_models["battery_pipe"]
    X_bat = test_df[TASK_RAW_INPUTS_MAP["battery"]].head(10)
    shap_bat, base_bat, _ = compute_shap(bat_pipe, X_bat)
    preds_bat = bat_pipe.predict(X_bat)
    sums_bat = base_bat + shap_bat.sum(axis=1).values
    mean_target_bat = test_df[TASK_TARGET_MAP["battery"]].mean()
    tol_bat = 1e-3 * mean_target_bat
    np.testing.assert_allclose(preds_bat, sums_bat, atol=tol_bat)

    # Test data (RandomForest)
    data_pipe = shared_data_and_models["data_pipe"]
    X_data = test_df[TASK_RAW_INPUTS_MAP["data"]].head(10)
    shap_data, base_data, _ = compute_shap(data_pipe, X_data)
    preds_data = data_pipe.predict(X_data)
    sums_data = base_data + shap_data.sum(axis=1).values
    mean_target_data = test_df[TASK_TARGET_MAP["data"]].mean()
    tol_data = 1e-3 * mean_target_data
    np.testing.assert_allclose(preds_data, sums_data, atol=tol_data)


def test_shap_shape_and_no_leakage(shared_data_and_models):
    """
    Verify shap_df matches transformed matrix shape and columns, and forbidden
    leakage columns are never present.
    """
    test_df = shared_data_and_models["test_df"]

    for task, pipe_key in [("battery", "battery_pipe"), ("data", "data_pipe"), ("screen_time", "screen_pipe")]:
        pipe = shared_data_and_models[pipe_key]
        X = test_df[TASK_RAW_INPUTS_MAP[task]].head(5)
        shap_df, _, X_trans = compute_shap(pipe, X)

        assert shap_df.shape == X_trans.shape
        assert list(shap_df.columns) == list(X_trans.columns)

        cols = set(shap_df.columns)
        # Forbidden in all tasks
        assert "behavior_class" not in cols
        assert "battery_mah" not in cols
        assert "data_mb" not in cols

        # screen_on_hours forbidden only in screen_time task
        if task == "screen_time":
            assert "screen_on_hours" not in cols
            assert "app_share_of_screen" not in cols
        else:
            assert "screen_on_hours" in cols


def test_group_device_features_preserves_sums(shared_data_and_models):
    """
    Verify group_device_features preserves row sums and consolidates one-hot columns
    into exactly one device_model column.
    """
    test_df = shared_data_and_models["test_df"]
    bat_pipe = shared_data_and_models["battery_pipe"]
    X = test_df[TASK_RAW_INPUTS_MAP["battery"]].head(10)
    shap_df, _, _ = compute_shap(bat_pipe, X)

    # 1. On DataFrame of signed SHAP values
    grouped_df = group_device_features(shap_df)
    assert "device_model" in grouped_df.columns
    device_cols = [c for c in grouped_df.columns if c.startswith("device_model_")]
    assert len(device_cols) == 0
    np.testing.assert_allclose(grouped_df.sum(axis=1), shap_df.sum(axis=1), atol=1e-4)

    # 2. On absolute-value table
    shap_abs = shap_df.abs()
    grouped_abs = group_device_features(shap_abs)
    assert "device_model" in grouped_abs.columns
    np.testing.assert_allclose(grouped_abs.sum(axis=1), shap_abs.sum(axis=1), atol=1e-4)

    # 3. On a single pd.Series
    single_series = shap_df.iloc[0]
    grouped_s = group_device_features(single_series)
    assert "device_model" in grouped_s.index
    np.testing.assert_allclose(grouped_s.sum(), single_series.sum(), atol=1e-3)


def test_mean_abs_importance(shared_data_and_models):
    """
    Verify mean_abs_importance returns a descending sorted Series.
    """
    test_df = shared_data_and_models["test_df"]
    bat_pipe = shared_data_and_models["battery_pipe"]
    X = test_df[TASK_RAW_INPUTS_MAP["battery"]].head(15)
    shap_df, _, _ = compute_shap(bat_pipe, X)

    imp_grouped = mean_abs_importance(shap_df, grouped=True)
    assert isinstance(imp_grouped, pd.Series)
    assert "device_model" in imp_grouped.index
    # Check descending sort
    assert imp_grouped.is_monotonic_decreasing

    imp_trans = mean_abs_importance(shap_df, grouped=False)
    assert imp_trans.is_monotonic_decreasing
    assert len(imp_trans) > len(imp_grouped)


def test_explain_one(shared_data_and_models):
    """
    Verify explain_one returns prediction, base_value, and top_k contributions
    sorted by absolute SHAP value matching row values.
    """
    test_df = shared_data_and_models["test_df"]
    bat_pipe = shared_data_and_models["battery_pipe"]
    row_df = test_df[TASK_RAW_INPUTS_MAP["battery"]].iloc[[0]]

    res = explain_one(bat_pipe, row_df, top_k=5)

    assert "prediction" in res
    assert "base_value" in res
    assert "contributions" in res

    pred = res["prediction"]
    assert np.isfinite(pred)

    contribs = res["contributions"]
    assert len(contribs) == 5

    # Check sorting descending by absolute SHAP value
    abs_shaps = [abs(c[2]) for c in contribs]
    assert abs_shaps == sorted(abs_shaps, reverse=True)

    # Check input values match the row
    for feat, val, _ in contribs:
        if feat in row_df.columns:
            assert val == row_df[feat].iloc[0]

    # Passing multiple rows raises ValueError
    with pytest.raises(ValueError, match="single-row"):
        explain_one(bat_pipe, test_df[TASK_RAW_INPUTS_MAP["battery"]].head(2))


def test_linear_model_raises_value_error(shared_data_and_models):
    """
    Verify passing a linear model pipeline raises ValueError with clear message.
    """
    linear_pipe = shared_data_and_models["linear_pipe"]
    test_df = shared_data_and_models["test_df"]
    X = test_df[TASK_RAW_INPUTS_MAP["battery"]].head(3)

    with pytest.raises(ValueError, match="not a supported tree-based model"):
        get_final_estimator(linear_pipe)

    with pytest.raises(ValueError, match="not a supported tree-based model"):
        compute_shap(linear_pipe, X)


def test_shap_determinism(shared_data_and_models):
    """
    Verify SHAP computations are strictly deterministic across repeated calls.
    """
    test_df = shared_data_and_models["test_df"]
    bat_pipe = shared_data_and_models["battery_pipe"]
    X = test_df[TASK_RAW_INPUTS_MAP["battery"]].head(5)

    shap1, base1, trans1 = compute_shap(bat_pipe, X)
    shap2, base2, trans2 = compute_shap(bat_pipe, X)

    pd.testing.assert_frame_equal(shap1, shap2)
    assert base1 == base2
    pd.testing.assert_frame_equal(trans1, trans2)
