"""
Tests for input mode constraints, leakage prevention, and unfitted preprocessing transformers.
"""

import numpy as np
import pytest

from pocketpilot.config import TARGET_BATTERY, TARGET_DATA, TARGET_SCREEN
from pocketpilot.data.cleaning import clean_ds1
from pocketpilot.data.loader import load_ds1
from pocketpilot.data.splitting import split_ds1
from pocketpilot.features.engineering import add_forecast_features
from pocketpilot.features.input_modes import (
    FORBIDDEN_IN_FORECAST,
    assert_no_leakage,
    get_feature_columns,
)
from pocketpilot.features.preprocessing import build_preprocessor


def test_forecast_features_exclude_forbidden_columns():
    """Verify that every forecast task contains none of the forbidden columns."""
    for task in ["battery", "data", "screen_time"]:
        cols = get_feature_columns(task, mode="forecast")
        overlap = set(cols) & FORBIDDEN_IN_FORECAST
        assert (
            not overlap
        ), f"Task '{task}' in forecast mode contains forbidden columns: {overlap}"


def test_task_specific_target_exclusions():
    """Verify specific exclusions for battery, data, and screen_time tasks."""
    battery_cols = get_feature_columns("battery", mode="forecast")
    data_cols = get_feature_columns("data", mode="forecast")
    screen_cols = get_feature_columns("screen_time", mode="forecast")
    cluster_cols = get_feature_columns("clustering", mode="profile")

    # Battery task must exclude data_mb and its own target battery_mah
    assert "data_mb" not in battery_cols
    assert "battery_mah" not in battery_cols

    # Data task must exclude battery_mah and its own target data_mb
    assert "battery_mah" not in data_cols
    assert "data_mb" not in data_cols

    # behavior_class must be in no feature list
    for lst in [battery_cols, data_cols, screen_cols, cluster_cols]:
        assert "behavior_class" not in lst

    # screen_time task must exclude screen_on_hours and app_share_of_screen
    assert "screen_on_hours" not in screen_cols
    assert "app_share_of_screen" not in screen_cols


def test_assert_no_leakage_enforcement():
    """Verify assert_no_leakage raises ValueError for leaks and passes for official feature sets."""
    # Official lists must pass without error
    for task in ["battery", "data", "screen_time"]:
        cols = get_feature_columns(task, mode="forecast")
        assert_no_leakage(cols, task, mode="forecast")

    # Injected leak: battery_mah in forecast mode
    with pytest.raises(ValueError, match="Forbidden columns present"):
        assert_no_leakage(["device_model", "app_usage_min", "battery_mah"], "battery", "forecast")

    # Injected leak: target data_mb in battery forecast
    with pytest.raises(ValueError, match="Forbidden columns present"):
        assert_no_leakage(["device_model", "app_usage_min", "data_mb"], "battery", "forecast")

    # Injected leak: app_share_of_screen in screen_time forecast
    with pytest.raises(ValueError, match="Forbidden columns present"):
        assert_no_leakage(["device_model", "app_usage_min", "app_share_of_screen"], "screen_time", "forecast")


def test_invalid_task_mode_combinations():
    """Verify get_feature_columns raises ValueError for invalid pairings or unknown tasks."""
    with pytest.raises(ValueError, match="forecast task and must be requested with mode='forecast'"):
        get_feature_columns("battery", mode="profile")

    with pytest.raises(ValueError, match="forecast task and must be requested with mode='forecast'"):
        get_feature_columns("data", mode="profile")

    with pytest.raises(ValueError, match="Profile-analysis mode only"):
        get_feature_columns("clustering", mode="forecast")

    with pytest.raises(ValueError, match="Invalid task"):
        get_feature_columns("unknown_task", mode="forecast")

    with pytest.raises(ValueError, match="Invalid mode"):
        get_feature_columns("battery", mode="invalid_mode")


def test_preprocessor_fitting_and_transformation():
    """Verify preprocessor is fit only on train split and maintains consistent feature output dimensions."""
    raw = load_ds1()
    clean = clean_ds1(raw)
    train_df, test_df = split_ds1(clean)

    # Add engineered forecast features
    train_feat = add_forecast_features(train_df)
    test_feat = add_forecast_features(test_df)
    full_feat = add_forecast_features(clean)

    cols = get_feature_columns("battery", mode="forecast")
    preprocessor = build_preprocessor(cols, scale_numeric=True)

    # Fit strictly on train split
    preprocessor.fit(train_feat[cols])

    # Check fitted scaler means
    num_cols = [c for c in cols if c != "device_model"]
    fitted_scaler = preprocessor.named_transformers_["num"]
    train_means = train_feat[num_cols].mean().values
    full_means = full_feat[num_cols].mean().values

    # Scaler mean must match train mean exactly (within numerical tolerance)
    assert np.allclose(fitted_scaler.mean_, train_means), "Fitted scaler mean does not match train set mean!"

    # Scaler mean must NOT match full dataset mean (verifies test data was not leaked into fit)
    assert not np.allclose(fitted_scaler.mean_, full_means), "Fitted scaler mean is identical to full dataset mean!"

    # Transformation checks
    X_train_trans = preprocessor.transform(train_feat[cols])
    X_test_trans = preprocessor.transform(test_feat[cols])

    assert X_train_trans.shape[0] == len(train_df)
    assert X_test_trans.shape[0] == len(test_df)
    assert (
        X_train_trans.shape[1] == X_test_trans.shape[1]
    ), f"Column count mismatch: train {X_train_trans.shape[1]} vs test {X_test_trans.shape[1]}"
