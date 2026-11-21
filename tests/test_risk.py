"""
Tests for budget-overrun risk layer (deterministic thresholds, probabilistic risk, calibration, and intervals).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from pocketpilot.data.cleaning import clean_ds1
from pocketpilot.data.loader import load_ds1
from pocketpilot.data.splitting import split_ds1
from pocketpilot.evaluation.classification_metrics import interval_coverage
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP, TASK_TARGET_MAP, train_task
from pocketpilot.risk.probabilistic_risk import (
    assess_budget,
    choose_residual_variant,
    fit_risk_model,
    overrun_probability,
    predict_interval,
)
from pocketpilot.risk.threshold_risk import risk_level, risk_levels


@pytest.fixture(scope="module")
def memory_pipeline():
    """Build train/test data and fast-trained models in memory without reading data/processed or models/."""
    raw = load_ds1()
    clean = clean_ds1(raw)
    train_df, test_df = split_ds1(clean)

    tasks_trained = {}
    risk_bundles = {}
    for task in ["battery", "data", "screen_time"]:
        res = train_task(task, train_df, test_df, fast=True)
        tasks_trained[task] = res
        risk_bundles[task] = fit_risk_model(task, res["selected_model"], train_df)

    return train_df, test_df, tasks_trained, risk_bundles


def test_threshold_risk_level_boundaries():
    """Verify exact threshold boundary transitions: Safe (<0.8*B), Watch (0.8*B..B), Likely exceed (>=B)."""
    budget = 1000.0

    # Just below 0.8 * budget
    assert risk_level(799.99, budget) == "Safe"
    # Exactly 0.8 * budget
    assert risk_level(800.0, budget) == "Watch"
    # Inside Watch window
    assert risk_level(850.0, budget) == "Watch"
    # Just below budget
    assert risk_level(999.99, budget) == "Watch"
    # Exactly budget
    assert risk_level(1000.0, budget) == "Likely exceed"
    # Well above budget
    assert risk_level(1200.0, budget) == "Likely exceed"

    # Vectorized check
    arr = np.array([799.0, 800.0, 950.0, 1000.0, 1100.0])
    levels = risk_levels(arr, budget)
    np.testing.assert_array_equal(levels, ["Safe", "Watch", "Watch", "Likely exceed", "Likely exceed"])


def test_risk_input_validation(memory_pipeline):
    """Verify ValueError is raised for non-positive budget, NaN budget, or NaN prediction."""
    _, _, _, risk_bundles = memory_pipeline
    bundle = risk_bundles["battery"]

    # Budget <= 0
    with pytest.raises(ValueError, match="strictly positive"):
        risk_level(500.0, 0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        risk_level(500.0, -100.0)
    with pytest.raises(ValueError, match="strictly positive"):
        assess_budget(bundle, 500.0, 0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        assess_budget(bundle, 500.0, -50.0)

    # NaN budget
    with pytest.raises(ValueError, match="finite"):
        risk_level(500.0, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        assess_budget(bundle, 500.0, float("nan"))

    # NaN y_hat
    with pytest.raises(ValueError, match="finite"):
        risk_level(float("nan"), 1000.0)
    with pytest.raises(ValueError, match="finite"):
        assess_budget(bundle, float("nan"), 1000.0)


def test_risk_bundle_fitted_on_train_split_only(memory_pipeline):
    """Verify bundle residual count matches training rows exactly (proving fit on training split only)."""
    train_df, _, _, risk_bundles = memory_pipeline
    for task in ["battery", "data", "screen_time"]:
        b = risk_bundles[task]
        assert b["n_residuals"] == len(train_df), (
            f"Task {task}: Expected n_residuals={len(train_df)}, got {b['n_residuals']}"
        )


def test_choose_residual_variant_logic():
    """Verify choose_residual_variant detects heteroscedastic spread vs homoscedastic constant spread."""
    rng = np.random.RandomState(42)
    oof_pred = np.linspace(100.0, 2000.0, 300)

    # 1. Growing spread: variance proportional to prediction -> relative
    growing_resid = oof_pred * 0.3 * rng.randn(300)
    variant_rel, rho_rel = choose_residual_variant(oof_pred, growing_resid)
    assert variant_rel == "relative"
    assert rho_rel > 0.2

    # 2. Constant spread: variance independent of prediction -> additive
    const_resid = 20.0 * rng.randn(300)
    variant_add, rho_add = choose_residual_variant(oof_pred, const_resid)
    assert variant_add == "additive"
    assert rho_add <= 0.2


def test_overrun_probability_monotonicity_and_bounds(memory_pipeline):
    """Verify overrun probability is strictly bounded in (0, 1) and monotonic in budget and y_hat."""
    _, _, _, risk_bundles = memory_pipeline
    bundle = risk_bundles["battery"]

    # 1. Bounded in (0, 1)
    p = overrun_probability(bundle, 1500.0, 1600.0)
    assert 0.0 < p < 1.0

    # 2. Decreases as budget increases
    p_low_b = overrun_probability(bundle, 1500.0, 1200.0)
    p_mid_b = overrun_probability(bundle, 1500.0, 1500.0)
    p_high_b = overrun_probability(bundle, 1500.0, 1800.0)
    assert p_low_b >= p_mid_b >= p_high_b

    # 3. Increases as y_hat increases
    p_low_y = overrun_probability(bundle, 1200.0, 1500.0)
    p_mid_y = overrun_probability(bundle, 1500.0, 1500.0)
    p_high_y = overrun_probability(bundle, 1800.0, 1500.0)
    assert p_low_y <= p_mid_y <= p_high_y


def test_probabilistic_calibration_and_clipping():
    """Verify median additive residual yields probability ~ 0.5, and extreme budgets hit theoretical clips."""
    # Construct a synthetic additive bundle with 501 symmetric residuals
    resids = np.linspace(-100.0, 100.0, 501)
    add_bundle = {
        "task": "battery",
        "variant": "additive",
        "rho": 0.05,
        "residuals": resids,
        "n_residuals": 501,
        "floor_value": 1.0,
        "seed": 42,
    }

    # At budget = y_hat + median(residuals), exactly half of residuals exceed budget
    med = float(np.median(resids))
    p_mid = overrun_probability(add_bundle, y_hat=1000.0, budget=1000.0 + med)
    assert abs(p_mid - 0.5) <= 0.05

    # At tiny budget, hits theoretical upper clip n / (n + 1)
    p_tiny = overrun_probability(add_bundle, y_hat=1000.0, budget=0.01)
    assert np.isclose(p_tiny, 501.0 / 502.0)

    # At huge budget, hits theoretical lower clip 1 / (n + 1)
    p_huge = overrun_probability(add_bundle, y_hat=1000.0, budget=1e8)
    assert np.isclose(p_huge, 1.0 / 502.0)


def test_prediction_interval_containment_and_nonnegativity(memory_pipeline):
    """Verify 90% interval is wider than 80%, intervals contain y_hat, and lower bound >= 0."""
    _, _, _, risk_bundles = memory_pipeline
    bundle = risk_bundles["battery"]
    y_hat = 1500.0

    low80, up80 = predict_interval(bundle, y_hat, level=0.8)
    low90, up90 = predict_interval(bundle, y_hat, level=0.9)

    assert (up90 - low90) >= (up80 - low80), "90% interval must be at least as wide as 80% interval"
    assert low80 <= y_hat <= up80, f"80% interval [{low80}, {up80}] does not contain y_hat {y_hat}"
    assert low90 <= y_hat <= up90, f"90% interval [{low90}, {up90}] does not contain y_hat {y_hat}"
    assert low80 >= 0.0 and low90 >= 0.0, "Lower bound must never be negative"


def test_empirical_interval_coverage_on_test_split(memory_pipeline):
    """Verify empirical 80% interval coverage on the TEST split is between 0.70 and 0.90 for all tasks."""
    _, test_df, tasks_trained, risk_bundles = memory_pipeline

    for task in ["battery", "data", "screen_time"]:
        b = risk_bundles[task]
        model = tasks_trained[task]["selected_model"]
        raw_inputs = TASK_RAW_INPUTS_MAP[task]
        target_col = TASK_TARGET_MAP[task]

        y_test_pred = model.predict(test_df[raw_inputs])
        y_test_true = test_df[target_col].values

        low80, up80 = predict_interval(b, y_test_pred, level=0.8)
        cov = interval_coverage(y_test_true, low80, up80)

        assert 0.70 <= cov <= 0.90, (
            f"Task '{task}': Empirical 80% coverage {cov:.4f} outside expected range [0.70, 0.90]"
        )


def test_assess_budget_keys_and_serialization(memory_pipeline, tmp_path):
    """Verify assess_budget dictionary schema, serialization parity, and deterministic evaluation."""
    _, _, _, risk_bundles = memory_pipeline
    bundle = risk_bundles["battery"]
    y_hat, budget = 1600.0, 1800.0

    out1 = assess_budget(bundle, y_hat, budget)
    expected_keys = {"predicted", "budget", "level", "p_overrun", "interval_80", "interval_90"}
    assert set(out1.keys()) == expected_keys

    # Serialization parity
    dump_path = tmp_path / "test_risk_model.joblib"
    joblib.dump({"battery": bundle}, dump_path)
    loaded_dict = joblib.load(dump_path)
    out2 = assess_budget(loaded_dict["battery"], y_hat, budget)

    assert out1["predicted"] == out2["predicted"]
    assert out1["budget"] == out2["budget"]
    assert out1["level"] == out2["level"]
    assert np.isclose(out1["p_overrun"], out2["p_overrun"])
    assert np.allclose(out1["interval_80"], out2["interval_80"])
    assert np.allclose(out1["interval_90"], out2["interval_90"])
