"""
Unit and integration tests for unified PocketPilot inference engine and battery survival estimator.

Part 1: In-memory tests using fast trained candidate models (zero dependency on models/ directory).
Part 2: Production tests using serialized model artifacts (safely skipped if models/ are missing).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pocketpilot import config
from pocketpilot.config import (
    CAPACITY_RANGE_MAH,
    DEFAULT_MIN_BATTERY_PCT,
    KNOWN_DEVICES,
    UNCONSTRAINED_BUDGET,
    WAKING_HOURS,
)
from pocketpilot.data.cleaning import clean_ds1
from pocketpilot.data.loader import load_ds1
from pocketpilot.data.splitting import split_ds1
from pocketpilot.inference.predictor import (
    forecast,
    load_artifacts,
    plan,
    profile,
    profile_history,
    survival,
    will_i_make_it,
)
from pocketpilot.models.clustering import fit_profile_model
from pocketpilot.models.trainer import train_task
from pocketpilot.planner.scenarios import training_bounds
from pocketpilot.risk.probabilistic_risk import fit_risk_model

# =====================================================================
# PART 1: IN-MEMORY TESTS (FAST MODELS, NO ARTIFACT DEPENDENCE)
# =====================================================================


@pytest.fixture(scope="module")
def in_memory_artifacts():
    """
    Construct fast in-memory models, risk bundles, and clustering artifacts.
    """
    raw_df = load_ds1()
    clean_df = clean_ds1(raw_df)
    train_df, test_df = split_ds1(clean_df)

    res_bat = train_task("battery", train_df, test_df, fast=True)
    res_data = train_task("data", train_df, test_df, fast=True)
    res_screen = train_task("screen_time", train_df, test_df, fast=True)

    # Use linear model candidate to guarantee strict mathematical monotonicity
    pipelines = {
        "battery": res_bat["candidates"]["linear"].best_estimator_,
        "data": res_data["candidates"]["linear"].best_estimator_,
        "screen_time": res_screen["candidates"]["linear"].best_estimator_,
    }

    risk_bundles = {
        "battery": fit_risk_model("battery", pipelines["battery"], train_df),
        "data": fit_risk_model("data", pipelines["data"], train_df),
        "screen_time": fit_risk_model("screen_time", pipelines["screen_time"], train_df),
    }

    clustering_bundle = fit_profile_model(train_df, k=3)
    bounds = training_bounds(train_df)

    return {
        "pipelines": pipelines,
        "risk": risk_bundles,
        "clustering": clustering_bundle,
        "training_bounds": bounds,
        "reference_df": train_df,
        "metadata": {},
        "warnings": [],
    }


def test_forecast_in_memory(in_memory_artifacts):
    """
    Verify forecast returns all required keys, finite values, and exhibits monotonicity.
    """
    inputs_low = {
        "device_model": "Google Pixel 5",
        "app_usage_min": 100,
        "screen_on_hours": 3.0,
        "apps_installed": 30,
    }
    inputs_high = {
        "device_model": "Google Pixel 5",
        "app_usage_min": 400,
        "screen_on_hours": 3.0,
        "apps_installed": 30,
    }
    budgets = {"battery_mah": 2000.0, "data_mb": 1200.0, "screen_hours": 5.0}

    res_low = forecast(in_memory_artifacts, inputs_low, budgets, screen_known=True)
    res_high = forecast(in_memory_artifacts, inputs_high, budgets, screen_known=True)

    for res in (res_low, res_high):
        assert "predictions" in res
        assert "risk" in res
        assert "screen_source" in res
        assert "screen_used" in res

        for task in ("battery_mah", "data_mb", "screen_hours"):
            assert np.isfinite(res["predictions"][task])

    # Monotonicity test: increasing app usage never lowers predicted battery drain
    # (guaranteed strictly monotone for linear candidate model)
    assert res_high["predictions"]["battery_mah"] >= res_low["predictions"]["battery_mah"]


def test_screen_known_toggle(in_memory_artifacts):
    """
    Verify screen_known False triggers model estimation, whereas True retains user input.
    """
    inputs = {
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 250,
        "screen_on_hours": 7.5,
        "apps_installed": 45,
    }
    budgets = {"battery_mah": 2000.0, "data_mb": 1000.0, "screen_hours": 6.0}

    # screen_known = True: retains 7.5
    res_user = forecast(in_memory_artifacts, inputs, budgets, screen_known=True)
    assert res_user["screen_source"] == "user"
    assert res_user["screen_used"] == 7.5
    assert res_user["predictions"]["screen_hours"] == 7.5
    assert "screen_model_estimate" in res_user
    assert res_user["screen_model_estimate"] != 7.5
    assert "budgets_used" in res_user

    # screen_known = False: estimated by screen_time pipeline
    res_est = forecast(in_memory_artifacts, inputs, budgets, screen_known=False)
    assert res_est["screen_source"] == "model_estimate"
    assert res_est["screen_used"] != 7.5
    assert np.isfinite(res_est["screen_used"])
    assert res_est["predictions"]["screen_hours"] == res_est["screen_model_estimate"]
    assert "screen_model_estimate" in res_est
    assert "budgets_used" in res_est


def test_survival_hand_calc_stub():
    """
    Verify hand-calculated survival example with stub forecast to decouple from model variance:
    - capacity: 4000 mAh
    - current: 80%, min: 15%
    - current_hour: 9.0, need_until: 21.0
    - predicted drain: 1600 mAh/day
    Calculated values:
    - usable: 2600 mAh
    - hourly rate: 100 mAh/h
    - hours_until_threshold: 26.0 h
    - hours_needed: 12.0 h
    - equivalent budget: 3466.67 mAh
    """
    stub_artifacts = {
        "risk": {
            "battery": {
                "task": "battery",
                "variant": "absolute",
                "residuals": np.array([0.0, 50.0, -50.0]),
                "n_residuals": 3,
                "floor_value": 0.0,
                "rho": 0.0,
                "seed": 42,
            }
        }
    }

    res = survival(
        artifacts=stub_artifacts,
        forecast_result=1600.0,
        battery_capacity_mah=4000.0,
        current_battery_pct=80.0,
        current_hour=9.0,
        need_until_hour=21.0,
        min_battery_pct=15.0,
    )

    assert res["status"] == "ok"
    assert res["usable_mah"] == 2600.0
    assert res["hourly_rate"] == 100.0
    assert res["hours_until_threshold"] == 26.0
    assert res["hours_needed"] == 12.0
    assert np.isclose(res["equivalent_daily_budget_mah"], 3466.67, atol=0.01)
    assert res["expected_to_make_it"] is True
    assert len(res["assumptions"]) >= 1


def test_survival_boundaries(in_memory_artifacts):
    """
    Verify boundary condition handling and input range validation for survival estimator.
    """
    stub_artifacts = in_memory_artifacts

    # 1. Current pct equal to min pct gives status already_at_or_below_threshold
    res_at_limit = survival(
        artifacts=stub_artifacts,
        forecast_result=1500.0,
        battery_capacity_mah=4000.0,
        current_battery_pct=15.0,
        current_hour=10.0,
        need_until_hour=18.0,
        min_battery_pct=15.0,
    )
    assert res_at_limit["status"] == "already_at_or_below_threshold"
    assert res_at_limit["hours_until_threshold"] == 0.0
    assert res_at_limit["p_make_it"] == 0.0
    assert res_at_limit["expected_to_make_it"] is False

    # 2. need_until_hour equal to current_hour raises ValueError
    with pytest.raises(ValueError, match="need_until_hour"):
        survival(stub_artifacts, 1500.0, 4000.0, 50.0, 10.0, 10.0)

    # 3. Overnight schedule: need_until_hour 25.5 with current_hour 22.0 accepted (3.5 hours)
    res_overnight = survival(
        artifacts=stub_artifacts,
        forecast_result=1600.0,
        battery_capacity_mah=4000.0,
        current_battery_pct=50.0,
        current_hour=22.0,
        need_until_hour=25.5,
        min_battery_pct=15.0,
    )
    assert res_overnight["hours_needed"] == 3.5
    assert res_overnight["status"] == "ok"

    # 4. Out-of-bounds parameter exceptions
    invalid_cases = [
        ("capacity 0", {"battery_capacity_mah": 0, "current_battery_pct": 50, "current_hour": 10, "need_until_hour": 15}),
        ("negative capacity", {"battery_capacity_mah": -500, "current_battery_pct": 50, "current_hour": 10, "need_until_hour": 15}),
        ("capacity 20000", {"battery_capacity_mah": 20000, "current_battery_pct": 50, "current_hour": 10, "need_until_hour": 15}),
        ("pct 0", {"battery_capacity_mah": 4000, "current_battery_pct": 0, "current_hour": 10, "need_until_hour": 15}),
        ("pct 101", {"battery_capacity_mah": 4000, "current_battery_pct": 101, "current_hour": 10, "need_until_hour": 15}),
        ("current_hour 24", {"battery_capacity_mah": 4000, "current_battery_pct": 50, "current_hour": 24, "need_until_hour": 26}),
    ]
    for label, kw in invalid_cases:
        with pytest.raises(ValueError):
            survival(stub_artifacts, 1500.0, **kw)


def test_survival_p_make_it_monotonicity(in_memory_artifacts):
    """
    Verify survival probability decreases with longer durations needed and increases with battery state.
    """
    # 1. As hours_needed increases, p_make_it decreases
    res_short = survival(in_memory_artifacts, 1500.0, 4000.0, 50.0, 10.0, 13.0)  # 3 hours
    res_long = survival(in_memory_artifacts, 1500.0, 4000.0, 50.0, 10.0, 22.0)   # 12 hours
    assert res_short["p_make_it"] >= res_long["p_make_it"]

    # 2. As current battery percentage increases, p_make_it increases
    res_low_pct = survival(in_memory_artifacts, 1500.0, 4000.0, 25.0, 10.0, 18.0)
    res_high_pct = survival(in_memory_artifacts, 1500.0, 4000.0, 75.0, 10.0, 18.0)
    assert res_high_pct["p_make_it"] >= res_low_pct["p_make_it"]


def test_profile_and_profile_history(in_memory_artifacts):
    """
    Verify profile and profile_history classification and validation.
    """
    clustering = in_memory_artifacts["clustering"]
    valid_labels = set(clustering["label_map"].values())

    # 1. Valid profile call
    valid_obs = {
        "app_usage_min": 200,
        "screen_on_hours": 4.5,
        "battery_mah": 1600,
        "data_mb": 800,
    }
    p_res = profile(in_memory_artifacts, valid_obs)
    assert p_res["profile_label"] in valid_labels
    assert "centroid_table" in p_res
    assert "percentile_report" in p_res

    # 2. Validation errors on invalid observed metrics
    with pytest.raises(ValueError, match="negative"):
        profile(in_memory_artifacts, {**valid_obs, "battery_mah": -100})
    with pytest.raises(ValueError):
        profile(in_memory_artifacts, {**valid_obs, "app_usage_min": np.nan})
    with pytest.raises(ValueError, match="Missing"):
        profile(in_memory_artifacts, {"app_usage_min": 200, "screen_on_hours": 4.5})

    # 3. Profile history across 3 records
    hist_df = pd.DataFrame([
        {"app_usage_min": 100, "screen_on_hours": 2.0, "battery_mah": 800, "data_mb": 300},
        {"app_usage_min": 250, "screen_on_hours": 5.0, "battery_mah": 1700, "data_mb": 900},
        {"app_usage_min": 450, "screen_on_hours": 8.0, "battery_mah": 2600, "data_mb": 1800},
    ])
    h_res = profile_history(in_memory_artifacts, hist_df)
    assert len(h_res["row_labels"]) == 3
    for lbl in h_res["row_labels"]:
        assert lbl in valid_labels
    assert h_res["average_profile"]["profile_label"] in valid_labels
    assert np.isclose(sum(h_res["label_shares"].values()), 1.0)


def test_validation_errors(in_memory_artifacts):
    """
    Verify invalid budgets and unknown device model raise informative ValueErrors.
    """
    valid_inputs = {
        "device_model": "Google Pixel 5",
        "app_usage_min": 200,
        "screen_on_hours": 4.0,
        "apps_installed": 40,
    }
    valid_budgets = {"battery_mah": 2000.0, "data_mb": 1000.0, "screen_hours": 5.0}

    # Unknown device model
    with pytest.raises(ValueError, match="Unknown device_model"):
        forecast(in_memory_artifacts, {**valid_inputs, "device_model": "Nokia 3310"}, valid_budgets)

    with pytest.raises(ValueError, match="Unknown device_model"):
        plan(in_memory_artifacts, {**valid_inputs, "device_model": "Nokia 3310"}, valid_budgets)

    # Invalid budgets
    for b_mod in [
        {"battery_mah": 0.0},
        {"battery_mah": -50.0},
        {"data_mb": np.nan},
        {"screen_hours": -1.0},
    ]:
        bad_b = {**valid_budgets, **b_mod}
        with pytest.raises(ValueError):
            forecast(in_memory_artifacts, valid_inputs, bad_b)
        with pytest.raises(ValueError):
            plan(in_memory_artifacts, valid_inputs, bad_b)


def test_inputs_not_mutated_and_deterministic(in_memory_artifacts):
    """
    Verify input dictionaries are not mutated and inference is strictly deterministic.
    """
    inputs = {
        "device_model": "Xiaomi Mi 11",
        "app_usage_min": 250,
        "screen_on_hours": 5.0,
        "apps_installed": 50,
    }
    inputs_copy = dict(inputs)
    budgets = {"battery_mah": 1800.0, "data_mb": 900.0, "screen_hours": 5.5}
    budgets_copy = dict(budgets)

    res1 = forecast(in_memory_artifacts, inputs, budgets)
    res2 = forecast(in_memory_artifacts, inputs, budgets)

    # Inputs unchanged
    assert inputs == inputs_copy
    assert budgets == budgets_copy

    # Outputs strictly identical
    assert res1["predictions"] == res2["predictions"]
    assert res1["risk"]["battery"]["p_overrun"] == res2["risk"]["battery"]["p_overrun"]


# =====================================================================
# PART 2: SAVED ARTIFACTS TESTS (MODELS/ DIRECTORY)
# =====================================================================


def _saved_artifacts_exist() -> bool:
    """Check whether production serialized artifacts exist in models/."""
    for path in config.MODEL_PATHS.values():
        if not path.exists():
            return False
    if not config.CLUSTERING_MODEL_PATH.exists():
        return False
    if not config.RISK_MODEL_PATH.exists():
        return False
    return True


@pytest.mark.skipif(not _saved_artifacts_exist(), reason="Production models missing from models/ directory")
def test_saved_artifacts_load_and_forecast_equality():
    """
    Verify production artifacts load properly and forecast matches direct pipeline calls.
    """
    artifacts = load_artifacts()
    assert "pipelines" in artifacts
    assert "clustering" in artifacts
    assert "risk" in artifacts

    inputs = {
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 250,
        "screen_on_hours": 5.0,
        "apps_installed": 50,
    }
    budgets = {"battery_mah": 1800.0, "data_mb": 1000.0, "screen_hours": 6.0}

    res = forecast(artifacts, inputs, budgets, screen_known=True)

    # Compare against direct pipeline calls
    raw_df_bat = pd.DataFrame([{
        "device_model": inputs["device_model"],
        "app_usage_min": inputs["app_usage_min"],
        "screen_on_hours": inputs["screen_on_hours"],
        "apps_installed": inputs["apps_installed"],
    }])
    raw_df_screen = pd.DataFrame([{
        "device_model": inputs["device_model"],
        "app_usage_min": inputs["app_usage_min"],
        "apps_installed": inputs["apps_installed"],
    }])

    pipe_bat = artifacts["pipelines"]["battery"]
    pipe_data = artifacts["pipelines"]["data"]
    pipe_screen = artifacts["pipelines"]["screen_time"]

    direct_bat = float(pipe_bat.predict(raw_df_bat)[0])
    direct_data = float(pipe_data.predict(raw_df_bat)[0])
    direct_screen = float(pipe_screen.predict(raw_df_screen)[0])

    assert np.isclose(res["predictions"]["battery_mah"], direct_bat)
    assert np.isclose(res["predictions"]["data_mb"], direct_data)
    # When screen_known is True, screen_hours predicted equals user input, and screen_model_estimate equals direct_screen
    assert np.isclose(res["predictions"]["screen_hours"], inputs["screen_on_hours"])
    assert np.isclose(res["screen_model_estimate"], direct_screen)
    assert "budgets_used" in res

    # Determinism across repeated calls
    res_repeat = forecast(artifacts, inputs, budgets, screen_known=True)
    for k in res["predictions"]:
        assert np.isclose(res["predictions"][k], res_repeat["predictions"][k])


@pytest.mark.skipif(not _saved_artifacts_exist(), reason="Production models missing from models/ directory")
def test_saved_artifacts_will_i_make_it():
    """
    Verify will_i_make_it returns distinct plain-English verdicts for achievable and unachievable goals.
    """
    artifacts = load_artifacts()

    # Achievable scenario: 95% battery, 4000 mAh capacity, only 2 hours needed
    inputs_easy = {
        "device_model": "Google Pixel 5",
        "app_usage_min": 100,
        "screen_on_hours": 2.0,
        "apps_installed": 25,
    }
    survival_easy = {
        "battery_capacity_mah": 4000,
        "current_battery_pct": 95,
        "current_hour": 14.0,
        "need_until_hour": 16.0,
        "min_battery_pct": 15,
    }
    res_easy = will_i_make_it(artifacts, inputs_easy, survival_easy)
    assert res_easy["survival"]["expected_to_make_it"] is True
    assert "will make it" in res_easy["verdict"].lower()
    assert "budgets_used" in res_easy

    # Highly unachievable scenario: heavy phone, 20% battery, 13 hours needed
    inputs_hard = {
        "device_model": "iPhone 12",
        "app_usage_min": 450,
        "screen_on_hours": 8.0,
        "apps_installed": 80,
    }
    survival_hard = {
        "battery_capacity_mah": 3000,
        "current_battery_pct": 20,
        "current_hour": 10.0,
        "need_until_hour": 23.0,
        "min_battery_pct": 15,
    }
    res_hard = will_i_make_it(artifacts, inputs_hard, survival_hard)
    assert res_hard["survival"]["expected_to_make_it"] is False
    assert "unlikely to make it" in res_hard["verdict"].lower()
    assert "budgets_used" in res_hard
    assert len(res_hard["verdict"]) > 10
