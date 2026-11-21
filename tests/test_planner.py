"""
Comprehensive unit and integration tests for what-if behavior planner and action-cost optimization.
"""

import copy
import numpy as np
import pandas as pd
import pytest

from pocketpilot.config import (
    PLANNER_APP_CUTS,
    PLANNER_SCREEN_CUTS,
    PLANNER_WIFI_SHARES,
    POWER_SAVING_BATTERY_MULTIPLIER,
    W1_MINUTES,
    W2_POWER_SAVING,
    W3_WIFI_PER_PCT_POINT,
)
from pocketpilot.data.cleaning import clean_ds1
from pocketpilot.data.loader import load_ds1
from pocketpilot.data.splitting import split_ds1
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP, train_task
from pocketpilot.planner.action_cost import action_cost
from pocketpilot.planner.optimizer import recommend
from pocketpilot.planner.scenarios import (
    build_scenario_grid,
    evaluate_scenarios,
    training_bounds,
)
from pocketpilot.risk.probabilistic_risk import fit_risk_model


@pytest.fixture(scope="module")
def planner_setup():
    """Train fast models and fit risk bundles entirely in-memory for testing."""
    raw = load_ds1()
    clean = clean_ds1(raw)
    train_df, test_df = split_ds1(clean)

    tasks = ["battery", "data", "screen_time"]
    pipelines = {}
    risk_bundles = {}

    for t in tasks:
        res = train_task(t, train_df, test_df, fast=True)
        pipe = res["selected_model"]
        pipelines[t] = pipe
        risk_bundles[t] = fit_risk_model(t, pipe, train_df)

    bounds = training_bounds(train_df)

    return {
        "pipelines": pipelines,
        "risk_bundles": risk_bundles,
        "bounds": bounds,
        "train_df": train_df,
    }


def test_grid_properties():
    """Verify grid shape, no-op uniqueness, and zero initial cost."""
    grid = build_scenario_grid()
    assert len(grid) == 224, f"Expected 224 rows, got {len(grid)}"

    # Exactly one all-levers-off row
    noop_mask = (
        (grid["app_cut_min"] == 0)
        & (grid["screen_cut_hours"] == 0.0)
        & (grid["wifi_share"] == 0.0)
        & (~grid["power_saving"])
    )
    assert noop_mask.sum() == 1

    noop_row = grid[noop_mask]
    cost = action_cost(noop_row).iloc[0]
    assert cost == 0.0, f"Expected cost 0 for no-op, got {cost}"


def test_cost_formula_and_monotonicity():
    """Verify specific cost value and non-decreasing cost property."""
    # for app_cut 45, screen_cut 1.0, wifi 0.4, power saving True
    df_sample = pd.DataFrame([{
        "app_cut_min": 45,
        "screen_cut_hours": 1.0,
        "wifi_share": 0.4,
        "power_saving": True,
    }])
    cost_val = action_cost(df_sample).iloc[0]
    expected_cost = 1.0 * (45 + 60 * 1.0) + 20.0 * 1 + 0.5 * (100 * 0.4)
    assert cost_val == 145.0, f"Expected cost 145.0, got {cost_val}"
    assert cost_val == expected_cost

    # Monotonicity checks: Cost never decreases when any single lever increases
    grid = build_scenario_grid()
    grid["cost"] = action_cost(grid)

    # 1. Increasing app cut
    sub1 = grid[(grid["screen_cut_hours"] == 0.0) & (grid["wifi_share"] == 0.0) & (~grid["power_saving"])]
    diffs_app = np.diff(sub1.sort_values("app_cut_min")["cost"].values)
    assert np.all(diffs_app >= 0)

    # 2. Increasing screen cut
    sub2 = grid[(grid["app_cut_min"] == 0) & (grid["wifi_share"] == 0.0) & (~grid["power_saving"])]
    diffs_screen = np.diff(sub2.sort_values("screen_cut_hours")["cost"].values)
    assert np.all(diffs_screen >= 0)


def test_noop_reproduces_direct_predictions(planner_setup):
    """Verify no-op scenario produces identical predictions to direct model calls."""
    pipelines = planner_setup["pipelines"]
    bounds = planner_setup["bounds"]

    base = {
        "device_model": "Google Pixel 5",
        "app_usage_min": 250,
        "screen_on_hours": 5.0,
        "apps_installed": 30,
    }

    grid = build_scenario_grid()
    eval_df = evaluate_scenarios(base, grid, pipelines, bounds)

    noop_row = eval_df[
        (eval_df["app_cut_min"] == 0)
        & (eval_df["screen_cut_hours"] == 0.0)
        & (eval_df["wifi_share"] == 0.0)
        & (~eval_df["power_saving"])
    ].iloc[0]

    # Direct predictions
    direct_bat_in = pd.DataFrame([base])[TASK_RAW_INPUTS_MAP["battery"]]
    direct_bat = float(pipelines["battery"].predict(direct_bat_in)[0])

    direct_data_in = pd.DataFrame([base])[TASK_RAW_INPUTS_MAP["data"]]
    direct_data = float(pipelines["data"].predict(direct_data_in)[0])

    assert np.isclose(noop_row["battery_pred"], direct_bat)
    assert np.isclose(noop_row["data_pred"], direct_data)
    assert np.isclose(noop_row["new_screen"], base["screen_on_hours"])


def test_extrapolation_bounds(planner_setup):
    """Verify out-of-bounds scenarios are tagged within_bounds=False and never recommended."""
    pipelines = planner_setup["pipelines"]
    risk_bundles = planner_setup["risk_bundles"]
    bounds = planner_setup["bounds"]

    # Low-usage base near bounds
    base = {
        "device_model": "Google Pixel 5",
        "app_usage_min": bounds["app_min"] + 10,  # 10 min above min
        "screen_on_hours": bounds["screen_min"] + 0.2,
        "apps_installed": 20,
    }

    grid = build_scenario_grid()
    eval_df = evaluate_scenarios(base, grid, pipelines, bounds)

    # Scenarios cutting 120 min will drop below min
    below_mask = eval_df["new_app"] < bounds["app_min"]
    assert below_mask.any()
    assert not eval_df.loc[below_mask, "within_bounds"].any()

    # In optimizer, out of bounds rows must never be recommended
    budgets = {"battery_mah": 3000.0, "data_mb": 2000.0, "screen_hours": 8.0}
    res = recommend(base, budgets, pipelines, risk_bundles, bounds)
    # The best plan must be within bounds
    best = res["best"]
    best_new_app = base["app_usage_min"] - best["app_cut_min"]
    assert best_new_app >= bounds["app_min"] or (best["app_cut_min"] == 0 and best["screen_cut_hours"] == 0.0)


def test_assumption_levers_isolation(planner_setup):
    """Verify power saving modifies battery only, Wi-Fi modifies data only, and flags match."""
    pipelines = planner_setup["pipelines"]
    bounds = planner_setup["bounds"]

    base = {
        "device_model": "OnePlus 9",
        "app_usage_min": 300,
        "screen_on_hours": 6.0,
        "apps_installed": 40,
    }

    grid = build_scenario_grid()
    eval_df = evaluate_scenarios(base, grid, pipelines, bounds)

    # Flag correctness
    expected_flags = (eval_df["wifi_share"] > 0.0) | (eval_df["power_saving"] == True)
    assert (eval_df["is_assumption_based"] == expected_flags).all()

    # Compare power saving vs non-power saving holding other levers at 0
    row_plain = eval_df[(eval_df["n_levers"] == 0)].iloc[0]
    row_ps = eval_df[
        (eval_df["app_cut_min"] == 0)
        & (eval_df["screen_cut_hours"] == 0.0)
        & (eval_df["wifi_share"] == 0.0)
        & (eval_df["power_saving"])
    ].iloc[0]

    # Power saving changes battery by exact multiplier
    assert np.isclose(row_ps["battery_pred"], row_plain["battery_pred"] * POWER_SAVING_BATTERY_MULTIPLIER)
    # Power saving does NOT modify data
    assert np.isclose(row_ps["data_pred"], row_plain["data_pred"])

    # Wi-Fi shift comparison
    row_wifi = eval_df[
        (eval_df["app_cut_min"] == 0)
        & (eval_df["screen_cut_hours"] == 0.0)
        & (eval_df["wifi_share"] == 0.4)
        & (~eval_df["power_saving"])
    ].iloc[0]

    # Wi-Fi changes data
    assert np.isclose(row_wifi["data_pred"], row_plain["data_pred"] * (1.0 - 0.4))
    # Wi-Fi does NOT modify battery
    assert np.isclose(row_wifi["battery_pred"], row_plain["battery_pred"])


def test_screen_follow_through(planner_setup):
    """Verify follow-through battery reduction is larger than fixed screen reduction."""
    pipelines = planner_setup["pipelines"]

    base_heavy = {
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 480,
        "screen_on_hours": 9.0,
        "apps_installed": 85,
    }

    base_in = pd.DataFrame([base_heavy])[TASK_RAW_INPUTS_MAP["battery"]]
    base_bat = float(pipelines["battery"].predict(base_in)[0])

    # Fixed screen reduction with app_cut = 45
    fixed_in = pd.DataFrame([{
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 480 - 45,
        "screen_on_hours": 9.0,
        "apps_installed": 85,
    }])[TASK_RAW_INPUTS_MAP["battery"]]
    fixed_bat = float(pipelines["battery"].predict(fixed_in)[0])
    reduction_fixed = base_bat - fixed_bat

    # Follow-through screen reduction
    screen_base_in = pd.DataFrame([base_heavy])[TASK_RAW_INPUTS_MAP["screen_time"]]
    screen_new_in = pd.DataFrame([{
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 480 - 45,
        "apps_installed": 85,
    }])[TASK_RAW_INPUTS_MAP["screen_time"]]
    s_base = float(pipelines["screen_time"].predict(screen_base_in)[0])
    s_new = float(pipelines["screen_time"].predict(screen_new_in)[0])
    delta_s = s_new - s_base
    new_s = 9.0 + delta_s

    follow_in = pd.DataFrame([{
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 480 - 45,
        "screen_on_hours": new_s,
        "apps_installed": 85,
    }])[TASK_RAW_INPUTS_MAP["battery"]]
    follow_bat = float(pipelines["battery"].predict(follow_in)[0])
    reduction_follow = base_bat - follow_bat

    assert reduction_follow > reduction_fixed, (
        f"Expected follow-through reduction ({reduction_follow:.2f}) > "
        f"fixed reduction ({reduction_fixed:.2f})"
    )


def test_already_within_budget(planner_setup):
    """Verify generous budgets return status 'already_within_budget' and zero cost."""
    pipelines = planner_setup["pipelines"]
    risk_bundles = planner_setup["risk_bundles"]
    bounds = planner_setup["bounds"]

    base = {
        "device_model": "Xiaomi Mi 11",
        "app_usage_min": 200,
        "screen_on_hours": 4.0,
        "apps_installed": 25,
    }
    budgets = {"battery_mah": 5000.0, "data_mb": 5000.0, "screen_hours": 15.0}

    res = recommend(base, budgets, pipelines, risk_bundles, bounds)
    assert res["status"] == "already_within_budget"
    assert res["best"]["action_cost"] == 0.0
    assert res["best"]["app_cut_min"] == 0
    assert res["best"]["screen_cut_hours"] == 0.0
    assert res["best"]["wifi_share"] == 0.0
    assert res["best"]["power_saving"] is False


def test_plan_found_optimality(planner_setup):
    """Verify plan_found identifies a valid, minimal-cost plan."""
    pipelines = planner_setup["pipelines"]
    risk_bundles = planner_setup["risk_bundles"]
    bounds = planner_setup["bounds"]

    base = {
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 350,
        "screen_on_hours": 7.0,
        "apps_installed": 50,
    }

    # Evaluate baseline battery
    base_bat_in = pd.DataFrame([base])[TASK_RAW_INPUTS_MAP["battery"]]
    base_bat = float(pipelines["battery"].predict(base_bat_in)[0])

    # Tighten battery budget to 90% of baseline
    budgets = {
        "battery_mah": base_bat * 0.90,
        "data_mb": 5000.0,
        "screen_hours": 15.0,
    }

    res = recommend(base, budgets, pipelines, risk_bundles, bounds)
    assert res["status"] == "plan_found"
    best = res["best"]
    assert best["battery_pred"] <= budgets["battery_mah"]
    assert best["action_cost"] > 0

    # Brute-force verification over full evaluated grid
    grid = build_scenario_grid()
    eval_df = evaluate_scenarios(base, grid, pipelines, bounds)
    eval_df["cost"] = action_cost(eval_df)
    feasible_mask = (
        eval_df["within_bounds"]
        & (eval_df["battery_pred"] <= budgets["battery_mah"])
        & (eval_df["data_pred"] <= budgets["data_mb"])
        & (eval_df["screen_pred"] <= budgets["screen_hours"])
    )
    cheapest_feasible_cost = eval_df.loc[feasible_mask, "cost"].min()
    assert np.isclose(best["action_cost"], cheapest_feasible_cost)


def test_not_achievable(planner_setup):
    """Verify tiny impossible budgets return status 'not_achievable' with best effort."""
    pipelines = planner_setup["pipelines"]
    risk_bundles = planner_setup["risk_bundles"]
    bounds = planner_setup["bounds"]

    base = {
        "device_model": "Google Pixel 5",
        "app_usage_min": 300,
        "screen_on_hours": 6.0,
        "apps_installed": 35,
    }
    # Impossible budgets
    budgets = {"battery_mah": 50.0, "data_mb": 1.0, "screen_hours": 0.2}

    res = recommend(base, budgets, pipelines, risk_bundles, bounds)
    assert res["status"] == "not_achievable"
    assert res["n_feasible"] == 0
    assert "best" in res
    assert res["best"]["battery_pred"] > budgets["battery_mah"]


def test_safety_margin(planner_setup):
    """Verify safety margin enforces proportional buffer below budget."""
    pipelines = planner_setup["pipelines"]
    risk_bundles = planner_setup["risk_bundles"]
    bounds = planner_setup["bounds"]

    base = {
        "device_model": "OnePlus 9",
        "app_usage_min": 300,
        "screen_on_hours": 6.0,
        "apps_installed": 40,
    }
    budgets = {"battery_mah": 2200.0, "data_mb": 1000.0, "screen_hours": 5.0}

    res = recommend(base, budgets, pipelines, risk_bundles, bounds, safety_margin=0.2)
    if res["status"] in ("plan_found", "already_within_budget"):
        best = res["best"]
        assert best["battery_pred"] <= budgets["battery_mah"] * 0.8 + 1e-5
        assert best["data_pred"] <= budgets["data_mb"] * 0.8 + 1e-5
        assert best["screen_pred"] <= budgets["screen_hours"] * 0.8 + 1e-5


def test_invalid_inputs_raise_error(planner_setup):
    """Verify invalid parameters (NaN, negative, zero budget, bad safety margin) raise ValueError."""
    pipelines = planner_setup["pipelines"]
    risk_bundles = planner_setup["risk_bundles"]
    bounds = planner_setup["bounds"]

    valid_base = {
        "device_model": "Google Pixel 5",
        "app_usage_min": 200,
        "screen_on_hours": 4.0,
        "apps_installed": 30,
    }
    valid_budgets = {"battery_mah": 2000.0, "data_mb": 1000.0, "screen_hours": 5.0}

    # 1. Zero budget
    with pytest.raises(ValueError):
        recommend(valid_base, {**valid_budgets, "battery_mah": 0.0}, pipelines, risk_bundles, bounds)

    # 2. Negative budget
    with pytest.raises(ValueError):
        recommend(valid_base, {**valid_budgets, "data_mb": -10.0}, pipelines, risk_bundles, bounds)

    # 3. NaN budget
    with pytest.raises(ValueError):
        recommend(valid_base, {**valid_budgets, "screen_hours": float("nan")}, pipelines, risk_bundles, bounds)

    # 4. Safety margin > 0.5
    with pytest.raises(ValueError):
        recommend(valid_base, valid_budgets, pipelines, risk_bundles, bounds, safety_margin=0.6)

    # 5. Missing base key
    bad_base = copy.deepcopy(valid_base)
    del bad_base["device_model"]
    with pytest.raises(ValueError):
        recommend(bad_base, valid_budgets, pipelines, risk_bundles, bounds)


def test_determinism_and_immutability(planner_setup):
    """Verify recommendation calls are deterministic and do not mutate caller arguments."""
    pipelines = planner_setup["pipelines"]
    risk_bundles = planner_setup["risk_bundles"]
    bounds = planner_setup["bounds"]

    base = {
        "device_model": "iPhone 12",
        "app_usage_min": 320,
        "screen_on_hours": 6.5,
        "apps_installed": 45,
    }
    budgets = {"battery_mah": 2100.0, "data_mb": 900.0, "screen_hours": 5.5}

    base_copy = copy.deepcopy(base)
    budgets_copy = copy.deepcopy(budgets)

    res1 = recommend(base, budgets, pipelines, risk_bundles, bounds)
    res2 = recommend(base, budgets, pipelines, risk_bundles, bounds)

    # Immutability
    assert base == base_copy
    assert budgets == budgets_copy

    # Determinism
    assert res1["status"] == res2["status"]
    assert res1["best"]["action_cost"] == res2["best"]["action_cost"]
    assert res1["best"]["app_cut_min"] == res2["best"]["app_cut_min"]
    assert res1["best"]["screen_cut_hours"] == res2["best"]["screen_cut_hours"]
    assert res1["best"]["wifi_share"] == res2["best"]["wifi_share"]
    assert res1["best"]["power_saving"] == res2["best"]["power_saving"]
    assert np.isclose(res1["best"]["battery_pred"], res2["best"]["battery_pred"])
