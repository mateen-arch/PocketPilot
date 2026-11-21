"""
What-if optimizer searching for minimal-disruption behavior modification plans.
"""

from typing import Any
import numpy as np
import pandas as pd

from pocketpilot.config import PLANNER_TOP_N
from pocketpilot.inference.validation import validate_inputs
from pocketpilot.planner.action_cost import action_cost
from pocketpilot.planner.scenarios import build_scenario_grid, evaluate_scenarios
from pocketpilot.risk.probabilistic_risk import assess_budget


def recommend(
    base: dict[str, Any],
    budgets: dict[str, float],
    pipelines: dict[str, Any],
    risk_bundles: dict[str, Any],
    bounds: dict[str, Any],
    safety_margin: float = 0.0,
    top_n: int = PLANNER_TOP_N,
    allow_assumptions: bool = True,
) -> dict[str, Any]:
    """
    Search and recommend minimal action-cost behavior change plans meeting user budgets.

    Validation & Feasibility Rules:
    1. Input validation:
       - base must contain 'device_model', 'app_usage_min', 'screen_on_hours', 'apps_installed'.
       - budgets must contain 'battery_mah', 'data_mb', 'screen_hours' (strictly positive, finite).
       - safety_margin must be in range [0.0, 0.5].
    2. Feasibility definition:
       - within_bounds is True (no extrapolation below training minimum).
       - predicted battery <= budget * (1 - safety_margin).
       - predicted data <= budget * (1 - safety_margin).
       - predicted screen time <= budget * (1 - safety_margin).
    3. Status resolution:
       - 'already_within_budget': The baseline (no-op) scenario is feasible (cost = 0).
       - 'plan_found': At least one behavior modification plan meets all budget constraints.
       - 'not_achievable': No within-bounds scenario satisfies the budgets simultaneously;
         a 'best_effort' plan minimizing the maximum resource overrun ratio is returned.
    4. Deterministic ranking:
       Feasible scenarios are ranked by (action_cost, n_levers, app_cut_min, screen_cut_hours,
       wifi_share, power_saving) ascending.

    Parameters
    ----------
    base : dict[str, Any]
        User baseline features.
    budgets : dict[str, float]
        Target budgets for 'battery_mah', 'data_mb', and 'screen_hours'.
    pipelines : dict[str, Any]
        Fitted regression pipelines: 'battery', 'data', 'screen_time'.
    risk_bundles : dict[str, Any]
        Fitted risk calibration bundles for each resource task.
    bounds : dict[str, Any]
        Training domain bounds from training_bounds().
    safety_margin : float, default=0.0
        Proportional safety buffer in [0.0, 0.5].
    top_n : int, default=PLANNER_TOP_N
        Number of top plans to retain (1 best + top_n-1 alternatives).

    Returns
    -------
    dict[str, Any]
        Planner outcome dictionary with status, baseline, best plan, alternatives,
        assumptions used, learned effects notes, and scenario counts.

    Raises
    ------
    ValueError
        If inputs or budgets are missing, non-positive, NaN, or out of valid bounds.
    """
    # 1. Validation: Safety margin
    if not isinstance(safety_margin, (int, float)) or not np.isfinite(safety_margin):
        raise ValueError(f"safety_margin must be a finite number, got {safety_margin}")
    if safety_margin < 0.0 or safety_margin > 0.5:
        raise ValueError(f"safety_margin must be in [0.0, 0.5], got {safety_margin}")

    # 2. Validation: Base user keys
    required_base_keys = ["device_model", "app_usage_min", "screen_on_hours", "apps_installed"]
    for k in required_base_keys:
        if k not in base:
            raise ValueError(f"Missing required key '{k}' in base dictionary.")

    # 3. Validation: Budget keys
    required_budget_keys = ["battery_mah", "data_mb", "screen_hours"]
    for k in required_budget_keys:
        if k not in budgets:
            raise ValueError(f"Missing required budget key '{k}' in budgets dictionary.")
        val_b = budgets[k]
        if val_b is None or not isinstance(val_b, (int, float)) or not np.isfinite(val_b) or val_b <= 0:
            raise ValueError(f"Budget '{k}' must be finite and strictly positive (> 0), got: {val_b}")

    # Re-validate with inference validation utility
    val_errors = validate_inputs(
        device_model=base.get("device_model"),
        app_usage_min=base.get("app_usage_min"),
        screen_on_hours=base.get("screen_on_hours"),
        apps_installed=base.get("apps_installed"),
        battery_budget=budgets.get("battery_mah"),
        data_budget=budgets.get("data_mb"),
        screen_budget=budgets.get("screen_hours"),
    )
    if val_errors:
        raise ValueError("; ".join(val_errors))

    # 4. Generate scenario grid and evaluate in batch
    mode = "all_levers" if allow_assumptions else "learned_only"
    grid = build_scenario_grid()
    if not allow_assumptions:
        grid = grid[(grid["wifi_share"] == 0.0) & (~grid["power_saving"])].copy()

    scenarios = evaluate_scenarios(base, grid, pipelines, bounds)
    scenarios["action_cost"] = action_cost(scenarios)

    # 5. Evaluate feasibility against adjusted budgets
    margin_factor = 1.0 - float(safety_margin)
    bat_limit = float(budgets["battery_mah"]) * margin_factor
    data_limit = float(budgets["data_mb"]) * margin_factor
    screen_limit = float(budgets["screen_hours"]) * margin_factor

    scenarios["is_feasible"] = (
        scenarios["within_bounds"]
        & (scenarios["battery_pred"] <= bat_limit)
        & (scenarios["data_pred"] <= data_limit)
        & (scenarios["screen_pred"] <= screen_limit)
    )

    # 6. Locate no-op scenario
    noop_mask = (
        (scenarios["app_cut_min"] == 0)
        & (scenarios["screen_cut_hours"] == 0.0)
        & (scenarios["wifi_share"] == 0.0)
        & (~scenarios["power_saving"])
    )
    noop_row = scenarios[noop_mask].iloc[0]

    # Baseline risk assessment for the no-op state
    baseline_risk = {
        "battery": assess_budget(risk_bundles["battery"], float(noop_row["battery_pred"]), float(budgets["battery_mah"])),
        "data": assess_budget(risk_bundles["data"], float(noop_row["data_pred"]), float(budgets["data_mb"])),
        "screen_time": assess_budget(risk_bundles["screen_time"], float(noop_row["screen_pred"]), float(budgets["screen_hours"])),
    }

    baseline_dict = {
        "app_cut_min": 0,
        "screen_cut_hours": 0.0,
        "wifi_share": 0.0,
        "power_saving": False,
        "battery_pred": float(noop_row["battery_pred"]),
        "data_pred": float(noop_row["data_pred"]),
        "screen_pred": float(noop_row["screen_pred"]),
        "action_cost": 0.0,
        "risk": baseline_risk,
    }

    # Deterministic sorting keys
    sort_cols = ["action_cost", "n_levers", "app_cut_min", "screen_cut_hours", "wifi_share", "power_saving"]

    # 7. Status & Plan resolution
    n_feasible = int(scenarios["is_feasible"].sum())
    is_noop_feasible = bool(noop_row["is_feasible"])

    if is_noop_feasible:
        status = "already_within_budget"
        best_row = noop_row
        feasible_sorted = scenarios[scenarios["is_feasible"]].sort_values(by=sort_cols, ascending=True)
        # Alternatives exclude no-op
        alternatives_df = feasible_sorted[feasible_sorted.index != noop_row.name].head(top_n - 1)
    elif n_feasible > 0:
        status = "plan_found"
        feasible_sorted = scenarios[scenarios["is_feasible"]].sort_values(by=sort_cols, ascending=True)
        best_row = feasible_sorted.iloc[0]
        alternatives_df = feasible_sorted.iloc[1:top_n]
    else:
        status = "not_achievable"
        # Best effort: within-bounds scenario minimizing max ratio of prediction to budget
        within_df = scenarios[scenarios["within_bounds"]].copy()
        ratio_bat = within_df["battery_pred"] / float(budgets["battery_mah"])
        ratio_data = within_df["data_pred"] / float(budgets["data_mb"])
        ratio_screen = within_df["screen_pred"] / float(budgets["screen_hours"])
        within_df["max_budget_ratio"] = np.maximum(np.maximum(ratio_bat, ratio_data), ratio_screen)

        best_effort_sorted = within_df.sort_values(
            by=["max_budget_ratio"] + sort_cols,
            ascending=[True] + [True] * len(sort_cols),
        )
        best_row = best_effort_sorted.iloc[0]
        alternatives_df = pd.DataFrame()

    # 8. Risk assessment for best (or best_effort) plan
    # Note: For assumption levers the prediction interval reuses the model's relative error
    # and does not include epistemic uncertainty regarding the assumed multiplier.
    best_risk = {
        "battery": assess_budget(risk_bundles["battery"], float(best_row["battery_pred"]), float(budgets["battery_mah"])),
        "data": assess_budget(risk_bundles["data"], float(best_row["data_pred"]), float(budgets["data_mb"])),
        "screen_time": assess_budget(risk_bundles["screen_time"], float(best_row["screen_pred"]), float(budgets["screen_hours"])),
    }

    best_dict = {
        "app_cut_min": int(best_row["app_cut_min"]),
        "screen_cut_hours": float(best_row["screen_cut_hours"]),
        "wifi_share": float(best_row["wifi_share"]),
        "power_saving": bool(best_row["power_saving"]),
        "battery_pred": float(best_row["battery_pred"]),
        "data_pred": float(best_row["data_pred"]),
        "screen_pred": float(best_row["screen_pred"]),
        "action_cost": float(best_row["action_cost"]),
        "risk": best_risk,
    }

    # 9. Human-readable explanation of assumption levers used
    assumptions_used: list[str] = []
    if allow_assumptions:
        if best_dict["power_saving"]:
            assumptions_used.append("Power-saving mode reduces battery by 15 percent (assumption, not learned from data)")
        if best_dict["wifi_share"] > 0.0:
            pct = int(round(best_dict["wifi_share"] * 100))
            assumptions_used.append(f"Shifting {pct} percent of mobile data to Wi-Fi is an assumption")

    learned_effects_note = (
        "Cutting app time also lowers screen time through the screen-time model."
    )

    return {
        "mode": mode,
        "status": status,
        "baseline": baseline_dict,
        "best": best_dict,
        "alternatives": alternatives_df,
        "assumptions_used": assumptions_used,
        "learned_effects_note": learned_effects_note,
        "n_feasible": n_feasible,
        "n_evaluated": len(scenarios),
    }
