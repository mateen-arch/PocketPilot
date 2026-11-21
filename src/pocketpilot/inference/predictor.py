"""
Unified inference engine and battery survival estimator for PocketPilot (Feature F8).

This is the single high-level API entry point consumed by Streamlit pages.
All outputs are driven by fitted machine learning models, calibrated risk bundles,
and empirical training distributions with no hardcoded fallback outputs.
"""

from pathlib import Path
from typing import Any
import json
import joblib
import numpy as np
import pandas as pd
import sklearn

from pocketpilot import config
from pocketpilot.config import (
    CAPACITY_RANGE_MAH,
    DEFAULT_MIN_BATTERY_PCT,
    KNOWN_DEVICES,
    UNCONSTRAINED_BUDGET,
    WAKING_HOURS,
)
from pocketpilot.features.input_modes import get_feature_columns
from pocketpilot.models.clustering import percentile_report, predict_profile
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP
from pocketpilot.planner.optimizer import recommend
from pocketpilot.planner.scenarios import training_bounds
from pocketpilot.risk.probabilistic_risk import assess_budget, overrun_probability, predict_interval


def load_artifacts(
    models_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Load all trained models, risk bundles, clustering models, training distributions,
    and metadata required for end-to-end inference.

    Parameters
    ----------
    models_dir : Path or str, optional
        Directory containing serialized joblib artifacts and metadata.json.
        Defaults to config.MODELS_DIR.
    data_dir : Path or str, optional
        Directory containing processed dataset splits.
        Defaults to config.PROCESSED_DIR.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'pipelines': dict of regression pipelines ('battery', 'data', 'screen_time')
        - 'clustering': clustering bundle dictionary
        - 'risk': dict of risk calibration bundles
        - 'training_bounds': training domain limits for planner feasibility
        - 'reference_df': processed training partition DataFrame
        - 'metadata': training and evaluation metadata
        - 'warnings': list of operational warnings (e.g. scikit-learn version mismatch)
    """
    m_dir = Path(models_dir) if models_dir is not None else config.MODELS_DIR
    d_dir = Path(data_dir) if data_dir is not None else config.PROCESSED_DIR

    warnings: list[str] = []

    # 1. Load regression pipelines
    pipelines = {
        "battery": joblib.load(m_dir / "battery_model.joblib"),
        "data": joblib.load(m_dir / "data_usage_model.joblib"),
        "screen_time": joblib.load(m_dir / "screen_time_model.joblib"),
    }

    # 2. Load clustering and risk models
    clustering = joblib.load(m_dir / "clustering_model.joblib")
    risk = joblib.load(m_dir / "risk_model.joblib")

    # 3. Load reference training distribution and calculate bounds
    train_path = d_dir / "ds1_train.csv"
    if train_path.exists():
        reference_df = pd.read_csv(train_path)
        bounds = training_bounds(reference_df)
    else:
        reference_df = pd.DataFrame()
        bounds = {}

    # 4. Load metadata and verify environment version
    metadata_path = m_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    for task_name in ("battery", "data", "screen_time"):
        if task_name in metadata and "sklearn_version" in metadata[task_name]:
            meta_ver = metadata[task_name]["sklearn_version"]
            if meta_ver != sklearn.__version__:
                warnings.append(
                    f"Model '{task_name}' was serialized with scikit-learn {meta_ver}, "
                    f"but runtime environment is using {sklearn.__version__}."
                )
                break

    return {
        "pipelines": pipelines,
        "clustering": clustering,
        "risk": risk,
        "training_bounds": bounds,
        "reference_df": reference_df,
        "metadata": metadata,
        "warnings": warnings,
    }


def _validate_forecast_inputs(
    inputs: dict[str, Any],
    budgets: dict[str, Any],
    screen_known: bool,
) -> None:
    """Validate raw inputs and user budget limits for forecast mode."""
    # 1. Validate device model
    if "device_model" not in inputs:
        raise ValueError("Missing required input: 'device_model'.")
    dev = inputs["device_model"]
    if dev not in KNOWN_DEVICES:
        raise ValueError(f"Unknown device_model '{dev}'. Supported devices: {KNOWN_DEVICES}.")

    # 2. Validate app usage minutes
    if "app_usage_min" not in inputs:
        raise ValueError("Missing required input: 'app_usage_min'.")
    try:
        app_min = float(inputs["app_usage_min"])
        if not np.isfinite(app_min) or app_min < 0:
            raise ValueError(f"app_usage_min must be positive and finite, got {inputs['app_usage_min']}.")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"app_usage_min must be a valid number, got {inputs['app_usage_min']}.") from exc

    # 3. Validate apps installed
    if "apps_installed" not in inputs:
        raise ValueError("Missing required input: 'apps_installed'.")
    try:
        apps = float(inputs["apps_installed"])
        if not np.isfinite(apps) or apps < 0:
            raise ValueError(f"apps_installed must be non-negative and finite, got {inputs['apps_installed']}.")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"apps_installed must be a valid number, got {inputs['apps_installed']}.") from exc

    # 4. Validate screen_on_hours if screen is known
    if screen_known:
        if "screen_on_hours" not in inputs:
            raise ValueError("screen_on_hours must be provided when screen_known is True.")
        try:
            screen_h = float(inputs["screen_on_hours"])
            if not np.isfinite(screen_h) or screen_h <= 0:
                raise ValueError(f"screen_on_hours must be strictly positive, got {inputs['screen_on_hours']}.")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"screen_on_hours must be a valid number, got {inputs['screen_on_hours']}.") from exc

    # 5. Validate budgets
    for b_key in ("battery_mah", "data_mb", "screen_hours"):
        if b_key not in budgets:
            # Check for alternative key
            alt_key = "screen_time" if b_key == "screen_hours" else None
            if alt_key and alt_key in budgets:
                continue
            raise ValueError(f"Missing required budget: '{b_key}'.")

        val = budgets.get(b_key)
        try:
            val_f = float(val)
            if not np.isfinite(val_f) or val_f <= 0:
                raise ValueError(f"Budget '{b_key}' must be strictly positive (> 0), got {val}.")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Budget '{b_key}' must be a valid positive number, got {val}.") from exc


def forecast(
    artifacts: dict[str, Any],
    inputs: dict[str, Any],
    budgets: dict[str, Any],
    screen_known: bool = True,
) -> dict[str, Any]:
    """
    Predict daily battery drain, mobile data consumption, and screen-on time,
    and assess budget overrun risk for each resource.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded system artifacts from load_artifacts().
    inputs : dict[str, Any]
        User input dictionary: 'device_model', 'app_usage_min', 'apps_installed',
        and optionally 'screen_on_hours'.
    budgets : dict[str, Any]
        User operational budgets: 'battery_mah', 'data_mb', 'screen_hours'.
    screen_known : bool, default=True
        If False, screen-on time is estimated via the screen_time model and fed
        into the downstream battery and data models.

    Returns
    -------
    dict[str, Any]
        Dictionary of predictions, overrun risks, prediction intervals, and screen source.
    """
    _validate_forecast_inputs(inputs, budgets, screen_known)
    pipelines = artifacts["pipelines"]
    risk = artifacts["risk"]

    device_model = inputs["device_model"]
    app_usage_min = float(inputs["app_usage_min"])
    apps_installed = float(inputs["apps_installed"])

    # 1. Screen time prediction
    screen_df = pd.DataFrame([{
        "device_model": device_model,
        "app_usage_min": app_usage_min,
        "apps_installed": apps_installed,
    }])
    model_screen_pred = float(pipelines["screen_time"].predict(screen_df)[0])

    if screen_known:
        screen_source = "user"
        screen_used = float(inputs["screen_on_hours"])
    else:
        screen_source = "model_estimate"
        screen_used = model_screen_pred

    # 2. Battery and Data usage predictions
    full_df = pd.DataFrame([{
        "device_model": device_model,
        "app_usage_min": app_usage_min,
        "screen_on_hours": screen_used,
        "apps_installed": apps_installed,
    }])
    bat_pred = float(pipelines["battery"].predict(full_df)[0])
    data_pred = float(pipelines["data"].predict(full_df)[0])

    # 3. Assess budget risks
    # Note: for the screen-time budget, the risk layer uses the screen-time model's
    # typical relative error as the uncertainty distribution, even when the user entered the value.
    screen_budget_val = float(budgets.get("screen_hours", budgets.get("screen_time", 0.0)))
    bat_risk = assess_budget(risk["battery"], bat_pred, float(budgets["battery_mah"]))
    data_risk = assess_budget(risk["data"], data_pred, float(budgets["data_mb"]))
    screen_risk = assess_budget(risk["screen_time"], model_screen_pred, screen_budget_val)

    return {
        "predictions": {
            "battery_mah": bat_pred,
            "data_mb": data_pred,
            "screen_hours": model_screen_pred,
        },
        "risk": {
            "battery": bat_risk,
            "data": data_risk,
            "screen_time": screen_risk,
        },
        "screen_source": screen_source,
        "screen_used": screen_used,
    }


def profile(
    artifacts: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    """
    Classify observed historical smartphone usage into a neutral behavioral persona.

    This function is strictly for retrospective profile analysis and must NEVER
    be called with forward-looking forecast outputs.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded system artifacts.
    observed : dict[str, Any]
        Observed daily averages: 'app_usage_min', 'screen_on_hours', 'battery_mah', 'data_mb'.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'profile_label', 'cluster_id', 'centroid_table', and 'percentile_report'.
    """
    clustering_bundle = artifacts["clustering"]
    req_cols = ["app_usage_min", "screen_on_hours", "battery_mah", "data_mb"]

    for col in req_cols:
        if col not in observed:
            raise ValueError(f"Missing required observed metric: '{col}'.")
        val = observed[col]
        try:
            val_f = float(val)
            if not np.isfinite(val_f) or val_f <= 0:
                raise ValueError(f"Observed '{col}' must be strictly positive and finite (negative or zero values not allowed), got {val}.")
        except (ValueError, TypeError) as exc:
            if "strictly positive" in str(exc):
                raise
            raise ValueError(f"Observed '{col}' must be a valid positive number, got {val}.") from exc

    obs_df = pd.DataFrame([{col: float(observed[col]) for col in req_cols}])
    pred_res = predict_profile(clustering_bundle, obs_df)
    cluster_id = int(pred_res["cluster_id"].iloc[0])
    profile_label = str(pred_res["profile_label"].iloc[0])

    pct_df = percentile_report(artifacts["reference_df"], obs_df, req_cols)

    return {
        "cluster_id": cluster_id,
        "profile_label": profile_label,
        "centroid_table": clustering_bundle["centroids_original_units"],
        "percentile_report": pct_df,
    }


def profile_history(
    artifacts: dict[str, Any],
    history_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Classify a multi-day history of observed usage records into behavioral personas.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded system artifacts.
    history_df : pd.DataFrame
        DataFrame with required columns: 'app_usage_min', 'screen_on_hours',
        'battery_mah', 'data_mb'. Extra columns (e.g. 'date') are ignored.

    Returns
    -------
    dict[str, Any]
        Dictionary containing per-row labels, profile of average row, and share of days per label.
    """
    if not isinstance(history_df, pd.DataFrame):
        raise ValueError(f"history_df must be a pandas DataFrame, got {type(history_df)}.")

    if len(history_df) < 1:
        raise ValueError("history_df must contain at least 1 record.")

    req_cols = ["app_usage_min", "screen_on_hours", "battery_mah", "data_mb"]
    missing = [c for c in req_cols if c not in history_df.columns]
    if missing:
        raise ValueError(f"history_df is missing required column(s): {missing}.")

    for col in req_cols:
        series = history_df[col]
        if series.isna().any() or np.isinf(series).any():
            raise ValueError(f"Column '{col}' in history_df contains NaN or infinite values.")
        if (series < 0).any():
            raise ValueError(f"Column '{col}' in history_df contains negative values.")

    clustering_bundle = artifacts["clustering"]
    eval_df = history_df[req_cols].copy()
    pred_df = predict_profile(clustering_bundle, eval_df)

    row_labels = list(pred_df["profile_label"])
    cluster_ids = [int(cid) for cid in pred_df["cluster_id"]]

    # Profile of average row
    avg_row = eval_df.mean().to_dict()
    avg_profile_res = profile(artifacts, avg_row)

    # Share of days per persona
    shares = pred_df["profile_label"].value_counts(normalize=True).to_dict()

    return {
        "row_labels": row_labels,
        "cluster_ids": cluster_ids,
        "average_profile": avg_profile_res,
        "label_shares": shares,
        "history_predictions": pred_df,
    }


def survival(
    artifacts: dict[str, Any],
    forecast_result: dict[str, Any] | float,
    battery_capacity_mah: float,
    current_battery_pct: float,
    current_hour: float,
    need_until_hour: float,
    min_battery_pct: float = DEFAULT_MIN_BATTERY_PCT,
) -> dict[str, Any]:
    """
    Estimate battery survival time and probability of lasting until a target hour (Feature F8).

    Methodology & Mathematical Formulation:
    - Usable Capacity:
        usable_mah = capacity * (current_pct - min_pct) / 100
    - Hourly Drain Rate:
        hourly_rate = predicted_daily_mah / WAKING_HOURS
    - Expected Hours of Remaining Life:
        hours_until_threshold = usable_mah / hourly_rate
    - Duration Needed:
        hours_needed = need_until_hour - current_hour
    - Equivalent Daily Budget:
        equivalent_daily_budget_mah = usable_mah * WAKING_HOURS / hours_needed
    - Survival Probability:
        P(make it) = 1.0 - P(drain > equivalent_daily_budget_mah)

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded system artifacts.
    forecast_result : dict or float
        Predicted daily battery consumption (or forecast dict containing it).
    battery_capacity_mah : float
        Smartphone battery rated capacity in mAh. Must be in CAPACITY_RANGE_MAH (1000 to 10000).
    current_battery_pct : float
        Current battery charge state in percent. Range: [1, 100].
    current_hour : float
        Current time in decimal hours [0, 24).
    need_until_hour : float
        Target time until which the phone is needed. Must be in (current_hour, current_hour + 24].
    min_battery_pct : float, default=DEFAULT_MIN_BATTERY_PCT
        Safety threshold reserve percentage below which phone is considered depleted. Range: [0, 99].

    Returns
    -------
    dict[str, Any]
        Dictionary with usable_mah, hourly_rate, hours_until_threshold, intervals,
        p_make_it, expected_to_make_it, assumptions, and status.
    """
    # 1. Validation
    try:
        cap = float(battery_capacity_mah)
        if not np.isfinite(cap) or cap < CAPACITY_RANGE_MAH[0] or cap > CAPACITY_RANGE_MAH[1]:
            raise ValueError(
                f"Battery capacity {battery_capacity_mah} mAh is outside valid range {CAPACITY_RANGE_MAH}."
            )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Battery capacity must be a valid number, got {battery_capacity_mah}.") from exc

    try:
        cur_pct = float(current_battery_pct)
        if not np.isfinite(cur_pct) or cur_pct < 1.0 or cur_pct > 100.0:
            raise ValueError(f"Current battery percentage {current_battery_pct}% must be between 1 and 100.")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Current battery percentage must be a valid number, got {current_battery_pct}.") from exc

    try:
        m_pct = float(min_battery_pct)
        if not np.isfinite(m_pct) or m_pct < 0.0 or m_pct > 99.0:
            raise ValueError(f"Minimum battery percentage {min_battery_pct}% must be between 0 and 99.")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Minimum battery percentage must be a valid number, got {min_battery_pct}.") from exc

    try:
        cur_h = float(current_hour)
        if not np.isfinite(cur_h) or cur_h < 0.0 or cur_h >= 24.0:
            raise ValueError(f"Current hour {current_hour} must be in range [0, 24).")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Current hour must be a valid number, got {current_hour}.") from exc

    try:
        need_h = float(need_until_hour)
        if not np.isfinite(need_h) or need_h <= cur_h or need_h > cur_h + 24.0:
            raise ValueError(
                f"need_until_hour {need_until_hour} must be strictly greater than "
                f"current_hour ({current_hour}) and at most current_hour + 24."
            )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"need_until_hour must be a valid number, got {need_until_hour}.") from exc

    # Extract predicted daily drain
    if isinstance(forecast_result, dict):
        if "predictions" in forecast_result and "battery_mah" in forecast_result["predictions"]:
            pred_daily_mah = float(forecast_result["predictions"]["battery_mah"])
        elif "battery_mah" in forecast_result:
            pred_daily_mah = float(forecast_result["battery_mah"])
        elif "battery_pred" in forecast_result:
            pred_daily_mah = float(forecast_result["battery_pred"])
        else:
            raise ValueError("forecast_result dictionary missing predicted battery drain.")
    else:
        pred_daily_mah = float(forecast_result)

    assumptions = [
        f"Daily battery drain is assumed to be distributed evenly over {WAKING_HOURS} waking hours.",
        "Battery drain during idle/sleep hours is assumed negligible compared to waking active drain.",
    ]

    hours_needed = need_h - cur_h

    # 2. Check threshold boundary condition
    if cur_pct <= m_pct:
        hourly_rate = pred_daily_mah / WAKING_HOURS
        return {
            "status": "already_at_or_below_threshold",
            "usable_mah": 0.0,
            "hourly_rate": hourly_rate,
            "hours_until_threshold": 0.0,
            "hours_until_lower": 0.0,
            "hours_until_upper": 0.0,
            "hours_needed": hours_needed,
            "equivalent_daily_budget_mah": 0.0,
            "p_make_it": 0.0,
            "expected_to_make_it": False,
            "assumptions": assumptions,
        }

    # 3. Standard survival calculation
    usable_mah = cap * (cur_pct - m_pct) / 100.0
    hourly_rate = pred_daily_mah / WAKING_HOURS
    hours_until_threshold = usable_mah / hourly_rate
    equivalent_daily_budget_mah = usable_mah * WAKING_HOURS / hours_needed

    # Probability of making it
    battery_risk_bundle = artifacts["risk"]["battery"]
    p_overrun = float(overrun_probability(battery_risk_bundle, pred_daily_mah, equivalent_daily_budget_mah))
    p_make_it = float(np.clip(1.0 - p_overrun, 0.0, 1.0))

    # 80% interval on survival hours
    lower_drain, upper_drain = predict_interval(battery_risk_bundle, pred_daily_mah, level=0.8)
    upper_hourly = upper_drain / WAKING_HOURS
    lower_hourly = lower_drain / WAKING_HOURS

    hours_until_lower = usable_mah / upper_hourly if upper_hourly > 0 else 0.0
    hours_until_upper = usable_mah / lower_hourly if lower_hourly > 0 else 1e9

    expected_to_make_it = bool(p_make_it >= 0.5)

    return {
        "status": "ok",
        "usable_mah": usable_mah,
        "hourly_rate": hourly_rate,
        "hours_until_threshold": hours_until_threshold,
        "hours_until_lower": hours_until_lower,
        "hours_until_upper": hours_until_upper,
        "hours_needed": hours_needed,
        "equivalent_daily_budget_mah": equivalent_daily_budget_mah,
        "p_make_it": p_make_it,
        "expected_to_make_it": expected_to_make_it,
        "assumptions": assumptions,
    }


def plan(
    artifacts: dict[str, Any],
    inputs: dict[str, Any],
    budgets: dict[str, Any],
    safety_margin: float = 0.0,
    allow_assumptions: bool = True,
) -> dict[str, Any]:
    """
    Search and recommend minimal action-cost behavior change plans meeting operational budgets.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded system artifacts.
    inputs : dict[str, Any]
        User inputs dictionary.
    budgets : dict[str, Any]
        Budget limits dictionary.
    safety_margin : float, default=0.0
        Safety headroom margin [0.0, 0.5].
    allow_assumptions : bool, default=True
        If False, disables Wi-Fi shift and power-saving assumptions.

    Returns
    -------
    dict[str, Any]
        Optimization recommendation dictionary from recommend().
    """
    # Validate inputs
    if "device_model" not in inputs:
        raise ValueError("Missing 'device_model' in plan inputs.")
    dev = inputs["device_model"]
    if dev not in KNOWN_DEVICES:
        raise ValueError(f"Unknown device_model '{dev}'. Supported devices: {KNOWN_DEVICES}.")

    for b_key in ("battery_mah", "data_mb", "screen_hours"):
        alt_key = "screen_time" if b_key == "screen_hours" else None
        actual_key = b_key if b_key in budgets else alt_key
        if actual_key is None or actual_key not in budgets:
            raise ValueError(f"Missing required budget: '{b_key}'.")
        val = budgets[actual_key]
        try:
            val_f = float(val)
            if not np.isfinite(val_f) or val_f <= 0:
                raise ValueError(f"Budget '{b_key}' must be strictly positive (> 0), got {val}.")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Budget '{b_key}' must be a valid positive number, got {val}.") from exc

    norm_budgets = {
        "battery_mah": float(budgets["battery_mah"]),
        "data_mb": float(budgets["data_mb"]),
        "screen_hours": float(budgets.get("screen_hours", budgets.get("screen_time", 0.0))),
    }

    return recommend(
        base=inputs,
        budgets=norm_budgets,
        pipelines=artifacts["pipelines"],
        risk_bundles=artifacts["risk"],
        bounds=artifacts["training_bounds"],
        safety_margin=safety_margin,
        allow_assumptions=allow_assumptions,
    )


def will_i_make_it(
    artifacts: dict[str, Any],
    inputs: dict[str, Any],
    survival_inputs: dict[str, Any],
    extra_budgets: dict[str, Any] | None = None,
    allow_assumptions: bool = True,
) -> dict[str, Any]:
    """
    End-to-end battery survival assessment and autonomous remediation planning.

    Evaluates whether the user's phone will survive until the target hour; if not,
    automatically activates the what-if optimizer to recommend minimal behavioral cuts.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded system artifacts.
    inputs : dict[str, Any]
        Usage forecast inputs.
    survival_inputs : dict[str, Any]
        Survival parameters: 'battery_capacity_mah', 'current_battery_pct',
        'current_hour', 'need_until_hour', and optional 'min_battery_pct'.
    extra_budgets : dict[str, Any], optional
        Additional constraints on data and screen time. Defaults to unconstrained.
    allow_assumptions : bool, default=True
        Whether optimizer may use assumed Wi-Fi shift and power saving levers.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'survival', 'forecast', 'plan', and a plain-English 'verdict'.
    """
    # 1. Calculate equivalent battery budget
    cap = float(survival_inputs["battery_capacity_mah"])
    cur_pct = float(survival_inputs["current_battery_pct"])
    min_pct = float(survival_inputs.get("min_battery_pct", DEFAULT_MIN_BATTERY_PCT))
    cur_h = float(survival_inputs["current_hour"])
    need_h = float(survival_inputs["need_until_hour"])

    hours_needed = need_h - cur_h
    if cur_pct > min_pct and hours_needed > 0:
        usable_mah = cap * (cur_pct - min_pct) / 100.0
        equiv_battery_budget = usable_mah * WAKING_HOURS / hours_needed
    else:
        equiv_battery_budget = 1.0  # nominal minimum to allow forecast validation

    extra = extra_budgets or {}
    budgets = {
        "battery_mah": equiv_battery_budget,
        "data_mb": float(extra.get("data_mb", UNCONSTRAINED_BUDGET)),
        "screen_hours": float(extra.get("screen_hours", extra.get("screen_time", UNCONSTRAINED_BUDGET))),
    }

    # 2. Run forecast and survival
    forecast_res = forecast(
        artifacts=artifacts,
        inputs=inputs,
        budgets=budgets,
        screen_known=inputs.get("screen_known", True),
    )

    survival_res = survival(
        artifacts=artifacts,
        forecast_result=forecast_res,
        battery_capacity_mah=cap,
        current_battery_pct=cur_pct,
        current_hour=cur_h,
        need_until_hour=need_h,
        min_battery_pct=min_pct,
    )

    # 3. Plan remediation if not expected to make it
    plan_res = None
    if not survival_res["expected_to_make_it"] and survival_res["status"] != "already_at_or_below_threshold":
        plan_res = plan(
            artifacts=artifacts,
            inputs=inputs,
            budgets=budgets,
            safety_margin=0.0,
            allow_assumptions=allow_assumptions,
        )

    # 4. Construct plain-English verdict
    if survival_res["status"] == "already_at_or_below_threshold":
        verdict = (
            f"Your battery is at {cur_pct:.0f}%, which is already at or below your safety threshold "
            f"of {min_pct:.0f}%."
        )
    elif survival_res["expected_to_make_it"]:
        verdict = (
            f"You will make it: your battery is expected to last {survival_res['hours_until_threshold']:.1f} hours "
            f"({survival_res['p_make_it']*100:.1f}% probability), covering your needed {survival_res['hours_needed']:.1f} hours."
        )
    else:
        if plan_res is not None and plan_res.get("status") == "plan_found":
            best = plan_res["recommended_plan"]
            cuts = []
            if best["app_cut_min"] > 0:
                cuts.append(f"cutting app usage by {best['app_cut_min']:.0f} min")
            if best["screen_cut_hours"] > 0:
                cuts.append(f"reducing screen time by {best['screen_cut_hours']:.1f} h")
            if best["power_saving"]:
                cuts.append("enabling power-saving mode")
            plan_desc = ", ".join(cuts) if cuts else "adjusting usage"
            verdict = (
                f"You are unlikely to make it (predicted survival: {survival_res['hours_until_threshold']:.1f} h "
                f"vs {survival_res['hours_needed']:.1f} h needed, P(make it)={survival_res['p_make_it']*100:.1f}%), "
                f"but a viable plan was found by {plan_desc}."
            )
        else:
            verdict = (
                f"You are unlikely to make it (predicted survival: {survival_res['hours_until_threshold']:.1f} h "
                f"vs {survival_res['hours_needed']:.1f} h needed, P(make it)={survival_res['p_make_it']*100:.1f}%), "
                f"and no viable plan was found within normal operating limits."
            )

    return {
        "survival": survival_res,
        "forecast": forecast_res,
        "plan": plan_res,
        "verdict": verdict,
    }
