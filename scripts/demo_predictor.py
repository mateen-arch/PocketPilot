"""
Demonstration script exercising PocketPilotPredictor on realistic user scenarios (Feature F8).

Loads production artifacts and runs end-to-end inference for:
1. Typical user persona: Samsung Galaxy S21, 250 min app usage, 5.0 h screen time, 50 apps.
   Battery capacity 4000 mAh, currently 38% at 16:00, needed until 22:00, 15% reserve limit.
   Budgets: 1800 mAh, 1000 MB, 6.0 h.
2. Heavy user persona: iPhone 12, 480 min app usage, 9.0 h screen time, 85 apps.
   Battery capacity 3000 mAh, currently 40% at 14:00, needed until 23:00, 15% reserve limit.
   Budgets: 1800 mAh, 1000 MB, 6.0 h.

Output is saved to reports/testing_evidence/predictor_demo.txt (UTF-8).
"""

import sys
from pathlib import Path

# Add project src directory to sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
from pocketpilot import config
from pocketpilot.inference.predictor import load_artifacts, will_i_make_it

EVIDENCE_FILE = config.PROJECT_ROOT / "reports" / "testing_evidence" / "predictor_demo.txt"


def format_demo_output() -> str:
    lines = []
    lines.append("=" * 75)
    lines.append("POCKETPILOT UNIFIED PREDICTOR DEMONSTRATION (FEATURE F8)")
    lines.append("=" * 75)

    artifacts = load_artifacts()
    lines.append(f"Loaded artifacts successfully from: {config.MODELS_DIR.name}/")
    if artifacts["warnings"]:
        for w in artifacts["warnings"]:
            lines.append(f"  [Operational Notice] {w}")
    lines.append("")

    # Scenario 1: Typical User
    lines.append("-" * 75)
    lines.append("SCENARIO 1: TYPICAL USER")
    lines.append("-" * 75)
    inputs_1 = {
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 250,
        "screen_on_hours": 5.0,
        "apps_installed": 50,
        "screen_known": True,
    }
    survival_1 = {
        "battery_capacity_mah": 4000,
        "current_battery_pct": 38,
        "current_hour": 16.0,
        "need_until_hour": 22.0,
        "min_battery_pct": 15,
    }
    extra_budgets_1 = {
        "battery_mah": 1800.0,
        "data_mb": 1000.0,
        "screen_hours": 6.0,
    }

    lines.append(f"Device: {inputs_1['device_model']} | App Usage: {inputs_1['app_usage_min']} min | "
                 f"Screen: {inputs_1['screen_on_hours']} h | Apps: {inputs_1['apps_installed']}")
    lines.append(f"Battery State: {survival_1['current_battery_pct']}% at {survival_1['current_hour']:.1f}:00 | "
                 f"Target: {survival_1['need_until_hour']:.1f}:00 | Min Reserve: {survival_1['min_battery_pct']}% | "
                 f"Capacity: {survival_1['battery_capacity_mah']} mAh")
    lines.append(f"Budgets: Battery={extra_budgets_1['battery_mah']} mAh, Data={extra_budgets_1['data_mb']} MB, "
                 f"Screen={extra_budgets_1['screen_hours']} h")
    lines.append("")

    res_1 = will_i_make_it(
        artifacts=artifacts,
        inputs=inputs_1,
        survival_inputs=survival_1,
        extra_budgets=extra_budgets_1,
        allow_assumptions=True,
    )

    f_1 = res_1["forecast"]["predictions"]
    r_1 = res_1["forecast"]["risk"]
    s_1 = res_1["survival"]

    lines.append("1. FORECAST & RISK:")
    lines.append(f"   * Battery Drain: {f_1['battery_mah']:.1f} mAh (Overrun Risk: {r_1['battery']['level']}, P(overrun)={r_1['battery']['p_overrun']*100:.1f}%)")
    lines.append(f"   * Data Usage:    {f_1['data_mb']:.1f} MB (Overrun Risk: {r_1['data']['level']}, P(overrun)={r_1['data']['p_overrun']*100:.1f}%)")
    lines.append(f"   * Screen Time:   {f_1['screen_hours']:.1f} h (Overrun Risk: {r_1['screen_time']['level']}, P(overrun)={r_1['screen_time']['p_overrun']*100:.1f}%)")
    lines.append("")
    lines.append("2. BATTERY SURVIVAL:")
    lines.append(f"   * Usable Capacity:       {s_1['usable_mah']:.1f} mAh")
    lines.append(f"   * Hourly Drain Rate:     {s_1['hourly_rate']:.1f} mAh/h (over {config.WAKING_HOURS} waking hours)")
    lines.append(f"   * Expected Survival:     {s_1['hours_until_threshold']:.1f} hours [80% Interval: {s_1['hours_until_lower']:.1f} - {s_1['hours_until_upper']:.1f} h]")
    lines.append(f"   * Duration Needed:       {s_1['hours_needed']:.1f} hours (until {survival_1['need_until_hour']:.1f}:00)")
    lines.append(f"   * Equivalent Daily Cap:  {s_1['equivalent_daily_budget_mah']:.1f} mAh")
    lines.append(f"   * P(make it):            {s_1['p_make_it']*100:.1f}%")
    lines.append(f"   * Expected to Make It:   {s_1['expected_to_make_it']}")
    lines.append("")
    lines.append(f"3. VERDICT: \"{res_1['verdict']}\"")
    if res_1["plan"] and res_1["plan"].get("recommended_plan"):
        best_p = res_1["plan"]["recommended_plan"]
        lines.append(f"   * Recommended Plan: Cost={best_p['action_cost']:.1f}, Cut Apps={best_p['app_cut_min']} min, "
                     f"Cut Screen={best_p['screen_cut_hours']} h, Power Saving={best_p['power_saving']}")
    lines.append("")

    # Scenario 2: Heavy User
    lines.append("-" * 75)
    lines.append("SCENARIO 2: HEAVY USER")
    lines.append("-" * 75)
    inputs_2 = {
        "device_model": "iPhone 12",
        "app_usage_min": 480,
        "screen_on_hours": 9.0,
        "apps_installed": 85,
        "screen_known": True,
    }
    survival_2 = {
        "battery_capacity_mah": 3000,
        "current_battery_pct": 40,
        "current_hour": 14.0,
        "need_until_hour": 23.0,
        "min_battery_pct": 15,
    }
    extra_budgets_2 = {
        "battery_mah": 1800.0,
        "data_mb": 1000.0,
        "screen_hours": 6.0,
    }

    lines.append(f"Device: {inputs_2['device_model']} | App Usage: {inputs_2['app_usage_min']} min | "
                 f"Screen: {inputs_2['screen_on_hours']} h | Apps: {inputs_2['apps_installed']}")
    lines.append(f"Battery State: {survival_2['current_battery_pct']}% at {survival_2['current_hour']:.1f}:00 | "
                 f"Target: {survival_2['need_until_hour']:.1f}:00 | Min Reserve: {survival_2['min_battery_pct']}% | "
                 f"Capacity: {survival_2['battery_capacity_mah']} mAh")
    lines.append(f"Budgets: Battery={extra_budgets_2['battery_mah']} mAh, Data={extra_budgets_2['data_mb']} MB, "
                 f"Screen={extra_budgets_2['screen_hours']} h")
    lines.append("")

    res_2 = will_i_make_it(
        artifacts=artifacts,
        inputs=inputs_2,
        survival_inputs=survival_2,
        extra_budgets=extra_budgets_2,
        allow_assumptions=True,
    )

    f_2 = res_2["forecast"]["predictions"]
    r_2 = res_2["forecast"]["risk"]
    s_2 = res_2["survival"]

    lines.append("1. FORECAST & RISK:")
    lines.append(f"   * Battery Drain: {f_2['battery_mah']:.1f} mAh (Overrun Risk: {r_2['battery']['level']}, P(overrun)={r_2['battery']['p_overrun']*100:.1f}%)")
    lines.append(f"   * Data Usage:    {f_2['data_mb']:.1f} MB (Overrun Risk: {r_2['data']['level']}, P(overrun)={r_2['data']['p_overrun']*100:.1f}%)")
    lines.append(f"   * Screen Time:   {f_2['screen_hours']:.1f} h (Overrun Risk: {r_2['screen_time']['level']}, P(overrun)={r_2['screen_time']['p_overrun']*100:.1f}%)")
    lines.append("")
    lines.append("2. BATTERY SURVIVAL:")
    lines.append(f"   * Usable Capacity:       {s_2['usable_mah']:.1f} mAh")
    lines.append(f"   * Hourly Drain Rate:     {s_2['hourly_rate']:.1f} mAh/h (over {config.WAKING_HOURS} waking hours)")
    lines.append(f"   * Expected Survival:     {s_2['hours_until_threshold']:.1f} hours [80% Interval: {s_2['hours_until_lower']:.1f} - {s_2['hours_until_upper']:.1f} h]")
    lines.append(f"   * Duration Needed:       {s_2['hours_needed']:.1f} hours (until {survival_2['need_until_hour']:.1f}:00)")
    lines.append(f"   * Equivalent Daily Cap:  {s_2['equivalent_daily_budget_mah']:.1f} mAh")
    lines.append(f"   * P(make it):            {s_2['p_make_it']*100:.1f}%")
    lines.append(f"   * Expected to Make It:   {s_2['expected_to_make_it']}")
    lines.append("")
    lines.append(f"3. VERDICT: \"{res_2['verdict']}\"")
    if res_2["plan"] and res_2["plan"].get("recommended_plan"):
        best_p2 = res_2["plan"]["recommended_plan"]
        lines.append(f"   * Recommended Plan: Cost={best_p2['action_cost']:.1f}, Cut Apps={best_p2['app_cut_min']} min, "
                     f"Cut Screen={best_p2['screen_cut_hours']} h, Power Saving={best_p2['power_saving']}")
    lines.append("=" * 75)

    return "\n".join(lines)


def main() -> None:
    output_text = format_demo_output()
    print(output_text)

    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE_FILE, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")
    print(f"\nSaved demonstration evidence to: {EVIDENCE_FILE}")


if __name__ == "__main__":
    main()
