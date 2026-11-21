"""
Evaluation script for PocketPilot What-If Planner (Feature F6 Hardening).

Evaluates planner solvability across budget difficulty settings on the test split
in both all_levers and learned_only modes, measures action cost and lever adoption,
evaluates representative user personas across both modes, produces follow-through check,
and outputs report figures and CSV metrics.
"""

from pathlib import Path
import sys

# Insert src folder into sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pocketpilot import config
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP
from pocketpilot.planner.optimizer import recommend
from pocketpilot.planner.scenarios import build_scenario_grid, evaluate_scenarios, training_bounds


def main() -> None:
    print("=" * 70)
    print("POCKETPILOT WHAT-IF PLANNER HARDENED EVALUATION")
    print("=" * 70)

    # 1. Load splits and models
    train_df = pd.read_csv(config.PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(config.PROCESSED_TEST_PATH)
    bounds = training_bounds(train_df)

    pipelines = {
        task_name: joblib.load(path)
        for task_name, path in config.MODEL_PATHS.items()
    }
    risk_bundles = joblib.load(config.RISK_MODEL_PATH)

    print(f"Loaded train split: {len(train_df)} rows, test split: {len(test_df)} rows.")
    print(f"Training bounds: app_min={bounds['app_min']} min, screen_min={bounds['screen_min']} hours.")

    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 2. Evaluate solvability and lever usage across modes and settings
    solvability_rows = []
    lever_usage_rows = []
    cost_data = {"all_levers": {}, "learned_only": {}}

    for setting_name, percentile in config.PLANNER_BUDGET_SETTINGS.items():
        bat_budget = float(np.percentile(train_df[config.TARGET_BATTERY], percentile))
        data_budget = float(np.percentile(train_df[config.TARGET_DATA], percentile))
        screen_budget = float(np.percentile(train_df[config.TARGET_SCREEN], percentile))

        budgets = {
            "battery_mah": bat_budget,
            "data_mb": data_budget,
            "screen_hours": screen_budget,
        }

        n_test = len(test_df)

        # Baseline prediction for test users
        raw_bat_in = test_df[TASK_RAW_INPUTS_MAP["battery"]]
        raw_data_in = test_df[TASK_RAW_INPUTS_MAP["data"]]

        base_bat_preds = pipelines["battery"].predict(raw_bat_in)
        base_data_preds = pipelines["data"].predict(raw_data_in)
        base_screen_preds = test_df["screen_on_hours"].values

        is_over_budget = (
            (base_bat_preds > bat_budget)
            | (base_data_preds > data_budget)
            | (base_screen_preds > screen_budget)
        )

        n_over_budget = int(is_over_budget.sum())

        solved_cases_by_mode = {"all_levers": {}, "learned_only": {}}

        for idx, row in test_df[is_over_budget].iterrows():
            base_user = {
                "device_model": str(row["device_model"]),
                "app_usage_min": float(row["app_usage_min"]),
                "screen_on_hours": float(row["screen_on_hours"]),
                "apps_installed": int(row["apps_installed"]),
            }

            # Evaluate all_levers
            rec_all = recommend(
                base=base_user,
                budgets=budgets,
                pipelines=pipelines,
                risk_bundles=risk_bundles,
                bounds=bounds,
                safety_margin=0.0,
                allow_assumptions=True,
            )
            if rec_all["status"] == "plan_found":
                solved_cases_by_mode["all_levers"][idx] = rec_all["best"]

            # Evaluate learned_only
            rec_learned = recommend(
                base=base_user,
                budgets=budgets,
                pipelines=pipelines,
                risk_bundles=risk_bundles,
                bounds=bounds,
                safety_margin=0.0,
                allow_assumptions=False,
            )
            if rec_learned["status"] == "plan_found":
                solved_cases_by_mode["learned_only"][idx] = rec_learned["best"]

        # Calculate metrics for both modes
        all_solved_idx = set(solved_cases_by_mode["all_levers"].keys())
        learned_solved_idx = set(solved_cases_by_mode["learned_only"].keys())

        # Requiring assumption levers: solved in all_levers but NOT solved in learned_only
        requiring_assumptions_idx = all_solved_idx - learned_solved_idx
        share_requiring = (
            float(len(requiring_assumptions_idx) / len(all_solved_idx))
            if len(all_solved_idx) > 0
            else 0.0
        )

        for mode_name in ("all_levers", "learned_only"):
            solved_cases = list(solved_cases_by_mode[mode_name].values())
            n_solved = len(solved_cases)
            share_solved = float(n_solved / n_over_budget) if n_over_budget > 0 else 0.0

            if n_solved > 0:
                costs = [c["action_cost"] for c in solved_cases]
                app_cuts = [c["app_cut_min"] for c in solved_cases]
                using_assumption = [
                    (c["wifi_share"] > 0.0 or c["power_saving"]) for c in solved_cases
                ]

                mean_cost = float(np.mean(costs))
                median_cost = float(np.median(costs))
                mean_app_cut = float(np.mean(app_cuts))
                share_using = float(np.mean(using_assumption))
                cost_data[mode_name][setting_name] = costs

                app_cut_share = float(np.mean([c["app_cut_min"] > 0 for c in solved_cases]))
                screen_cut_share = float(np.mean([c["screen_cut_hours"] > 0.0 for c in solved_cases]))
                wifi_shift_share = float(np.mean([c["wifi_share"] > 0.0 for c in solved_cases]))
                ps_share = float(np.mean([c["power_saving"] for c in solved_cases]))
            else:
                mean_cost = 0.0
                median_cost = 0.0
                mean_app_cut = 0.0
                share_using = 0.0
                cost_data[mode_name][setting_name] = []
                app_cut_share = 0.0
                screen_cut_share = 0.0
                wifi_shift_share = 0.0
                ps_share = 0.0

            req_val = share_requiring if mode_name == "all_levers" else 0.0

            solvability_rows.append({
                "mode": mode_name,
                "setting": setting_name,
                "percentile": percentile,
                "n_test": n_test,
                "n_over_budget": n_over_budget,
                "n_solved": n_solved,
                "share_solved": share_solved,
                "mean_action_cost": mean_cost,
                "median_action_cost": median_cost,
                "mean_app_cut_min": mean_app_cut,
                "share_using_assumption_levers": share_using,
                "share_requiring_assumption_levers": req_val,
            })

            lever_usage_rows.append({
                "mode": mode_name,
                "setting": setting_name,
                "app_cut_share": app_cut_share,
                "screen_cut_share": screen_cut_share,
                "wifi_shift_share": wifi_shift_share,
                "power_saving_share": ps_share,
            })

    # Sort rows so all_levers come first, then learned_only, or by mode and setting
    # Keep consistent with prior ordering
    solvability_rows.sort(key=lambda r: (0 if r["mode"] == "all_levers" else 1, r["percentile"]))
    lever_usage_rows.sort(key=lambda r: (0 if r["mode"] == "all_levers" else 1, r["setting"]))

    solvability_df = pd.DataFrame(solvability_rows)
    lever_usage_df = pd.DataFrame(lever_usage_rows)

    solvability_csv_path = metrics_dir / "planner_solvability.csv"
    lever_usage_csv_path = metrics_dir / "planner_lever_usage.csv"
    solvability_df.to_csv(solvability_csv_path, index=False)
    lever_usage_df.to_csv(lever_usage_csv_path, index=False)

    print("\n--- TABLE 1: PLANNER SOLVABILITY (BOTH MODES) ---")
    print(solvability_df.to_string(index=False))

    print("\n--- TABLE 2: PLANNER LEVER USAGE (BOTH MODES) ---")
    print(lever_usage_df.to_string(index=False))

    # 3. Personas in both modes
    # Light, Typical, Above Average, Heavy
    personas = [
        ("light", {"device_model": "Google Pixel 5", "app_usage_min": 100, "screen_on_hours": 2.5, "apps_installed": 30}),
        ("typical", {"device_model": "Samsung Galaxy S21", "app_usage_min": 250, "screen_on_hours": 5.0, "apps_installed": 50}),
        ("above_average", {"device_model": "Xiaomi Mi 11", "app_usage_min": 350, "screen_on_hours": 6.5, "apps_installed": 65}),
        ("heavy", {"device_model": "iPhone 12", "app_usage_min": 480, "screen_on_hours": 9.0, "apps_installed": 85}),
    ]

    persona_budgets = {
        "battery_mah": 1800.0,
        "data_mb": 1000.0,
        "screen_hours": 6.0,
    }

    whatif_rows = []
    modes = [("all_levers", True), ("learned_only", False)]
    for persona_name, user_base in personas:
        for mode_name, allow_assump in modes:
            rec = recommend(
                base=user_base,
                budgets=persona_budgets,
                pipelines=pipelines,
                risk_bundles=risk_bundles,
                bounds=bounds,
                safety_margin=0.0,
                allow_assumptions=allow_assump,
            )

            base_res = rec["baseline"]
            best_res = rec["best"]
            risk_best = best_res["risk"]

            whatif_rows.append({
                "user_persona": persona_name,
                "mode": mode_name,
                "device_model": user_base["device_model"],
                "app_usage_min": user_base["app_usage_min"],
                "screen_on_hours": user_base["screen_on_hours"],
                "apps_installed": user_base["apps_installed"],
                "status": rec["status"],
                "baseline_bat_pred": round(base_res["battery_pred"], 1),
                "baseline_data_pred": round(base_res["data_pred"], 1),
                "baseline_screen_pred": round(base_res["screen_pred"], 2),
                "plan_app_cut_min": best_res["app_cut_min"],
                "plan_screen_cut_hours": best_res["screen_cut_hours"],
                "plan_wifi_share": best_res["wifi_share"],
                "plan_power_saving": best_res["power_saving"],
                "plan_action_cost": round(best_res["action_cost"], 1),
                "post_bat_pred": round(best_res["battery_pred"], 1),
                "post_data_pred": round(best_res["data_pred"], 1),
                "post_screen_pred": round(best_res["screen_pred"], 2),
                "bat_p_overrun": round(risk_best["battery"]["p_overrun"], 4),
                "data_p_overrun": round(risk_best["data"]["p_overrun"], 4),
                "screen_p_overrun": round(risk_best["screen_time"]["p_overrun"], 4),
            })

    whatif_df = pd.DataFrame(whatif_rows)
    whatif_csv_path = metrics_dir / "planner_whatif_cases.csv"
    whatif_df.to_csv(whatif_csv_path, index=False)

    print("\n--- TABLE 3: PLANNER WHAT-IF PERSONAS (BOTH MODES) ---")
    print(whatif_df.to_string(index=False))

    # 4. Follow-through check CSV & Test split change share
    followthrough_rows = []
    grid_full = build_scenario_grid()

    for persona_name, user_base in personas:
        eval_p = evaluate_scenarios(user_base, grid_full, pipelines, bounds)
        cuts_p = eval_p.drop_duplicates("app_cut_min").sort_values("app_cut_min")
        for _, r in cuts_p.iterrows():
            followthrough_rows.append({
                "persona": persona_name,
                "device_model": user_base["device_model"],
                "app_usage_min": user_base["app_usage_min"],
                "app_cut_min": int(r["app_cut_min"]),
                "delta_screen_raw": round(float(r["delta_screen_raw"]), 4),
                "delta_screen_bounded": round(float(r["delta_screen"]), 4),
                "bound_active": bool(not np.isclose(r["delta_screen_raw"], r["delta_screen"])),
            })

    followthrough_df = pd.DataFrame(followthrough_rows)
    ft_csv_path = metrics_dir / "planner_followthrough_check.csv"
    followthrough_df.to_csv(ft_csv_path, index=False)

    # Compute overall test split change share
    n_changed = 0
    n_total = 0
    for _, row in test_df.iterrows():
        base = {
            "device_model": row["device_model"],
            "app_usage_min": row["app_usage_min"],
            "screen_on_hours": row["screen_on_hours"],
            "apps_installed": row["apps_installed"],
        }
        eval_df = evaluate_scenarios(base, grid_full, pipelines, bounds)
        unique_cuts = eval_df.drop_duplicates("app_cut_min")
        for _, u in unique_cuts.iterrows():
            if not np.isclose(u["delta_screen_raw"], u["delta_screen"]):
                n_changed += 1
            n_total += 1

    share_changed = (n_changed / n_total) * 100.0 if n_total > 0 else 0.0

    print("\n--- TABLE 4: PLANNER FOLLOW-THROUGH CHECK (SAMPLE PERSONAS) ---")
    print(followthrough_df.to_string(index=False))
    print(f"\nOverall Test Split Follow-Through Bound Share: {n_changed} / {n_total} pairs ({share_changed:.2f}%)")

    # 5. Generate Figures Showing Both Modes
    # Figure 1: Cost distribution (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    settings_order = ["tight", "moderate", "loose"]
    colors = {"all_levers": "#e74c3c", "learned_only": "#2980b9"}

    for row_idx, mode_name in enumerate(["all_levers", "learned_only"]):
        for col_idx, setting_name in enumerate(settings_order):
            ax = axes[row_idx, col_idx]
            costs = cost_data[mode_name].get(setting_name, [])
            if len(costs) > 0:
                ax.hist(costs, bins=12, color=colors[mode_name], edgecolor="black", alpha=0.75)
                ax.set_title(f"{mode_name.replace('_', ' ').title()}: {setting_name.capitalize()} (P{config.PLANNER_BUDGET_SETTINGS[setting_name]})", fontsize=10, fontweight="bold")
                ax.set_xlabel("Action Cost (Disruption)", fontsize=9)
                ax.set_ylabel("Number of Solved Cases", fontsize=9)
                ax.grid(True, linestyle="--", alpha=0.5)
            else:
                ax.text(0.5, 0.5, "No Solved Cases", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{mode_name.replace('_', ' ').title()}: {setting_name.capitalize()}", fontsize=10)

    fig.suptitle("PocketPilot Planner: Action Cost Distributions (All Levers vs. Learned Only)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_cost_path = figures_dir / "planner_cost_distribution.png"
    plt.savefig(fig_cost_path, dpi=300)
    plt.close()
    print(f"\nSaved {fig_cost_path.name}")

    # Figure 2: Solvability bar chart (Grouped by setting: All Levers vs Learned Only)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(config.PLANNER_BUDGET_SETTINGS))
    width = 0.35

    all_shares = solvability_df[solvability_df["mode"] == "all_levers"]["share_solved"].values * 100
    learned_shares = solvability_df[solvability_df["mode"] == "learned_only"]["share_solved"].values * 100

    bars1 = ax.bar(x_pos - width / 2, all_shares, width, label="All Levers (Incl. Assumptions)", color="#27ae60", edgecolor="black")
    bars2 = ax.bar(x_pos + width / 2, learned_shares, width, label="Learned Only (No Assumptions)", color="#2980b9", edgecolor="black")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s.capitalize()} (P{p})" for s, p in config.PLANNER_BUDGET_SETTINGS.items()], fontsize=10)
    ax.set_ylabel("Share of Over-Budget Cases Solved (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Planner Feasibility: All Levers vs. Learned Only", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_solv_path = figures_dir / "planner_solvability.png"
    plt.savefig(fig_solv_path, dpi=300)
    plt.close()
    print(f"Saved {fig_solv_path.name}")

    # Figure 3: Lever usage grouped bar chart (2 panels: All Levers vs Learned Only)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    levers = ["app_cut_share", "screen_cut_share", "wifi_shift_share", "power_saving_share"]
    lever_names = ["App Time Cut", "Extra Screen Cut", "Wi-Fi Shift", "Power Saving"]
    bar_width = 0.2
    x_base = np.arange(len(config.PLANNER_BUDGET_SETTINGS))

    for ax_idx, mode_name in enumerate(["all_levers", "learned_only"]):
        ax = axes[ax_idx]
        sub_df = lever_usage_df[lever_usage_df["mode"] == mode_name]
        for idx, (lever_col, lever_label) in enumerate(zip(levers, lever_names)):
            vals = sub_df[lever_col].values * 100
            offset = (idx - 1.5) * bar_width
            ax.bar(x_base + offset, vals, width=bar_width, label=lever_label, edgecolor="black")

        ax.set_xticks(x_base)
        ax.set_xticklabels([f"{s.capitalize()} (P{p})" for s, p in config.PLANNER_BUDGET_SETTINGS.items()], fontsize=10)
        ax.set_title(f"Lever Adoption: {mode_name.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if ax_idx == 0:
            ax.set_ylabel("Adoption Rate in Solved Cases (%)", fontsize=11)
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle("PocketPilot Planner: Lever Adoption by Mode Across Settings", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_lever_path = figures_dir / "planner_lever_usage.png"
    plt.savefig(fig_lever_path, dpi=300)
    plt.close()
    print(f"Saved {fig_lever_path.name}")

    # Single recommend() call check on typical persona
    typical_persona = {
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 250.0,
        "screen_on_hours": 5.0,
        "apps_installed": 50,
    }
    typical_budgets = {"battery_mah": 1800.0, "data_mb": 1000.0, "screen_hours": 6.0}
    rec_typ_all = recommend(typical_persona, typical_budgets, pipelines, risk_bundles, bounds, allow_assumptions=True)
    rec_typ_learned = recommend(typical_persona, typical_budgets, pipelines, risk_bundles, bounds, allow_assumptions=False)
    print("\n--- SINGLE RECOMMEND CALL EVALUATION (TYPICAL PERSONA) ---")
    print(f"  * all_levers mode:   n_evaluated = {rec_typ_all['n_evaluated']} (expected: 224)")
    print(f"  * learned_only mode: n_evaluated = {rec_typ_learned['n_evaluated']} (expected: 28)")


if __name__ == "__main__":
    main()
