"""
Evaluation script for PocketPilot What-If Planner (Feature F6).

Evaluates planner solvability across budget difficulty settings on the test split,
measures action cost and lever adoption distributions, evaluates representative
user personas, and produces report figures and CSV metrics.
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
from pocketpilot.planner.scenarios import training_bounds


def main() -> None:
    print("=" * 70)
    print("POCKETPILOT WHAT-IF PLANNER EVALUATION (FEATURE F6)")
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

    # 2. Evaluate solvability across settings
    solvability_rows = []
    lever_usage_rows = []
    cost_data_by_setting = {}

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
        over_budget_cases = []
        solved_cases = []

        # Vectorized baseline prediction for all test users
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

        for idx, row in test_df[is_over_budget].iterrows():
            base_user = {
                "device_model": str(row["device_model"]),
                "app_usage_min": float(row["app_usage_min"]),
                "screen_on_hours": float(row["screen_on_hours"]),
                "apps_installed": int(row["apps_installed"]),
            }

            rec = recommend(
                base=base_user,
                budgets=budgets,
                pipelines=pipelines,
                risk_bundles=risk_bundles,
                bounds=bounds,
                safety_margin=0.0,
            )

            over_budget_cases.append(rec)
            if rec["status"] == "plan_found":
                solved_cases.append(rec["best"])

        n_solved = len(solved_cases)
        share_solved = float(n_solved / n_over_budget) if n_over_budget > 0 else 0.0

        if n_solved > 0:
            costs = [c["action_cost"] for c in solved_cases]
            app_cuts = [c["app_cut_min"] for c in solved_cases]
            needs_assumption = [
                (c["wifi_share"] > 0.0 or c["power_saving"]) for c in solved_cases
            ]

            mean_cost = float(np.mean(costs))
            median_cost = float(np.median(costs))
            mean_app_cut = float(np.mean(app_cuts))
            share_assumption = float(np.mean(needs_assumption))
            cost_data_by_setting[setting_name] = costs

            app_cut_share = float(np.mean([c["app_cut_min"] > 0 for c in solved_cases]))
            screen_cut_share = float(np.mean([c["screen_cut_hours"] > 0.0 for c in solved_cases]))
            wifi_shift_share = float(np.mean([c["wifi_share"] > 0.0 for c in solved_cases]))
            ps_share = float(np.mean([c["power_saving"] for c in solved_cases]))
        else:
            mean_cost = 0.0
            median_cost = 0.0
            mean_app_cut = 0.0
            share_assumption = 0.0
            cost_data_by_setting[setting_name] = []
            app_cut_share = 0.0
            screen_cut_share = 0.0
            wifi_shift_share = 0.0
            ps_share = 0.0

        solvability_rows.append({
            "setting": setting_name,
            "percentile": percentile,
            "n_test": n_test,
            "n_over_budget": n_over_budget,
            "n_solved": n_solved,
            "share_solved": share_solved,
            "mean_action_cost": mean_cost,
            "median_action_cost": median_cost,
            "mean_app_cut_min": mean_app_cut,
            "share_assumption_lever": share_assumption,
        })

        lever_usage_rows.append({
            "setting": setting_name,
            "app_cut_share": app_cut_share,
            "screen_cut_share": screen_cut_share,
            "wifi_shift_share": wifi_shift_share,
            "power_saving_share": ps_share,
        })

    solvability_df = pd.DataFrame(solvability_rows)
    lever_usage_df = pd.DataFrame(lever_usage_rows)

    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    solvability_csv_path = metrics_dir / "planner_solvability.csv"
    lever_usage_csv_path = metrics_dir / "planner_lever_usage.csv"
    solvability_df.to_csv(solvability_csv_path, index=False)
    lever_usage_df.to_csv(lever_usage_csv_path, index=False)

    print("\n--- TABLE 1: PLANNER SOLVABILITY ---")
    print(solvability_df.to_string(index=False))

    print("\n--- TABLE 2: PLANNER LEVER USAGE ---")
    print(lever_usage_df.to_string(index=False))

    # 3. Three fixed example users
    # Light: Google Pixel 5, 100 min, 2.5 h, 30 apps
    # Typical: Samsung Galaxy S21, 250 min, 5.0 h, 50 apps
    # Heavy: iPhone 12, 480 min, 9.0 h, 85 apps
    # Budgets: battery 1800 mAh, data 1000 MB, screen 6.0 h
    example_users = [
        ("light", {"device_model": "Google Pixel 5", "app_usage_min": 100, "screen_on_hours": 2.5, "apps_installed": 30}),
        ("typical", {"device_model": "Samsung Galaxy S21", "app_usage_min": 250, "screen_on_hours": 5.0, "apps_installed": 50}),
        ("heavy", {"device_model": "iPhone 12", "app_usage_min": 480, "screen_on_hours": 9.0, "apps_installed": 85}),
    ]

    example_budgets = {
        "battery_mah": 1800.0,
        "data_mb": 1000.0,
        "screen_hours": 6.0,
    }

    whatif_rows = []
    for persona_name, user_base in example_users:
        rec = recommend(
            base=user_base,
            budgets=example_budgets,
            pipelines=pipelines,
            risk_bundles=risk_bundles,
            bounds=bounds,
            safety_margin=0.0,
        )

        base_res = rec["baseline"]
        best_res = rec["best"]
        risk_best = best_res["risk"]

        whatif_rows.append({
            "user_persona": persona_name,
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

    print("\n--- TABLE 3: PLANNER WHAT-IF PERSONAS ---")
    print(whatif_df.to_string(index=False))

    # 4. Generate Figures
    # Figure 1: Cost distribution (one panel per setting)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    settings_order = ["tight", "moderate", "loose"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    for ax, setting_name, color in zip(axes, settings_order, colors):
        costs = cost_data_by_setting.get(setting_name, [])
        if len(costs) > 0:
            ax.hist(costs, bins=12, color=color, edgecolor="black", alpha=0.75)
            ax.set_title(f"Setting: {setting_name.capitalize()} (P{config.PLANNER_BUDGET_SETTINGS[setting_name]})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Action Cost (Disruption Index)", fontsize=10)
            ax.set_ylabel("Number of Solved Cases", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No Solved Cases", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Setting: {setting_name.capitalize()}", fontsize=11)

    fig.suptitle("PocketPilot Planner: Action-Cost Distribution Across Budget Settings", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_cost_path = figures_dir / "planner_cost_distribution.png"
    plt.savefig(fig_cost_path, dpi=300)
    plt.close()
    print(f"\nSaved {fig_cost_path.name}")

    # Figure 2: Solvability bar chart
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x_pos = np.arange(len(solvability_df))
    bars = ax.bar(x_pos, solvability_df["share_solved"] * 100, color=["#e74c3c", "#e67e22", "#27ae60"], width=0.5, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s.capitalize()} (P{p})" for s, p in zip(solvability_df["setting"], solvability_df["percentile"])], fontsize=10)
    ax.set_ylabel("Share of Over-Budget Cases Solved (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Planner Feasibility: Share of Cases Solved", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig_solv_path = figures_dir / "planner_solvability.png"
    plt.savefig(fig_solv_path, dpi=300)
    plt.close()
    print(f"Saved {fig_solv_path.name}")

    # Figure 3: Lever usage grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    levers = ["app_cut_share", "screen_cut_share", "wifi_shift_share", "power_saving_share"]
    lever_names = ["App Time Cut", "Extra Screen Cut", "Wi-Fi Shift (Assump.)", "Power Saving (Assump.)"]
    bar_width = 0.2
    x_base = np.arange(len(solvability_df))

    for idx, (lever_col, lever_label) in enumerate(zip(levers, lever_names)):
        vals = lever_usage_df[lever_col] * 100
        offset = (idx - 1.5) * bar_width
        ax.bar(x_base + offset, vals, width=bar_width, label=lever_label, edgecolor="black")

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{s.capitalize()} (P{p})" for s, p in zip(solvability_df["setting"], solvability_df["percentile"])], fontsize=10)
    ax.set_ylabel("Share of Solved Cases Utilizing Lever (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Adoption Rate of Planner Levers Across Settings", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_lever_path = figures_dir / "planner_lever_usage.png"
    plt.savefig(fig_lever_path, dpi=300)
    plt.close()
    print(f"Saved {fig_lever_path.name}")


if __name__ == "__main__":
    main()
