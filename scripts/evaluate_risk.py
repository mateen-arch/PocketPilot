"""
Evaluation script for the budget-overrun risk layer.

Evaluates deterministic threshold rules and probabilistic risk estimates across a grid of budgets
on the held-out test split, generating:
- reports/metrics/risk_classification.csv
- reports/metrics/risk_level_outcomes.csv
- reports/metrics/risk_calibration.csv
- reports/metrics/risk_interval_coverage.csv
- reports/figures/risk_reliability_{task}.png
- reports/figures/risk_interval_coverage.png
- reports/figures/risk_f1_by_budget.png
"""

from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pocketpilot.config import (
    BUDGET_PERCENTILES,
    INTERVAL_LEVELS,
    MODEL_PATHS,
    OVERRUN_PROB_DECISION,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    RISK_MODEL_PATH,
    WATCH_FRACTION,
)
from pocketpilot.evaluation.classification_metrics import (
    brier_score,
    expected_calibration_error,
    interval_coverage,
    overrun_classification_metrics,
    reliability_table,
)
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP, TASK_TARGET_MAP
from pocketpilot.risk.probabilistic_risk import overrun_probability, predict_interval
from pocketpilot.risk.threshold_risk import risk_levels

TASKS = ["battery", "data", "screen_time"]


def main() -> None:
    print("=" * 80)
    print("PocketPilot: Evaluating Budget-Overrun Risk on Held-Out Test Split")
    print("=" * 80)

    reports_metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    reports_figures_dir = PROJECT_ROOT / "reports" / "figures"
    reports_metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_figures_dir.mkdir(parents=True, exist_ok=True)

    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Risk models not found at {RISK_MODEL_PATH}. Run scripts/train_risk.py first.")

    risk_bundles = joblib.load(RISK_MODEL_PATH)
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    classification_rows: list[dict] = []
    level_outcome_rows: list[dict] = []
    calibration_rows: list[dict] = []
    interval_rows: list[dict] = []

    task_pooled_probs: dict[str, list[float]] = {t: [] for t in TASKS}
    task_pooled_labels: dict[str, list[int]] = {t: [] for t in TASKS}
    task_eces: dict[str, float] = {}

    for task in TASKS:
        print(f"\n--- Evaluating Risk for Task: {task.upper()} ---")
        pipeline = joblib.load(MODEL_PATHS[task])
        bundle = risk_bundles[task]
        target_col = TASK_TARGET_MAP[task]
        raw_inputs = TASK_RAW_INPUTS_MAP[task]

        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        X_test = test_df[raw_inputs]

        y_test_pred = pipeline.predict(X_test)

        # 1. Prediction interval coverage across test split
        for level in INTERVAL_LEVELS:
            lower, upper = predict_interval(bundle, y_test_pred, level=level)
            emp_cov = interval_coverage(y_test, lower, upper)
            interval_rows.append({
                "task": task,
                "nominal_level": level,
                "empirical_coverage": emp_cov,
            })
            print(f"  Interval Level {int(level*100)}%: Nominal = {level:.2f}, Empirical = {emp_cov:.4f}")

        # 2. Budget grid evaluation
        for pct in BUDGET_PERCENTILES:
            budget_val = float(np.percentile(y_train, pct))
            y_true_exceeds = (y_test > budget_val).astype(int)

            # Threshold rule: alert if level == 'Likely exceed'
            t_levels = risk_levels(y_test_pred, budget_val, watch_fraction=WATCH_FRACTION)
            y_pred_thresh = (t_levels == "Likely exceed").astype(int)
            thresh_metrics = overrun_classification_metrics(y_true_exceeds, y_pred_thresh)

            classification_rows.append({
                "task": task,
                "budget_percentile": pct,
                "budget_value": budget_val,
                "rule": "threshold",
                "precision": thresh_metrics["precision"],
                "recall": thresh_metrics["recall"],
                "f1": thresh_metrics["f1"],
                "support_positive": thresh_metrics["support_positive"],
                "support_total": thresh_metrics["support_total"],
                "brier": np.nan,
            })

            # Probabilistic rule: alert if p_overrun >= OVERRUN_PROB_DECISION (0.5)
            p_overrun = overrun_probability(bundle, y_test_pred, budget_val)
            y_pred_prob = (p_overrun >= OVERRUN_PROB_DECISION).astype(int)
            prob_metrics = overrun_classification_metrics(y_true_exceeds, y_pred_prob)
            bs = brier_score(y_true_exceeds, p_overrun)

            classification_rows.append({
                "task": task,
                "budget_percentile": pct,
                "budget_value": budget_val,
                "rule": "probabilistic",
                "precision": prob_metrics["precision"],
                "recall": prob_metrics["recall"],
                "f1": prob_metrics["f1"],
                "support_positive": prob_metrics["support_positive"],
                "support_total": prob_metrics["support_total"],
                "brier": bs,
            })

            # Record pooled probabilities for calibration
            task_pooled_probs[task].extend(list(p_overrun))
            task_pooled_labels[task].extend(list(y_true_exceeds))

            # Record outcome rate per risk level (Safe, Watch, Likely exceed)
            for lvl in ["Safe", "Watch", "Likely exceed"]:
                lvl_mask = (t_levels == lvl)
                lvl_count = int(np.sum(lvl_mask))
                if lvl_count > 0:
                    exceed_rate = float(np.mean(y_true_exceeds[lvl_mask]))
                else:
                    exceed_rate = 0.0

                level_outcome_rows.append({
                    "task": task,
                    "budget_percentile": pct,
                    "level": lvl,
                    "count": lvl_count,
                    "observed_exceed_rate": exceed_rate,
                })

        # Compute pooled calibration table and ECE for this task
        rel_tab = reliability_table(task_pooled_labels[task], task_pooled_probs[task], n_bins=5)
        task_ece = expected_calibration_error(rel_tab)
        task_eces[task] = task_ece
        rel_tab.insert(0, "task", task)
        rel_tab["ece"] = task_ece
        calibration_rows.extend(rel_tab.to_dict(orient="records"))
        print(f"  Pooled Expected Calibration Error (ECE): {task_ece:.4f}")

    # Save metrics CSVs
    df_class = pd.DataFrame(classification_rows)
    df_level = pd.DataFrame(level_outcome_rows)
    df_calib = pd.DataFrame(calibration_rows)
    df_cov = pd.DataFrame(interval_rows)

    df_class.to_csv(reports_metrics_dir / "risk_classification.csv", index=False)
    df_level.to_csv(reports_metrics_dir / "risk_level_outcomes.csv", index=False)
    df_calib.to_csv(reports_metrics_dir / "risk_calibration.csv", index=False)
    df_cov.to_csv(reports_metrics_dir / "risk_interval_coverage.csv", index=False)

    print("\nSaved metric reports:")
    print("  -> reports/metrics/risk_classification.csv")
    print("  -> reports/metrics/risk_level_outcomes.csv")
    print("  -> reports/metrics/risk_calibration.csv")
    print("  -> reports/metrics/risk_interval_coverage.csv")

    # Generate Figures
    print("\nGenerating diagnostic risk figures...")

    # Figure 1: Reliability diagrams for each task
    for task in TASKS:
        t_cal = df_calib[df_calib["task"] == task]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect Calibration")
        valid_bins = t_cal[t_cal["count"] > 0]
        ax.plot(
            valid_bins["mean_predicted_probability"],
            valid_bins["observed_exceed_rate"],
            marker="o",
            linewidth=2,
            color="#2563EB",
            label=f"Observed Calibration (ECE={task_eces[task]:.3f})",
        )
        ax.set_title(f"Risk Calibration: {task.replace('_', ' ').title()} Overrun", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Overrun Probability", fontsize=10)
        ax.set_ylabel("Observed Overrun Rate", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        fig_path = reports_figures_dir / f"risk_reliability_{task}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"  Saved figure -> {fig_path}")

    # Figure 2: Prediction Interval Coverage (Nominal vs Empirical)
    fig, ax = plt.subplots(figsize=(7, 5))
    tasks_arr = np.arange(len(TASKS))
    bar_width = 0.35

    cov_80 = [df_cov[(df_cov["task"] == t) & (df_cov["nominal_level"] == 0.8)]["empirical_coverage"].iloc[0] for t in TASKS]
    cov_90 = [df_cov[(df_cov["task"] == t) & (df_cov["nominal_level"] == 0.9)]["empirical_coverage"].iloc[0] for t in TASKS]

    rects1 = ax.bar(tasks_arr - bar_width / 2, cov_80, bar_width, label="80% Nominal Interval", color="#3B82F6", edgecolor="black")
    rects2 = ax.bar(tasks_arr + bar_width / 2, cov_90, bar_width, label="90% Nominal Interval", color="#10B981", edgecolor="black")

    ax.axhline(0.80, color="#3B82F6", linestyle="--", linewidth=1.2, alpha=0.8, label="80% Target Line")
    ax.axhline(0.90, color="#10B981", linestyle="--", linewidth=1.2, alpha=0.8, label="90% Target Line")

    ax.set_title("Empirical Prediction Interval Coverage on Test Split", fontsize=12, fontweight="bold")
    ax.set_xlabel("Regression Task", fontsize=11)
    ax.set_ylabel("Empirical Coverage Fraction", fontsize=11)
    ax.set_xticks(tasks_arr)
    ax.set_xticklabels([t.replace("_", " ").title() for t in TASKS], fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate bar heights
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(f"{height:.2%}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_cov_path = reports_figures_dir / "risk_interval_coverage.png"
    fig.savefig(fig_cov_path, dpi=200)
    plt.close(fig)
    print(f"  Saved figure -> {fig_cov_path}")

    # Figure 3: F1 Score comparison by Budget Percentile
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        t_sub = df_class[df_class["task"] == task]
        t_thresh = t_sub[t_sub["rule"] == "threshold"].sort_values("budget_percentile")
        t_prob = t_sub[t_sub["rule"] == "probabilistic"].sort_values("budget_percentile")

        ax.plot(t_thresh["budget_percentile"], t_thresh["f1"], marker="o", linewidth=2, color="#EF4444", label="Threshold Rule")
        ax.plot(t_prob["budget_percentile"], t_prob["f1"], marker="s", linewidth=2, color="#2563EB", label="Probabilistic (p>=0.5)")

        ax.set_title(f"{task.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Budget Percentile (% of Training Target)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_xticks(list(BUDGET_PERCENTILES))
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Budget-Overrun Alert F1 Score Across Budget Grid", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_f1_path = reports_figures_dir / "risk_f1_by_budget.png"
    fig.savefig(fig_f1_path, dpi=200)
    plt.close(fig)
    print(f"  Saved figure -> {fig_f1_path}")

    print("=" * 80)
    print("Risk evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
