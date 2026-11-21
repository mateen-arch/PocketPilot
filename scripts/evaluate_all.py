"""
Evaluation and diagnostic reporting script for PocketPilot regression models.

Loads saved regression models and held-out test data to generate:
1. Model comparison metric tables (reports/metrics/model_comparison_{task}.csv)
2. Subgroup residual breakdown tables (reports/metrics/residuals_{task}_by_{group}.csv)
3. Diagnostic figures (reports/figures/):
   - Predicted vs. Actual scatter plots
   - Residual error distribution histograms
   - Candidate model Test MAE comparison bar charts
"""

import json
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
    DEVICE_OS_MAP,
    PROCESSED_TEST_PATH,
    TARGET_BATTERY,
    TARGET_DATA,
    TARGET_SCREEN,
)
from pocketpilot.evaluation.regression_metrics import regression_metrics, residual_table

TASKS = ["battery", "data", "screen_time"]
TASK_CONFIG = {
    "battery": {
        "target_col": TARGET_BATTERY,
        "target_label": "Daily Battery Drain",
        "units": "mAh",
        "raw_inputs": ["device_model", "app_usage_min", "screen_on_hours", "apps_installed"],
        "selected_model_file": "battery_model.joblib",
    },
    "data": {
        "target_col": TARGET_DATA,
        "target_label": "Daily Mobile Data Usage",
        "units": "MB",
        "raw_inputs": ["device_model", "app_usage_min", "screen_on_hours", "apps_installed"],
        "selected_model_file": "data_usage_model.joblib",
    },
    "screen_time": {
        "target_col": TARGET_SCREEN,
        "target_label": "Daily Screen-On Time",
        "units": "hours",
        "raw_inputs": ["device_model", "app_usage_min", "apps_installed"],
        "selected_model_file": "screen_time_model.joblib",
    },
}

CANDIDATES = ["linear", "random_forest", "xgboost", "mlp", "mean_baseline"]


def main() -> None:
    print("=" * 80)
    print("PocketPilot: Evaluating Regression Models on Test Split")
    print("=" * 80)

    models_dir = PROJECT_ROOT / "models"
    candidates_dir = models_dir / "candidates"
    reports_metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    reports_figures_dir = PROJECT_ROOT / "reports" / "figures"

    reports_metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_figures_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    # Ensure OS is present for residual subgroup analysis
    test_df["os"] = test_df["device_model"].map(DEVICE_OS_MAP)

    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for task in TASKS:
        cfg = TASK_CONFIG[task]
        target_col = cfg["target_col"]
        target_label = cfg["target_label"]
        units = cfg["units"]
        raw_inputs = cfg["raw_inputs"]
        selected_model_name = metadata[task]["selected_model"]

        print(f"\n--- Evaluating Task: {task.upper()} (Selected: {selected_model_name}) ---")

        X_test = test_df[raw_inputs]
        y_test = test_df[target_col]

        # 1. Model comparison table across all candidate models
        comparison_rows = []
        for cand in CANDIDATES:
            cand_path = candidates_dir / f"{task}_{cand}.joblib"
            if not cand_path.exists():
                raise FileNotFoundError(f"Candidate model not found at {cand_path}")

            model_pipe = joblib.load(cand_path)
            y_pred = model_pipe.predict(X_test)
            metrics = regression_metrics(y_test, y_pred)

            comparison_rows.append({
                "task": task,
                "model": cand,
                "test_mae": metrics["mae"],
                "test_rmse": metrics["rmse"],
                "test_r2": metrics["r2"],
                "is_selected": (cand == selected_model_name),
            })

        comp_df = pd.DataFrame(comparison_rows)
        comp_csv_path = reports_metrics_dir / f"model_comparison_{task}.csv"
        comp_df.to_csv(comp_csv_path, index=False)
        print(f"Saved candidate comparison metrics -> {comp_csv_path}")

        # 2. Residual analysis for the SELECTED model
        selected_pipe = joblib.load(models_dir / cfg["selected_model_file"])
        y_selected_pred = selected_pipe.predict(X_test)

        groups_to_analyze = [
            ("device_model", "device_model"),
            ("os", "os"),
            ("behavior_class", "behavior_class"),
        ]

        for group_name, group_col in groups_to_analyze:
            res_table = residual_table(test_df, y_test, y_selected_pred, group_col=group_col)
            res_csv_path = reports_metrics_dir / f"residuals_{task}_by_{group_name}.csv"
            res_table.to_csv(res_csv_path, index=False)
            print(f"Saved residual analysis by {group_name} -> {res_csv_path}")

        # 3. Generate diagnostic figures
        residuals = y_test - y_selected_pred

        # 3a. Predicted vs Actual scatter plot
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, y_selected_pred, color="#2563EB", alpha=0.7, edgecolors="none", s=40)
        min_val = min(y_test.min(), y_selected_pred.min())
        max_val = max(y_test.max(), y_selected_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect 45° Line")
        ax.set_title(f"{target_label} ({task}): Predicted vs. Actual\nModel: {selected_model_name.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Actual {target_label} ({units})", fontsize=11)
        ax.set_ylabel(f"Predicted {target_label} ({units})", fontsize=11)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        pred_plot_path = reports_figures_dir / f"predicted_vs_actual_{task}.png"
        fig.savefig(pred_plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure -> {pred_plot_path}")

        # 3b. Residual distribution histogram
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residuals, bins=20, color="#3B82F6", edgecolor="black", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line")
        ax.set_title(f"{target_label} ({task}): Residual Distribution (Actual - Predicted)\nModel: {selected_model_name.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Residual Error ({units})", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        hist_plot_path = reports_figures_dir / f"residual_hist_{task}.png"
        fig.savefig(hist_plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure -> {hist_plot_path}")

        # 3c. Candidate model Test MAE comparison bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#10B981" if c == selected_model_name else "#94A3B8" for c in comp_df["model"]]
        bars = ax.bar(comp_df["model"], comp_df["test_mae"], color=colors, edgecolor="black")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{target_label} ({task}): Test MAE by Candidate Model\n(Green = Selected Model)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Candidate Model", fontsize=11)
        ax.set_ylabel(f"Test MAE ({units})", fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        bar_plot_path = reports_figures_dir / f"model_comparison_mae_{task}.png"
        fig.savefig(bar_plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure -> {bar_plot_path}")

    print("\n" + "=" * 80)
    print("Model evaluation reports and figures generated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
