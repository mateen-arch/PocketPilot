"""
Generate global and local SHAP explanations across all regression tasks.

Computes:
1. Mean absolute SHAP importance per transformed and grouped feature on the test split.
   Saved to reports/metrics/shap_importance_{task}.csv.
2. SHAP rank stability (Spearman rank correlation between train and test splits).
   Saved to reports/metrics/shap_rank_stability.csv.
3. Publication figures in reports/figures/:
   - shap_summary_{task}.png (beeswarm plot)
   - shap_bar_{task}.png (grouped mean absolute SHAP bar chart)
   - shap_local_{task}.png (three-panel local explanation for light, typical, and heavy personas)
"""

import sys
from pathlib import Path

# Add project src directory to sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import shap

from pocketpilot import config
from pocketpilot.explain.shap_utils import (
    compute_shap,
    explain_one,
    group_device_features,
    mean_abs_importance,
)
from pocketpilot.models.trainer import TASK_RAW_INPUTS_MAP

METRICS_DIR = config.PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = config.PROJECT_ROOT / "reports" / "figures"

TASK_UNITS = {
    "battery": "mAh",
    "data": "MB",
    "screen_time": "hours",
}

PERSONAS = {
    "Light": {
        "device_model": "Google Pixel 5",
        "app_usage_min": 100,
        "screen_on_hours": 2.5,
        "apps_installed": 30,
    },
    "Typical": {
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 250,
        "screen_on_hours": 5.0,
        "apps_installed": 50,
    },
    "Heavy": {
        "device_model": "iPhone 12",
        "app_usage_min": 480,
        "screen_on_hours": 9.0,
        "apps_installed": 85,
    },
}


def run_explain_all() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RUNNING SHAP EXPLANATIONS & STABILITY ANALYSIS (F7)")
    print("=" * 70)

    train_df = pd.read_csv(config.PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(config.PROCESSED_TEST_PATH)

    stability_rows = []

    for task, model_path in config.MODEL_PATHS.items():
        print(f"\n--- Processing Task: {task.upper()} ---")
        pipeline = joblib.load(model_path)
        model_name = type(pipeline.named_steps["model"]).__name__
        raw_cols = TASK_RAW_INPUTS_MAP[task]
        unit = TASK_UNITS[task]

        X_train = train_df[raw_cols]
        X_test = test_df[raw_cols]

        # 1. Compute SHAP on test split
        print(f"Computing test SHAP ({len(X_test)} samples)...")
        shap_test_df, base_test, X_trans_test = compute_shap(pipeline, X_test)

        # Transformed & Grouped feature importance
        trans_imp = mean_abs_importance(shap_test_df, grouped=False)
        grouped_imp = mean_abs_importance(shap_test_df, grouped=True)

        # Save importance table
        imp_records = []
        for feat, val in trans_imp.items():
            imp_records.append({"feature": feat, "mean_abs_shap": val, "feature_type": "transformed"})
        for feat, val in grouped_imp.items():
            imp_records.append({"feature": feat, "mean_abs_shap": val, "feature_type": "grouped"})
        imp_df = pd.DataFrame(imp_records)
        imp_path = METRICS_DIR / f"shap_importance_{task}.csv"
        imp_df.to_csv(imp_path, index=False)
        print(f"Saved importance metrics to: {imp_path.name}")

        # 2. Compute SHAP on train split for stability check
        print(f"Computing train SHAP ({len(X_train)} samples)...")
        shap_train_df, base_train, _ = compute_shap(pipeline, X_train)
        train_grouped_imp = mean_abs_importance(shap_train_df, grouped=True)

        # Spearman rank correlation on grouped features
        common_feats = list(train_grouped_imp.index)
        train_ranks = [train_grouped_imp[f] for f in common_feats]
        test_ranks = [grouped_imp[f] for f in common_feats]
        rho, _ = scipy.stats.spearmanr(train_ranks, test_ranks)

        top3_train = ", ".join(list(train_grouped_imp.index[:3]))
        top3_test = ", ".join(list(grouped_imp.index[:3]))

        # With 3 to 5 features a rank correlation is a weak check, and this check compares one fitted model on two data splits, not refits.
        stability_rows.append({
            "task": task,
            "model": model_name,
            "n_features": len(common_feats),
            "spearman_rho": float(rho),
            "top3_train": top3_train,
            "top3_test": top3_test,
        })

        # 3. Figure: Beeswarm Summary Plot
        print("Generating beeswarm summary plot...")
        plt.figure(figsize=(9, 5))
        shap.summary_plot(shap_test_df.values, X_trans_test, show=False)
        plt.title(f"SHAP Summary (Beeswarm) - {task.title()} ({model_name})", fontsize=12, pad=10)
        summary_fig_path = FIGURES_DIR / f"shap_summary_{task}.png"
        plt.tight_layout()
        plt.savefig(summary_fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        # 4. Figure: Grouped Mean Absolute SHAP Bar Chart
        print("Generating grouped importance bar chart...")
        plt.figure(figsize=(8, 4.5))
        sorted_imp = grouped_imp.iloc[::-1]  # ascending for barh so largest is at top
        bars = plt.barh(sorted_imp.index, sorted_imp.values, color="#2b5c8f", edgecolor="#1a365d")
        for bar in bars:
            width = bar.get_width()
            plt.text(width * 1.01, bar.get_y() + bar.get_height() / 2, f"{width:.1f} {unit}",
                     va="center", ha="left", fontsize=9)
        plt.title(f"Grouped Feature Importance - {task.title()} ({model_name})", fontsize=12)
        plt.xlabel(f"Mean |SHAP Value| ({unit})", fontsize=10)
        plt.ylabel("Feature", fontsize=10)
        plt.xlim(0, sorted_imp.max() * 1.25)
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        bar_fig_path = FIGURES_DIR / f"shap_bar_{task}.png"
        plt.tight_layout()
        plt.savefig(bar_fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        # 5. Figure: Three-Panel Local Explanation for Personas
        print("Generating three-panel local explanation...")
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        for ax, (persona_name, p_data) in zip(axes, PERSONAS.items()):
            row_dict = {col: p_data[col] for col in raw_cols}
            row_df = pd.DataFrame([row_dict])
            exp_res = explain_one(pipeline, row_df, top_k=5)

            contribs = exp_res["contributions"]
            # Reverse order for horizontal bar chart (largest absolute contribution at top)
            plot_contribs = contribs[::-1]
            labels = [f"{c[0]}\n({c[1]})" for c in plot_contribs]
            values = [c[2] for c in plot_contribs]
            colors = ["#2e7d32" if v >= 0 else "#c62828" for v in values]

            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, color=colors, edgecolor="black", height=0.6)
            ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel(f"SHAP Contribution ({unit})", fontsize=9)
            ax.set_title(
                f"{persona_name} User\nPred: {exp_res['prediction']:.1f} {unit} (Base: {exp_res['base_value']:.1f})",
                fontsize=10,
                fontweight="bold"
            )
            ax.grid(axis="x", linestyle=":", alpha=0.6)

        fig.suptitle(
            f"Local Feature Attributions Across User Personas - {task.title()} ({model_name})",
            fontsize=13,
            fontweight="bold",
            y=1.02
        )
        local_fig_path = FIGURES_DIR / f"shap_local_{task}.png"
        plt.tight_layout()
        plt.savefig(local_fig_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Save stability table
    stability_df = pd.DataFrame(stability_rows)
    stab_path = METRICS_DIR / "shap_rank_stability.csv"
    stability_df.to_csv(stab_path, index=False)
    print(f"\nSaved rank stability metrics to: {stab_path.name}")

    # Print summary tables to console
    print("\n" + "=" * 70)
    print("SHAP GROUPED IMPORTANCE SUMMARY (TEST SPLIT)")
    print("=" * 70)
    for task in config.MODEL_PATHS:
        imp_path = METRICS_DIR / f"shap_importance_{task}.csv"
        df_task = pd.read_csv(imp_path)
        grouped_sub = df_task[df_task["feature_type"] == "grouped"]
        print(f"\nTask: {task.upper()} ({TASK_UNITS[task]}):")
        print(grouped_sub[["feature", "mean_abs_shap"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("SHAP RANK STABILITY (TRAIN VS TEST SPLIT)")
    print("=" * 70)
    print(stability_df[["task", "model", "n_features", "spearman_rho", "top3_train", "top3_test"]].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    run_explain_all()
