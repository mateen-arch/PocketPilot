"""Page 4: Model Evaluation and Transparency.

Read-only dashboard displaying transparent evaluation evidence across all
models and pipelines in PocketPilot: regression models, explainability (SHAP),
usage clustering, probabilistic risk calibration, what-if planning, and test coverage.
"""

from pathlib import Path
import sys
import json
import pandas as pd
import streamlit as st

# Bootstrap sys.path so the page can run both via router and standalone
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
for p in (ROOT_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.title("Model Evaluation & Transparency")
st.caption(
    "Comprehensive empirical audit of models, calibration, assumptions, and validation metrics."
)


def load_csv_safe(rel_path: str) -> pd.DataFrame | None:
    """Safely load a CSV report, showing st.warning if absent."""
    p = ROOT_DIR / rel_path
    if not p.is_file():
        st.warning(f"File not found: {rel_path}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:
        st.warning(f"Error loading {rel_path}: {exc}")
        return None


def load_json_safe(rel_path: str) -> dict | None:
    """Safely load a JSON file, showing st.warning if absent."""
    p = ROOT_DIR / rel_path
    if not p.is_file():
        st.warning(f"File not found: {rel_path}")
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.warning(f"Error loading {rel_path}: {exc}")
        return None


def show_image_safe(rel_path: str, caption: str = "") -> None:
    """Safely display a figure image, showing st.warning if absent."""
    p = ROOT_DIR / rel_path
    if not p.is_file():
        st.warning(f"Figure not found: {rel_path}")
        return
    st.image(str(p), caption=caption, use_container_width=True)


def read_text_safe(rel_path: str) -> str | None:
    """Safely read a text file, showing st.warning if absent."""
    p = ROOT_DIR / rel_path
    if not p.is_file():
        st.warning(f"File not found: {rel_path}")
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        st.warning(f"Error reading {rel_path}: {exc}")
        return None


# Load global model metadata
metadata = load_json_safe("models/metadata.json") or {}

# Define tabs
tabs = st.tabs([
    "Overview and limitations",
    "Regression models",
    "Explainability",
    "Clustering",
    "Risk and calibration",
    "Planner",
    "Testing",
])

# ---------------------------------------------------------
# TAB 1: Overview and limitations
# ---------------------------------------------------------
with tabs[0]:
    st.header("Dataset Summary & Methodological Limitations")
    
    # Dataset statistics loaded from processed splits
    train_df = load_csv_safe("data/processed/ds1_train.csv")
    test_df = load_csv_safe("data/processed/ds1_test.csv")
    
    if train_df is not None and test_df is not None:
        total_rows = len(train_df) + len(test_df)
        col_count = len(train_df.columns)
        devices = sorted(train_df["device_model"].unique().tolist())
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{total_rows}")
        c2.metric("Train / Test Split", f"{len(train_df)} / {len(test_df)} (80/20)")
        c3.metric("Dataset Columns", f"{col_count}")
        c4.metric("Device Models", f"{len(devices)}")
        
        st.write(f"**Supported Devices:** {', '.join(devices)}")
    
    st.subheader("Simulated-Data Limitations")
    st.warning(
        "**Core Limitation:** All training and evaluation data is derived from synthetic mobile behavior records "
        "(Dataset 1: 700 records total). Key behavioral metrics—app usage minutes, screen-on hours, battery drain (mAh), "
        "and mobile data consumption (MB)—exhibit pairwise linear correlations between 0.93 and 0.98. "
        "These correlations reflect the linear equations embedded within the synthetic data generation script rather "
        "than diverse real-world mobile telemetry."
    )
    
    st.info(
        "**State Note on Accuracy Scores:** High benchmark scores (e.g., test R² > 0.93 across all regression tasks) "
        "reflect the rigid mathematical structure and low noise profile of the simulated dataset. "
        "They are NOT evidence of high real-world accuracy on actual device hardware with irregular background processes, "
        "varying network signals, battery degradation, or screen brightness fluctuations."
    )
    
    st.divider()
    st.markdown("### Research & Prototype Disclaimer")
    st.markdown(
        "> PocketPilot is a research and educational prototype trained on aggregated public mobile-usage records. "
        "It estimates daily behavior patterns and should not be interpreted as an operating-system-level battery monitor."
    )

# ---------------------------------------------------------
# TAB 2: Regression models
# ---------------------------------------------------------
with tabs[1]:
    st.header("Regression Model Benchmarking & Residuals")
    st.markdown(
        "Three regression tasks predict daily mobile resource demands from user inputs. "
        "Candidates include a Mean Baseline, Ordinary Least Squares Linear Regression, Random Forest, "
        "XGBoost, and Multi-Layer Perceptron (MLP)."
    )
    
    task_labels = {
        "battery": "Battery Drain (battery_mah)",
        "data": "Data Consumption (data_mb)",
        "screen_time": "Screen Time (screen_on_hours)",
    }
    
    for task_key, task_title in task_labels.items():
        st.subheader(f"Task: {task_title}")
        
        comp_df = load_csv_safe(f"reports/metrics/model_comparison_{task_key}.csv")
        selected_model = metadata.get(task_key, {}).get("selected_model", "")
        
        if comp_df is not None:
            # Mark selected model
            if "selected" not in comp_df.columns:
                comp_df["selected"] = comp_df["model"].apply(
                    lambda m: "★ Selected" if str(m).lower() == selected_model.lower() else ""
                )
                cols = ["selected"] + [c for c in comp_df.columns if c != "selected"]
                comp_df = comp_df[cols]
            st.dataframe(comp_df, use_container_width=True)
        
        col_fig1, col_fig2, col_fig3 = st.columns(3)
        with col_fig1:
            show_image_safe(
                f"reports/figures/model_comparison_mae_{task_key}.png",
                caption=f"MAE Comparison: {task_key}",
            )
        with col_fig2:
            show_image_safe(
                f"reports/figures/predicted_vs_actual_{task_key}.png",
                caption=f"Predicted vs Actual: {task_key} ({selected_model})",
            )
        with col_fig3:
            show_image_safe(
                f"reports/figures/residual_hist_{task_key}.png",
                caption=f"Residual Distribution: {task_key} ({selected_model})",
            )
        
        # Residuals by device model
        res_device = load_csv_safe(f"reports/metrics/residuals_{task_key}_by_device_model.csv")
        if res_device is not None:
            with st.expander(f"Residuals by Device Model ({task_key})"):
                st.dataframe(res_device, use_container_width=True)
        st.divider()

# ---------------------------------------------------------
# TAB 3: Explainability
# ---------------------------------------------------------
with tabs[2]:
    st.header("Explainability & SHAP Attribution Stability")
    st.markdown(
        "Global and local feature attributions computed using TreeSHAP (TreeExplainer). "
        "Attributions quantify the contribution of each input feature to the predicted resource usage."
    )
    
    t_cols = st.columns(3)
    for idx, (t_key, t_name) in enumerate(task_labels.items()):
        with t_cols[idx]:
            st.subheader(t_name)
            imp_df = load_csv_safe(f"reports/metrics/shap_importance_{t_key}.csv")
            if imp_df is not None:
                st.dataframe(imp_df, use_container_width=True)
            show_image_safe(
                f"reports/figures/shap_bar_{t_key}.png",
                caption=f"Mean |SHAP|: {t_key}",
            )
            show_image_safe(
                f"reports/figures/shap_summary_{t_key}.png",
                caption=f"SHAP Summary (Beeswarm): {t_key}",
            )
    
    st.subheader("Feature Rank Stability Check")
    stability_df = load_csv_safe("reports/metrics/shap_rank_stability.csv")
    if stability_df is not None:
        st.dataframe(stability_df, use_container_width=True)
    st.caption(
        "Methodological Note: With 3 to 5 features a rank correlation is a weak check; "
        "this evaluates one fitted model across two data splits, not refits."
    )

# ---------------------------------------------------------
# TAB 4: Clustering
# ---------------------------------------------------------
with tabs[3]:
    st.header("Digital Usage Profiling (Clustering)")
    st.markdown(
        "Unsupervised K-Means clustering fitted exclusively on the 4 observed usage columns "
        "of the training split (StandardScaler + K-Means, $k=4$, random_state=42). "
        "Clusters are labeled neutrally based on ascending usage intensity."
    )
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.subheader("k-Selection Metric Curve")
        k_df = load_csv_safe("reports/metrics/clustering_k_selection.csv")
        if k_df is not None:
            st.dataframe(k_df, use_container_width=True)
    with col_k2:
        show_image_safe(
            "reports/figures/clustering_k_selection.png",
            caption="Silhouette & Davies-Bouldin Scores Across k=2..8",
        )
    
    st.subheader("Cluster Quality on Train and Test Partitions")
    qual_df = load_csv_safe("reports/metrics/clustering_quality.csv")
    if qual_df is not None:
        st.dataframe(qual_df, use_container_width=True)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Profile Centroids (Training Split)")
        cent_train = load_csv_safe("reports/metrics/clustering_profiles_train.csv")
        if cent_train is not None:
            st.dataframe(cent_train, use_container_width=True)
    with col_p2:
        st.subheader("Profile Centroids (Test Split)")
        cent_test = load_csv_safe("reports/metrics/clustering_profiles_test.csv")
        if cent_test is not None:
            st.dataframe(cent_test, use_container_width=True)
            
    col_pca1, col_pca2 = st.columns(2)
    with col_pca1:
        show_image_safe("reports/figures/clustering_pca.png", caption="PCA 2D Cluster Projection")
    with col_pca2:
        show_image_safe("reports/figures/clustering_centroids.png", caption="Profile Feature Medians")
        
    st.subheader("Sanity Check: Clusters vs Ground Truth Behavior Class")
    bc_df = load_csv_safe("reports/metrics/clustering_vs_behavior_class.csv")
    if bc_df is not None:
        st.dataframe(bc_df, use_container_width=True)
    st.caption("Sanity check: clusters align with behavioral intensity classes from the synthetic dataset generation.")

# ---------------------------------------------------------
# TAB 5: Risk and calibration
# ---------------------------------------------------------
with tabs[4]:
    st.header("Probabilistic Risk Layer & Uncertainty Calibration")
    st.markdown(
        "Quantifies budget-overrun probabilities using empirical out-of-fold cross-validated "
        "residuals from the training partition. Residuals are sampled with Laplace smoothing "
        "to prevent uncalibrated zero/one certainty."
    )
    
    st.subheader("Classification Performance Across Budgets")
    risk_class_df = load_csv_safe("reports/metrics/risk_classification.csv")
    if risk_class_df is not None:
        if "budget_percentile" in risk_class_df.columns:
            percentiles = sorted(risk_class_df["budget_percentile"].unique().tolist())
            selected_p = st.selectbox(
                "Filter by Budget Percentile:",
                options=["All"] + [f"p{int(p*100)}" for p in percentiles],
                key="risk_p_filter",
            )
            if selected_p != "All":
                p_val = float(selected_p.replace("p", "")) / 100.0
                filtered_class = risk_class_df[risk_class_df["budget_percentile"] == p_val]
            else:
                filtered_class = risk_class_df
            st.dataframe(filtered_class, use_container_width=True)
        else:
            st.dataframe(risk_class_df, use_container_width=True)
            
    c_f1, c_cov = st.columns(2)
    with c_f1:
        show_image_safe("reports/figures/risk_f1_by_budget.png", caption="F1 Score Across Budget Levels")
    with c_cov:
        show_image_safe("reports/figures/risk_interval_coverage.png", caption="80% Prediction Interval Coverage")
        
    st.subheader("Probability Calibration & Reliability")
    cal_df = load_csv_safe("reports/metrics/risk_calibration.csv")
    if cal_df is not None:
        st.dataframe(cal_df, use_container_width=True)
        
    r_cols = st.columns(3)
    with r_cols[0]:
        show_image_safe("reports/figures/risk_reliability_battery.png", caption="Battery Calibration")
    with r_cols[1]:
        show_image_safe("reports/figures/risk_reliability_data.png", caption="Data Calibration")
    with r_cols[2]:
        show_image_safe("reports/figures/risk_reliability_screen_time.png", caption="Screen Time Calibration")
        
    st.subheader("Empirical Interval Coverage on Test Split")
    cov_df = load_csv_safe("reports/metrics/risk_interval_coverage.csv")
    if cov_df is not None:
        st.dataframe(cov_df, use_container_width=True)
        
    st.caption(
        "Note on Decision Boundaries: The two decision rules coincide near the median and "
        "the value of the probabilistic layer is calibrated probabilities."
    )

# ---------------------------------------------------------
# TAB 6: Planner
# ---------------------------------------------------------
with tabs[5]:
    st.header("What-If Action Planner Evaluation")
    st.markdown(
        "Evaluates the what-if intervention planner across simulated test users. "
        "Compares `all_levers` (learned + heuristic assumptions) against `learned_only` "
        "(app usage reduction only)."
    )
    
    st.subheader("Planner Solvability & Reliance on Assumption Levers")
    solv_df = load_csv_safe("reports/metrics/planner_solvability.csv")
    if solv_df is not None:
        st.dataframe(solv_df, use_container_width=True)
        
    st.caption(
        "Honest Methodological Finding: A large share of solved cases strictly depends on assumption levers. "
        "As documented in `share_requiring_assumption_levers`, approximately 43% to 54% of solvable cases "
        "cannot be brought into budget by app cuts alone and strictly require heuristic actions "
        "(Wi-Fi offload and/or battery saver mode)."
    )
    
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        show_image_safe("reports/figures/planner_solvability.png", caption="Planner Solvability by Mode")
    with c_p2:
        show_image_safe("reports/figures/planner_lever_usage.png", caption="Action Frequency Across Scenarios")
        
    st.subheader("What-If Test Cases (Persona Scenarios)")
    whatif_df = load_csv_safe("reports/metrics/planner_whatif_cases.csv")
    if whatif_df is not None:
        st.dataframe(whatif_df, use_container_width=True)
        
    show_image_safe("reports/figures/planner_cost_distribution.png", caption="User Disruption Cost Distribution")

# ---------------------------------------------------------
# TAB 7: Testing
# ---------------------------------------------------------
with tabs[6]:
    st.header("Automated Test Suite & Verification Inventory")
    st.markdown(
        "All pipeline modules, ML models, calibration routines, what-if planners, "
        "and UI components are verified via automated regression and boundary tests."
    )
    
    raw_inventory = read_text_safe("reports/testing_evidence/test_inventory.txt")
    if raw_inventory:
        # Parse test items and count per file
        test_lines = [
            line.strip()
            for line in raw_inventory.splitlines()
            if "::" in line and not line.startswith("=")
        ]
        
        file_counts: dict[str, int] = {}
        for line in test_lines:
            file_name = line.split("::")[0]
            file_counts[file_name] = file_counts.get(file_name, 0) + 1
            
        summary_rows = [
            {"Test File": k, "Test Cases": v} for k, v in sorted(file_counts.items())
        ]
        summary_df = pd.DataFrame(summary_rows)
        
        c_inv1, c_inv2 = st.columns([1, 2])
        with c_inv1:
            st.metric("Total Test Cases", f"{len(test_lines)}")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        with c_inv2:
            st.subheader("Full Test Case Inventory")
            with st.expander("View All Registered Test Identifiers", expanded=False):
                st.code("\n".join(test_lines), language="text")
    
    st.subheader("Recent Pytest Run Output")
    full_run = read_text_safe("reports/testing_evidence/pytest_full_run.txt")
    if full_run:
        lines = full_run.splitlines()
        tail_lines = lines[-35:] if len(lines) > 35 else lines
        st.code("\n".join(tail_lines), language="text")
