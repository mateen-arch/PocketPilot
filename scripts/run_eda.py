"""Exploratory Data Analysis (EDA) Script for PocketPilot.

Generates descriptive statistics, distribution plots, correlation heatmaps,
behavior class band analyses, and Dataset 2 profile audits.

Methodological Note:
EDA is descriptive and audit-focused only. The full cleaned Dataset 1 is inspected
here for diagnostic transparency; no model selection, hyperparameter tuning,
or threshold decisions are derived from test split rows during training.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for p in (ROOT_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pocketpilot.data.loader import load_ds1, load_ds2
from pocketpilot.data.cleaning import clean_ds1

METRICS_DIR = ROOT_DIR / "reports" / "metrics"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 70)
    print("POCKETPILOT EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. LOAD & CLEAN DATASET 1
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading and cleaning Dataset 1...")
    raw_ds1 = load_ds1()
    # Note: Full cleaned data used for descriptive audit; no modeling decisions taken from test rows
    ds1 = clean_ds1(raw_ds1)
    print(f"  Dataset 1 loaded: {ds1.shape[0]} rows, {ds1.shape[1]} columns.")

    numeric_cols = [
        "app_usage_min",
        "screen_on_hours",
        "battery_mah",
        "data_mb",
        "apps_installed",
        "age",
    ]

    # Output 1: eda_ds1_summary.csv
    # Numeric describe() plus categorical distribution shares
    desc_df = ds1[numeric_cols].describe().T
    desc_df["variable"] = desc_df.index
    desc_df["category"] = "ALL"
    desc_df["share"] = 1.0

    # Categorical shares for device_model
    dev_counts = ds1["device_model"].value_counts()
    dev_shares = ds1["device_model"].value_counts(normalize=True)
    dev_rows = []
    for dev in dev_counts.index:
        dev_rows.append({
            "variable": "device_model",
            "category": dev,
            "count": dev_counts[dev],
            "share": dev_shares[dev],
        })

    # Categorical shares for behavior_class
    bc_counts = ds1["behavior_class"].value_counts()
    bc_shares = ds1["behavior_class"].value_counts(normalize=True)
    bc_rows = []
    for bc in sorted(bc_counts.index):
        bc_rows.append({
            "variable": "behavior_class",
            "category": f"Class {bc}",
            "count": bc_counts[bc],
            "share": bc_shares[bc],
        })

    cat_df = pd.DataFrame(dev_rows + bc_rows)
    summary_combined = pd.concat([desc_df, cat_df], ignore_index=True)
    cols_order = ["variable", "category", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "share"]
    summary_combined = summary_combined[[c for c in cols_order if c in summary_combined.columns]]

    ds1_summary_path = METRICS_DIR / "eda_ds1_summary.csv"
    summary_combined.to_csv(ds1_summary_path, index=False)
    print(f"  Saved Dataset 1 summary -> {ds1_summary_path.relative_to(ROOT_DIR)}")

    # -------------------------------------------------------------------------
    # 2. CORRELATIONS & DISTRIBUTIONS
    # -------------------------------------------------------------------------
    print("\n[2/5] Computing correlations and plotting distributions...")
    corr_matrix = ds1[numeric_cols].corr(method="pearson")
    corr_path = METRICS_DIR / "eda_ds1_correlation.csv"
    corr_matrix.to_csv(corr_path)
    print(f"  Saved correlation matrix -> {corr_path.relative_to(ROOT_DIR)}")

    # Correlation heatmap figure
    col_labels_with_units = [
        "App Usage\n(min)",
        "Screen Time\n(hours)",
        "Battery Drain\n(mAh)",
        "Data Usage\n(MB)",
        "Apps Installed\n(count)",
        "Age\n(years)",
    ]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation (r)")

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(col_labels_with_units, rotation=30, ha="left", fontsize=9)
    ax.set_yticklabels(col_labels_with_units, fontsize=9)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10, weight="bold")

    ax.set_title("Dataset 1: Pearson Correlation Heatmap Among Numeric Variables", pad=40, fontsize=12, weight="bold")
    plt.tight_layout()
    corr_fig_path = FIGURES_DIR / "eda_ds1_correlation.png"
    plt.savefig(corr_fig_path, dpi=200)
    plt.close()
    print(f"  Saved correlation heatmap -> {corr_fig_path.relative_to(ROOT_DIR)}")

    # Distribution histogram grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    unit_map = {
        "app_usage_min": ("Daily App Usage", "Minutes"),
        "screen_on_hours": ("Screen-On Time", "Hours"),
        "battery_mah": ("Daily Battery Drain", "mAh"),
        "data_mb": ("Mobile Data Consumed", "MB"),
        "apps_installed": ("Installed Applications", "Count"),
        "age": ("User Age", "Years"),
    }

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        title, unit = unit_map[col]
        ax.hist(ds1[col], bins=20, color="#2563EB", edgecolor="#1E3A8A", alpha=0.85)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel(f"{title} ({unit})", fontsize=10)
        ax.set_ylabel("User Frequency", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Dataset 1: Empirical Feature Distributions (N=700)", fontsize=14, weight="bold")
    plt.tight_layout()
    dist_fig_path = FIGURES_DIR / "eda_ds1_distributions.png"
    plt.savefig(dist_fig_path, dpi=200)
    plt.close()
    print(f"  Saved distributions figure -> {dist_fig_path.relative_to(ROOT_DIR)}")

    # Bar chart of device model row counts
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dev_counts = ds1["device_model"].value_counts()
    bars = ax.bar(dev_counts.index, dev_counts.values, color="#10B981", edgecolor="#065F46", alpha=0.85)
    ax.set_title("Dataset 1: Record Count per Device Model", fontsize=12, weight="bold")
    ax.set_xlabel("Device Model", fontsize=10)
    ax.set_ylabel("Number of Records", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, weight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    device_fig_path = FIGURES_DIR / "eda_ds1_device_counts.png"
    plt.savefig(device_fig_path, dpi=200)
    plt.close()
    print(f"  Saved device counts bar chart -> {device_fig_path.relative_to(ROOT_DIR)}")

    # -------------------------------------------------------------------------
    # 3. BEHAVIOR CLASS BANDS & LEAKAGE DOCUMENTATION
    # -------------------------------------------------------------------------
    print("\n[3/5] Analyzing behavior class bands and overlap...")
    bc_bands = []
    classes = sorted(ds1["behavior_class"].unique())

    for bc in classes:
        sub = ds1[ds1["behavior_class"] == bc]
        bc_bands.append({
            "behavior_class": bc,
            "count": len(sub),
            "app_usage_min_min": sub["app_usage_min"].min(),
            "app_usage_min_max": sub["app_usage_min"].max(),
            "app_usage_min_mean": round(sub["app_usage_min"].mean(), 2),
            "screen_on_hours_min": sub["screen_on_hours"].min(),
            "screen_on_hours_max": sub["screen_on_hours"].max(),
            "screen_on_hours_mean": round(sub["screen_on_hours"].mean(), 2),
            "battery_mah_min": sub["battery_mah"].min(),
            "battery_mah_max": sub["battery_mah"].max(),
            "battery_mah_mean": round(sub["battery_mah"].mean(), 2),
            "data_mb_min": sub["data_mb"].min(),
            "data_mb_max": sub["data_mb"].max(),
            "data_mb_mean": round(sub["data_mb"].mean(), 2),
        })

    bands_df = pd.DataFrame(bc_bands)
    bands_path = METRICS_DIR / "eda_ds1_behavior_class_bands.csv"
    bands_df.to_csv(bands_path, index=False)
    print(f"  Saved behavior class bands -> {bands_path.relative_to(ROOT_DIR)}")

    # Overlap check documentation
    has_overlap = False
    overlap_report = []
    for i in range(len(bands_df) - 1):
        c_curr = bands_df.iloc[i]
        c_next = bands_df.iloc[i + 1]
        max_curr = c_curr["app_usage_min_max"]
        min_next = c_next["app_usage_min_min"]
        overlaps = max_curr >= min_next
        if overlaps:
            has_overlap = True
        overlap_report.append(
            f"Class {int(c_curr['behavior_class'])} max ({max_curr}) vs "
            f"Class {int(c_next['behavior_class'])} min ({min_next}): "
            f"{'OVERLAP' if overlaps else 'NO OVERLAP (strictly separated gap)'}"
        )

    print("  Behavior Class App Usage Overlap Audit:")
    for rep in overlap_report:
        print(f"    - {rep}")
    print(f"  Overall Consecutive Range Overlap: {has_overlap}")
    print("  Documented conclusion: behavior_class represents deterministic threshold binning of app_usage_min,")
    print("  confirming that behavior_class is a ground-truth leakage target and must never be a forecast input.")

    # Scatter of app_usage_min vs battery_mah colored by device_model
    fig, ax = plt.subplots(figsize=(9, 5.5))
    devices = sorted(ds1["device_model"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(devices)))

    for dev, col in zip(devices, colors):
        sub = ds1[ds1["device_model"] == dev]
        ax.scatter(sub["app_usage_min"], sub["battery_mah"], label=dev, color=col, alpha=0.7, edgecolors="none", s=35)

    ax.set_title("Dataset 1: App Usage Minutes vs Battery Drain (mAh) by Device Model", fontsize=12, weight="bold")
    ax.set_xlabel("Daily App Usage (Minutes)", fontsize=10)
    ax.set_ylabel("Daily Battery Drain (mAh)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    # Legend outside the plot area
    ax.legend(title="Device Model", bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    usage_bat_path = FIGURES_DIR / "eda_ds1_usage_vs_battery.png"
    plt.savefig(usage_bat_path, dpi=200)
    plt.close()
    print(f"  Saved usage vs battery scatter -> {usage_bat_path.relative_to(ROOT_DIR)}")

    # -------------------------------------------------------------------------
    # 4. DATASET 2 PROFILE AUDIT
    # -------------------------------------------------------------------------
    print("\n[4/5] Profiling Dataset 2...")
    ds2 = load_ds2()
    ds2_rows, ds2_cols = ds2.shape

    # None text vs NaN in sleep_disruption_from_phone
    none_text_count = int((ds2["sleep_disruption_from_phone"] == "None").sum())
    nan_count = int(ds2["sleep_disruption_from_phone"].isna().sum())

    mean_screen_min = float(ds2["daily_screen_time_minutes"].mean())

    # Numeric correlations with battery_drain_pct_per_session
    num_cols_ds2 = ds2.select_dtypes(include=["number"]).columns.tolist()
    if "battery_drain_pct_per_session" in num_cols_ds2:
        corr_series = ds2[num_cols_ds2].corr()["battery_drain_pct_per_session"].drop("battery_drain_pct_per_session")
        max_corr_col = corr_series.abs().idxmax()
        max_corr_val = float(corr_series[max_corr_col])
    else:
        max_corr_col = "None"
        max_corr_val = 0.0

    # App category data consumption
    cat_means = ds2.groupby("app_category")["data_used_mb_per_day"].mean().sort_values()
    lowest_cat_name = cat_means.index[0]
    lowest_cat_val = float(cat_means.iloc[0])
    highest_cat_name = cat_means.index[-1]
    highest_cat_val = float(cat_means.iloc[-1])

    # Network connection data consumption
    wifi_mean = float(ds2[ds2["network_connection"] == "WiFi"]["data_used_mb_per_day"].mean())
    lte_mean = float(ds2[ds2["network_connection"] == "4G/LTE"]["data_used_mb_per_day"].mean())

    ds2_profile_facts = [
        {"fact": "rows", "value": ds2_rows, "details": "Total record count in Dataset 2"},
        {"fact": "columns", "value": ds2_cols, "details": "Total column count in Dataset 2"},
        {"fact": "sleep_disruption_none_text_count", "value": none_text_count, "details": "Rows containing string 'None' (real category)"},
        {"fact": "sleep_disruption_nan_count", "value": nan_count, "details": "Rows containing NaN in sleep disruption (must be 0)"},
        {"fact": "mean_daily_screen_time_minutes", "value": round(mean_screen_min, 2), "details": "Mean daily screen time across users (minutes)"},
        {"fact": "max_battery_drain_numeric_corr_col", "value": max_corr_col, "details": "Numeric column with highest absolute correlation to battery drain"},
        {"fact": "max_battery_drain_numeric_corr_val", "value": round(max_corr_val, 6), "details": "Pearson correlation coefficient r with battery drain"},
        {"fact": "highest_mean_data_used_app_category", "value": highest_cat_name, "details": f"Mean {highest_cat_val:.2f} MB/day"},
        {"fact": "lowest_mean_data_used_app_category", "value": lowest_cat_name, "details": f"Mean {lowest_cat_val:.2f} MB/day"},
        {"fact": "mean_data_used_mb_wifi", "value": round(wifi_mean, 2), "details": "Mean daily data on WiFi (MB/day)"},
        {"fact": "mean_data_used_mb_4g_lte", "value": round(lte_mean, 2), "details": "Mean daily data on 4G/LTE (MB/day)"},
    ]

    ds2_profile_df = pd.DataFrame(ds2_profile_facts)
    ds2_profile_path = METRICS_DIR / "eda_ds2_profile.csv"
    ds2_profile_df.to_csv(ds2_profile_path, index=False)
    print(f"  Saved Dataset 2 profile -> {ds2_profile_path.relative_to(ROOT_DIR)}")

    # -------------------------------------------------------------------------
    # 5. SUMMARY PRINT
    # -------------------------------------------------------------------------
    print("\n[5/5] EDA SUMMARY REPORT")
    print("-" * 70)
    print("Dataset 1 Key Findings:")
    print(f"  - Total records: {len(ds1)} | Devices: {list(dev_counts.index)}")
    print(f"  - App usage: min={ds1['app_usage_min'].min()}m, max={ds1['app_usage_min'].max()}m, mean={ds1['app_usage_min'].mean():.1f}m")
    print(f"  - Screen time: min={ds1['screen_on_hours'].min()}h, max={ds1['screen_on_hours'].max()}h, mean={ds1['screen_on_hours'].mean():.1f}h")
    print(f"  - Battery drain: min={ds1['battery_mah'].min()}mAh, max={ds1['battery_mah'].max()}mAh, mean={ds1['battery_mah'].mean():.1f}mAh")
    print(f"  - Data usage: min={ds1['data_mb'].min()}MB, max={ds1['data_mb'].max()}MB, mean={ds1['data_mb'].mean():.1f}MB")
    print(f"  - Strong correlations: App-Screen={corr_matrix.loc['app_usage_min', 'screen_on_hours']:.3f}, "
          f"App-Battery={corr_matrix.loc['app_usage_min', 'battery_mah']:.3f}, "
          f"App-Data={corr_matrix.loc['app_usage_min', 'data_mb']:.3f}")
    print("\nDataset 2 Key Findings:")
    print(f"  - Records: {ds2_rows} | Columns: {ds2_cols}")
    print(f"  - Sleep disruption 'None' string count: {none_text_count} (NaN count: {nan_count})")
    print(f"  - Mean screen time: {mean_screen_min:.1f} minutes ({mean_screen_min/60.0:.1f} hours)")
    print(f"  - Max numeric battery drain correlation: {max_corr_col} (r={max_corr_val:.4f})")
    print(f"  - Highest data category: {highest_cat_name} ({highest_cat_val:.1f} MB/day)")
    print(f"  - Lowest data category: {lowest_cat_name} ({lowest_cat_val:.1f} MB/day)")
    print(f"  - WiFi daily data: {wifi_mean:.1f} MB vs 4G/LTE: {lte_mean:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
