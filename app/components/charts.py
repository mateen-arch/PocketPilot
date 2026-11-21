"""
Chart components for SHAP local attributions, profile history trends, and planner comparison tables.
"""

from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_shap_waterfall_bar(
    contributions: list[Any],
    prediction: float,
    base_value: float,
    task_name: str,
    unit: str,
    key_prefix: str = "",
) -> None:
    """
    Render local feature contributions as an interactive horizontal bar chart.

    Parameters
    ----------
    contributions : list[Any]
        List of Contribution(feature, value, shap_value) from explain_one().
    prediction : float
        Point prediction for the sample.
    base_value : float
        Baseline explainer expected value.
    task_name : str
        Resource task title (e.g. 'Battery Drain').
    unit : str
        Target unit (e.g. 'mAh', 'MB', 'hours').
    key_prefix : str, optional
        Unique key prefix.
    """
    if not contributions:
        st.info("No feature contributions available for explanation.")
        return

    # Invert so largest absolute contribution is at top
    rev_contribs = contributions[::-1]
    features = [f"{c[0]} ({c[1]})" if c[1] is not None else str(c[0]) for c in rev_contribs]
    shap_vals = [float(c[2]) for c in rev_contribs]
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in shap_vals]

    fig = go.Figure(
        go.Bar(
            x=shap_vals,
            y=features,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.1f} {unit}" for v in shap_vals],
            textposition="auto",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#374151")
    fig.update_layout(
        title=f"<b>{task_name} Local Feature Attributions</b><br><sup>Base: {base_value:.1f} {unit} → Predicted: {prediction:.1f} {unit}</sup>",
        xaxis_title=f"SHAP Impact ({unit})",
        yaxis_title="Feature (Entered Value)",
        height=320,
        margin=dict(l=10, r=20, t=50, b=30),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{task_name}_shap_bar")


def render_history_trend_chart(
    history_df: pd.DataFrame,
    pred_history: pd.DataFrame,
    key_prefix: str = "",
) -> None:
    """
    Render a multi-day timeline of usage and persona classifications.

    Parameters
    ----------
    history_df : pd.DataFrame
        Cleaned multi-day history DataFrame.
    pred_history : pd.DataFrame
        Predicted labels and cluster IDs aligned with history_df index.
    key_prefix : str, optional
        Unique key prefix.
    """
    df = history_df.copy()
    df["Profile Persona"] = pred_history["profile_label"].values

    x_col = "date" if "date" in df.columns else df.index.name or "Day Index"
    if x_col not in df.columns:
        df[x_col] = range(1, len(df) + 1)

    fig = px.scatter(
        df,
        x=x_col,
        y="app_usage_min",
        size="battery_mah",
        color="Profile Persona",
        hover_data=["screen_on_hours", "data_mb", "battery_mah"],
        title="<b>Daily Usage Intensity & Persona Classification Over Time</b>",
        labels={"app_usage_min": "App Usage (min)", "battery_mah": "Battery (mAh)"},
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_history_trend")


def render_alternatives_table(
    alternatives_df: pd.DataFrame,
    battery_budget: float,
    battery_risk_bundle: dict[str, Any] | None = None,
) -> None:
    """
    Render a formatted comparison table of alternative behavior plans.

    Parameters
    ----------
    alternatives_df : pd.DataFrame
        Alternatives DataFrame from recommend().
    battery_budget : float
        Battery budget against which alternatives are evaluated.
    battery_risk_bundle : dict[str, Any], optional
        Fitted battery risk bundle for re-evaluating overrun probability.
    """
    if alternatives_df is None or len(alternatives_df) == 0:
        st.info("No alternative plans generated.")
        return

    from pocketpilot.inference.formatting import format_probability
    from pocketpilot.risk.probabilistic_risk import overrun_probability

    display_rows = []
    for rank, (_, row) in enumerate(alternatives_df.iterrows(), 1):
        # Format levers
        levers = []
        if row["app_cut_min"] > 0:
            levers.append(f"Cut {int(row['app_cut_min'])} min apps")
        if row["screen_cut_hours"] > 0:
            levers.append(f"Cut {row['screen_cut_hours']:.1f} h screen")
        if row["wifi_share"] > 0:
            levers.append(f"Shift {int(row['wifi_share']*100)}% Wi-Fi (assumption)")
        if row["power_saving"]:
            levers.append("Power Saving (assumption)")
        lever_str = ", ".join(levers) if levers else "No change"

        # Compute P(make it) if bundle available
        p_make_it_str = "N/A"
        if battery_risk_bundle is not None:
            p_ov = float(overrun_probability(battery_risk_bundle, float(row["battery_pred"]), battery_budget))
            p_make_it_str = format_probability(1.0 - p_ov)

        display_rows.append({
            "Rank": f"Option {rank}",
            "Action Cost": f"{row['action_cost']:.1f}",
            "Recommended Levers": lever_str,
            "Battery (mAh)": f"{row['battery_pred']:.1f}",
            "Meets Goal": "Yes" if row["is_feasible"] else "No",
            "P(make it)": p_make_it_str,
        })

    st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)
