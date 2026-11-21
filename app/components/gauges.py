"""
Plotly gauge chart component with accessible, explicit text-based metric displays.
"""

from typing import Any
import plotly.graph_objects as go
import streamlit as st


def render_resource_gauge(
    title: str,
    predicted: float,
    budget: float,
    unit: str,
    risk_level: str,
    p_overrun_str: str,
    interval_80: tuple[float, float] | None = None,
    key_prefix: str = "",
) -> None:
    """
    Render an accessible resource consumption gauge chart alongside explicit text metrics.

    Never relies on visual color alone: full numbers, thresholds, risk levels, and intervals
    are always printed as readable markdown text.

    Parameters
    ----------
    title : str
        Resource title (e.g. 'Battery Drain', 'Mobile Data', 'Screen Time').
    predicted : float
        Predicted consumption value.
    budget : float
        Operational budget threshold.
    unit : str
        Measurement unit (e.g. 'mAh', 'MB', 'hours').
    risk_level : str
        Categorical risk state: 'Safe', 'Watch', or 'Likely exceed'.
    p_overrun_str : str
        Formatted overrun probability (e.g. 'under 1%', '15%', 'over 99%').
    interval_80 : tuple[float, float], optional
        80% central prediction interval (lower, upper).
    key_prefix : str, optional
        Unique key prefix for Streamlit elements.
    """
    color_map = {
        "Safe": "#10B981",          # Green
        "Watch": "#F59E0B",         # Amber
        "Likely exceed": "#EF4444", # Red
    }
    bar_color = color_map.get(risk_level, "#3B82F6")

    max_axis = max(budget * 1.4, predicted * 1.2, 1.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted,
            number={"suffix": f" {unit}", "font": {"size": 22}},
            title={"text": f"<b>{title}</b>", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, max_axis], "tickwidth": 1},
                "bar": {"color": bar_color},
                "bgcolor": "#F3F4F6",
                "borderwidth": 1,
                "bordercolor": "#D1D5DB",
                "steps": [
                    {"range": [0, budget], "color": "#E5E7EB"},
                    {"range": [budget, max_axis], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "thickness": 0.85,
                    "value": budget,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=25, r=25, t=40, b=10),
        font=dict(family="sans-serif"),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{title}_gauge")

    # Explicit text display (accessibility requirement: never rely on color alone)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Predicted:** `{predicted:.1f} {unit}`")
        st.markdown(f"**Budget:** `{budget:.1f} {unit}`")
    with col_b:
        st.markdown(f"**Risk State:** **{risk_level}**")
        st.markdown(f"**Overrun Risk:** `{p_overrun_str}`")

    if interval_80 is not None:
        low, high = interval_80
        st.caption(f"80% Prediction Interval: `{low:.1f}` to `{high:.1f} {unit}`")
