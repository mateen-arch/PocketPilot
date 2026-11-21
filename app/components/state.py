"""
Global application state and cached artifact loading for PocketPilot.
"""

from typing import Any
import streamlit as st
from pocketpilot.inference import predictor


@st.cache_resource
def _cached_artifacts() -> dict[str, Any]:
    """Cached loader for regression pipelines, clustering bundles, and risk dicts."""
    return predictor.load_artifacts()


def get_artifacts() -> dict[str, Any]:
    """
    Retrieve cached production artifacts and display startup warnings once.

    Returns
    -------
    dict[str, Any]
        Loaded system artifacts dictionary.
    """
    artifacts = _cached_artifacts()

    # Show environment warnings once per session
    if artifacts.get("warnings") and not st.session_state.get("_artifact_warnings_shown", False):
        for warning_msg in artifacts["warnings"]:
            st.warning(f"Notice: {warning_msg}")
        st.session_state["_artifact_warnings_shown"] = True

    return artifacts
