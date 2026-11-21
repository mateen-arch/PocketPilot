"""
Smoke tests for PocketPilot deployment application and inference input validation.
"""

import pytest

from pocketpilot.config import PROJECT_ROOT
from pocketpilot.inference.validation import validate_inputs


def test_validate_inputs_valid():
    """Verify that legitimate user inputs pass validation without any error messages."""
    errors = validate_inputs(
        device_model="Google Pixel 5",
        app_usage_min=240,
        screen_on_hours=5.0,
        apps_installed=45,
        battery_budget=2500.0,
        data_budget=1200.0,
        screen_budget=6.0,
    )
    assert errors == []


def test_validate_inputs_zero_budget():
    """Verify that zero budgets are rejected with clear error messages."""
    errors_battery = validate_inputs(
        device_model="Google Pixel 5",
        app_usage_min=240,
        screen_on_hours=5.0,
        apps_installed=45,
        battery_budget=0.0,
        data_budget=1000.0,
        screen_budget=5.0,
    )
    assert any("Battery budget" in err and "strictly positive" in err for err in errors_battery)

    errors_data = validate_inputs(
        device_model="Google Pixel 5",
        app_usage_min=240,
        screen_on_hours=5.0,
        apps_installed=45,
        battery_budget=2500.0,
        data_budget=0.0,
        screen_budget=5.0,
    )
    assert any("Data budget" in err and "strictly positive" in err for err in errors_data)

    errors_screen = validate_inputs(
        device_model="Google Pixel 5",
        app_usage_min=240,
        screen_on_hours=5.0,
        apps_installed=45,
        battery_budget=2500.0,
        data_budget=1000.0,
        screen_budget=0.0,
    )
    assert any("Screen time budget" in err and "strictly positive" in err for err in errors_screen)


def test_validate_inputs_negative_budget():
    """Verify that negative budgets are rejected."""
    errors = validate_inputs(
        device_model="iPhone 12",
        app_usage_min=100,
        screen_on_hours=2.0,
        apps_installed=20,
        battery_budget=-500.0,
        data_budget=-200.0,
        screen_budget=-1.5,
    )
    assert len(errors) == 3
    assert any("Battery budget" in err for err in errors)
    assert any("Data budget" in err for err in errors)
    assert any("Screen time budget" in err for err in errors)


def test_validate_inputs_missing_values():
    """Verify that None/empty values are caught gracefully."""
    errors = validate_inputs(
        device_model=None,
        app_usage_min=None,
        screen_on_hours=None,
        apps_installed=None,
        battery_budget=None,
        data_budget=None,
        screen_budget=None,
    )
    assert len(errors) >= 7
    assert any("Device model" in err for err in errors)
    assert any("App usage" in err for err in errors)
    assert any("screen-on hours" in err for err in errors)
    assert any("apps installed" in err for err in errors)


def test_validate_inputs_out_of_range():
    """Verify that values outside physical domain bounds are flagged."""
    errors = validate_inputs(
        device_model="Unknown Phone Model",
        app_usage_min=2000,   # Max 1440 min
        screen_on_hours=25.0, # Max 24.0 h
        apps_installed=600,   # Max 500
        battery_budget=2000.0,
        data_budget=500.0,
        screen_budget=4.0,
    )
    assert any("not recognized" in err for err in errors)
    assert any("1 and 1440 minutes" in err for err in errors)
    assert any("0.1 and 24.0 hours" in err for err in errors)
    assert any("0 and 500" in err for err in errors)


def test_streamlit_app_smoke():
    """Verify Streamlit app entry point initializes cleanly and renders Home page content."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit is not installed in the active environment.")

    app_path = str(PROJECT_ROOT / "streamlit_app.py")
    at = AppTest.from_file(app_path, default_timeout=90)
    at.run()
    assert not at.exception, f"App raised exception on initial run: {at.exception}"

    # Verify Home page content appears
    titles = [t.value for t in at.title]
    assert any("PocketPilot" in t for t in titles), f"Home title not found in {titles}"
    
    # Verify disclaimer appears on home page
    info_boxes = [i.value for i in at.info]
    assert any("research and educational prototype" in i for i in info_boxes)
