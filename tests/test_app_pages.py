"""End-to-end and component tests for PocketPilot multi-page Streamlit application.

Covers:
- Page 1: Daily Phone Planner (default run, execution, battery boundary, model estimate toggle)
- Page 2: Will I Make It? (default run, valid goal, violated goal, impossible goal)
- Page 3: Digital Use Profile (default run, persona classification, negative value error)
- Page 4: Model Evaluation (read-only transparent reporting, regression comparisons)
- Direct validation tests for parse_history_csv (valid, dates, missing, types, empty, bounds)
- Architectural boundary check: strictly prohibiting profile() calls in forecast pages
"""

import io
from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

from pocketpilot.config import PROJECT_ROOT
from pocketpilot.inference.validation import parse_history_csv


# -----------------------------------------------------------------------------
# PAGE 1 TESTS
# -----------------------------------------------------------------------------

def test_page_1_default_run():
    """Verify Page 1 loads cleanly without any uncaught exceptions on default initialization."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "1_Daily_Phone_Planner.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    assert not at.exception, f"Page 1 default run raised exception: {at.exception}"


def test_page_1_typical_user_execution():
    """Verify Page 1 produces a verdict, survival estimate, and 3 risk gauges upon clicking Run."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "1_Daily_Phone_Planner.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    
    # Click forecast run button
    btn = [b for b in at.button if b.key == "p1_run_btn"][0]
    btn.click().run()
    assert not at.exception, f"Page 1 run button click raised exception: {at.exception}"

    # Verdict banner
    banners = [w.value for w in list(at.success) + list(at.warning) if "Verdict:" in w.value]
    assert len(banners) >= 1, "Expected verdict banner not found in Page 1 output."

    # Survival estimate metric
    metric_labels = [m.label for m in at.metric]
    assert "Expected Survival Time" in metric_labels

    # 3 operational resource gauges (verified via explicit text 'Risk State')
    risk_state_count = sum(1 for m in at.markdown if "**Risk State:**" in m.value)
    assert risk_state_count == 3, f"Expected 3 resource risk gauges, found {risk_state_count}."


def test_page_1_depleted_battery_boundary():
    """Verify that current battery at or below minimum threshold triggers an error and suppresses predictions."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "1_Daily_Phone_Planner.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    
    # Set current battery to 10% when min reserve is 15%
    at.number_input(key="p1_current_battery_pct").set_value(10)
    at.number_input(key="p1_min_battery_pct").set_value(15)
    btn = [b for b in at.button if b.key == "p1_run_btn"][0]
    btn.click().run()

    assert not at.exception
    assert len(at.error) >= 1
    assert any("already at or below" in e.value for e in at.error)
    # Ensure no prediction metric is displayed
    assert not any(m.label == "Expected Survival Time" for m in at.metric)


def test_page_1_estimate_screen_toggle():
    """Verify that ticking 'estimate screen time for me' tags screen time as model estimate."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "1_Daily_Phone_Planner.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    
    # Check the estimate screen checkbox
    at.checkbox(key="p1_estimate_screen").check().run()
    btn = [b for b in at.button if b.key == "p1_run_btn"][0]
    btn.click().run()

    assert not at.exception
    # Verify screen time source indicates model estimate
    screen_metric = [m for m in at.metric if m.label == "Predicted Screen Time"]
    assert len(screen_metric) >= 1
    assert "Model estimate" in screen_metric[0].delta


# -----------------------------------------------------------------------------
# PAGE 2 TESTS
# -----------------------------------------------------------------------------

def test_page_2_default_run():
    """Verify Page 2 loads cleanly without exceptions."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "2_Will_I_Make_It.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    assert not at.exception, f"Page 2 default run raised exception: {at.exception}"


def test_page_2_valid_goal_verdict():
    """Verify Page 2 evaluates a valid goal and displays a verdict."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "2_Will_I_Make_It.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    
    btn = [b for b in at.button if b.key == "p2_eval_btn"][0]
    btn.click().run()

    assert not at.exception
    banners = [w.value for w in list(at.success) + list(at.warning) if "Verdict:" in w.value]
    assert len(banners) >= 1, "Expected verdict banner not found in Page 2."


def test_page_2_already_violated_goal():
    """Verify Page 2 displays an error when current battery is already below the desired target reserve."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "2_Will_I_Make_It.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    
    # Target reserve is 25%, current battery is 20%
    at.number_input(key="p2_current_battery_pct").set_value(20)
    at.number_input(key="p2_target_min_pct").set_value(25)
    btn = [b for b in at.button if b.key == "p2_eval_btn"][0]
    btn.click().run()

    assert not at.exception
    assert len(at.error) >= 1
    assert any("Goal already violated" in e.value for e in at.error)


def test_page_2_impossible_goal_not_achievable():
    """Verify Page 2 correctly declares an impossible goal not-achievable without claiming success."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "2_Will_I_Make_It.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    
    # Heavy user demanding 90% battery left after 8 hours of heavy usage
    at.number_input(key="p2_app_usage_min").set_value(480)
    at.number_input(key="p2_screen_on_hours").set_value(9.0)
    at.number_input(key="p2_apps_installed").set_value(85)
    at.number_input(key="p2_battery_capacity_mah").set_value(3000)
    at.number_input(key="p2_current_battery_pct").set_value(95)
    at.number_input(key="p2_target_min_pct").set_value(90)
    at.number_input(key="p2_target_duration_hours").set_value(8.0)
    
    btn = [b for b in at.button if b.key == "p2_eval_btn"][0]
    btn.click().run()

    assert not at.exception
    # Must explicitly state not achievable
    error_texts = [e.value for e in at.error]
    assert any("Goal is Not Achievable" in txt for txt in error_texts)
    # Must NOT claim goal achievable in success banners
    assert not any("Goal Achievable" in s.value for s in at.success)


# -----------------------------------------------------------------------------
# PAGE 3 TESTS
# -----------------------------------------------------------------------------

def test_page_3_default_run():
    """Verify Page 3 loads cleanly without exceptions."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "3_Digital_Use_Profile.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    assert not at.exception, f"Page 3 default run raised exception: {at.exception}"


def test_page_3_manual_entry_valid():
    """Verify Page 3 classifies valid manual inputs into one of the designated neutral personas."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "3_Digital_Use_Profile.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    
    btn = [b for b in at.button if b.key == "p3_classify_btn"][0]
    btn.click().run()

    assert not at.exception
    success_texts = [s.value for s in at.success]
    assert any("Assigned Behavioral Persona:" in txt for txt in success_texts)
    
    valid_personas = ["Light use", "Moderate use", "Heavy use", "Very heavy use"]
    assert any(any(vp in txt for vp in valid_personas) for txt in success_texts)


def test_page_3_negative_input_error():
    """Verify Page 3 rejects negative numbers with an error banner."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "3_Digital_Use_Profile.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    
    at.number_input(key="p3_obs_app").set_value(-50.0)
    btn = [b for b in at.button if b.key == "p3_classify_btn"][0]
    btn.click().run()

    assert not at.exception
    assert len(at.error) >= 1
    assert any("strictly positive" in e.value for e in at.error)


# -----------------------------------------------------------------------------
# PAGE 4 TESTS
# -----------------------------------------------------------------------------

def test_page_4_renders_without_exception():
    """Verify Page 4 loads all tabs and displays regression benchmarks for all three tasks."""
    page_path = str(PROJECT_ROOT / "app" / "pages" / "4_Model_Evaluation.py")
    at = AppTest.from_file(page_path, default_timeout=90).run()
    assert not at.exception, f"Page 4 raised exception: {at.exception}"

    subheaders = [s.value for s in at.subheader]
    assert any("Battery Drain" in sh for sh in subheaders)
    assert any("Data Consumption" in sh for sh in subheaders)
    assert any("Screen Time" in sh for sh in subheaders)


# -----------------------------------------------------------------------------
# PARSE_HISTORY_CSV DIRECT TESTS
# -----------------------------------------------------------------------------

def test_parse_history_csv_valid_with_and_without_date():
    """Verify parse_history_csv handles valid CSV files both with and without date columns."""
    csv_without_date = "app_usage_min,screen_on_hours,battery_mah,data_mb\n120,3.5,900,450\n"
    df1 = parse_history_csv(io.StringIO(csv_without_date))
    assert len(df1) == 1
    assert list(df1.columns) == ["app_usage_min", "screen_on_hours", "battery_mah", "data_mb"]
    assert df1["app_usage_min"].iloc[0] == 120.0

    csv_with_date = "date,app_usage_min,screen_on_hours,battery_mah,data_mb\n2026-01-01,200,4.0,1100,550\n"
    df2 = parse_history_csv(csv_with_date.encode("utf-8"))
    assert len(df2) == 1
    assert "date" in df2.columns
    assert pd.api.types.is_datetime64_any_dtype(df2["date"])


def test_parse_history_csv_missing_column():
    """Verify parse_history_csv raises ValueError when a required column is missing."""
    csv_missing = "app_usage_min,screen_on_hours,battery_mah\n120,3.5,900\n"
    with pytest.raises(ValueError, match="missing required column"):
        parse_history_csv(csv_missing)


def test_parse_history_csv_non_numeric():
    """Verify parse_history_csv raises ValueError naming the column when non-numeric data is present."""
    csv_bad = "app_usage_min,screen_on_hours,battery_mah,data_mb\n120,not_a_number,900,450\n"
    with pytest.raises(ValueError, match="screen_on_hours"):
        parse_history_csv(csv_bad)


def test_parse_history_csv_empty_file():
    """Verify parse_history_csv rejects empty files and files with headers only."""
    with pytest.raises(ValueError, match="empty"):
        parse_history_csv("")
    
    with pytest.raises(ValueError, match="no data rows"):
        parse_history_csv("app_usage_min,screen_on_hours,battery_mah,data_mb\n")


def test_parse_history_csv_too_many_rows():
    """Verify parse_history_csv enforces the max_rows limit (default 366)."""
    header = "app_usage_min,screen_on_hours,battery_mah,data_mb\n"
    rows = "100,3.0,800,400\n" * 367
    with pytest.raises(ValueError, match="exceeding maximum allowed limit"):
        parse_history_csv(header + rows, max_rows=366)


def test_parse_history_csv_negative_value():
    """Verify parse_history_csv rejects zero and negative values."""
    csv_neg = "app_usage_min,screen_on_hours,battery_mah,data_mb\n120,3.5,-500,450\n"
    with pytest.raises(ValueError, match="non-positive values"):
        parse_history_csv(csv_neg)


def test_parse_history_csv_case_insensitive():
    """Verify parse_history_csv trims whitespace and is case-insensitive."""
    csv_upper = " APP_USAGE_MIN , Screen_On_Hours , BATTERY_MAH , Data_MB \n150,4.0,1000,500\n"
    df = parse_history_csv(csv_upper)
    assert set(df.columns) == {"app_usage_min", "screen_on_hours", "battery_mah", "data_mb"}
    assert df["battery_mah"].iloc[0] == 1000.0


# -----------------------------------------------------------------------------
# ARCHITECTURAL SEPARATION AUDIT
# -----------------------------------------------------------------------------

def test_no_profile_call_in_forecast_pages():
    """Verify architectural boundary: Pages 1 and 2 must never call profile() on forecast outputs."""
    p1_file = PROJECT_ROOT / "app" / "pages" / "1_Daily_Phone_Planner.py"
    p2_file = PROJECT_ROOT / "app" / "pages" / "2_Will_I_Make_It.py"

    p1_text = p1_file.read_text(encoding="utf-8")
    p2_text = p2_file.read_text(encoding="utf-8")

    assert "profile(" not in p1_text, "Found forbidden call 'profile(' in 1_Daily_Phone_Planner.py"
    assert "profile(" not in p2_text, "Found forbidden call 'profile(' in 2_Will_I_Make_It.py"
