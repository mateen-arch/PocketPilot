"""
Input validation utilities for PocketPilot inference and deployment apps.
"""

import io
from typing import Any
import numpy as np
import pandas as pd

from pocketpilot.config import KNOWN_DEVICES


def validate_inputs(
    device_model: Any,
    app_usage_min: Any,
    screen_on_hours: Any,
    apps_installed: Any,
    battery_budget: Any,
    data_budget: Any,
    screen_budget: Any,
) -> list[str]:
    """
    Validate user inputs for PocketPilot forecast and risk estimation.

    Parameters
    ----------
    device_model : Any
        Selected smartphone device model. Must be in KNOWN_DEVICES.
    app_usage_min : Any
        Daily app usage in minutes. Range: [1, 1440].
    screen_on_hours : Any
        Expected screen-on time in hours. Range: [0.1, 24.0].
    apps_installed : Any
        Number of installed applications. Range: [0, 500].
    battery_budget : Any
        User battery budget in mAh. Must be strictly positive (> 0).
    data_budget : Any
        User data budget in MB. Must be strictly positive (> 0).
    screen_budget : Any
        User screen time budget in hours. Must be strictly positive (> 0).

    Returns
    -------
    list[str]
        List of human-readable error messages. Empty list if all inputs are valid.
    """
    errors: list[str] = []

    # 1. Device model validation
    if device_model is None or not isinstance(device_model, str) or not device_model.strip():
        errors.append("Device model must be provided.")
    elif device_model not in KNOWN_DEVICES:
        errors.append(f"Device model '{device_model}' is not recognized. Expected one of: {KNOWN_DEVICES}.")

    # 2. App usage minutes validation: [1, 1440]
    if app_usage_min is None:
        errors.append("App usage minutes must be provided.")
    else:
        try:
            val_app = float(app_usage_min)
            if not np.isfinite(val_app) or val_app < 1.0 or val_app > 1440.0:
                errors.append("App usage must be between 1 and 1440 minutes.")
        except (ValueError, TypeError):
            errors.append("App usage minutes must be a valid number.")

    # 3. Expected screen-on hours validation: [0.1, 24.0]
    if screen_on_hours is None:
        errors.append("Expected screen-on hours must be provided.")
    else:
        try:
            val_screen = float(screen_on_hours)
            if not np.isfinite(val_screen) or val_screen < 0.1 or val_screen > 24.0:
                errors.append("Expected screen-on hours must be between 0.1 and 24.0 hours.")
        except (ValueError, TypeError):
            errors.append("Expected screen-on hours must be a valid number.")

    # 4. Apps installed validation: [0, 500]
    if apps_installed is None:
        errors.append("Number of apps installed must be provided.")
    else:
        try:
            val_apps = float(apps_installed)
            if not np.isfinite(val_apps) or val_apps < 0.0 or val_apps > 500.0:
                errors.append("Number of apps installed must be between 0 and 500.")
        except (ValueError, TypeError):
            errors.append("Number of apps installed must be a valid number.")

    # 5. Battery budget validation: > 0 mAh
    if battery_budget is None:
        errors.append("Battery budget must be provided.")
    else:
        try:
            val_bat_b = float(battery_budget)
            if not np.isfinite(val_bat_b) or val_bat_b <= 0.0:
                errors.append("Battery budget must be strictly positive (> 0 mAh).")
        except (ValueError, TypeError):
            errors.append("Battery budget must be a valid number.")

    # 6. Data budget validation: > 0 MB
    if data_budget is None:
        errors.append("Data budget must be provided.")
    else:
        try:
            val_data_b = float(data_budget)
            if not np.isfinite(val_data_b) or val_data_b <= 0.0:
                errors.append("Data budget must be strictly positive (> 0 MB).")
        except (ValueError, TypeError):
            errors.append("Data budget must be a valid number.")

    # 7. Screen time budget validation: > 0 hours
    if screen_budget is None:
        errors.append("Screen time budget must be provided.")
    else:
        try:
            val_scr_b = float(screen_budget)
            if not np.isfinite(val_scr_b) or val_scr_b <= 0.0:
                errors.append("Screen time budget must be strictly positive (> 0 hours).")
        except (ValueError, TypeError):
            errors.append("Screen time budget must be a valid number.")

    return errors


def parse_history_csv(
    file_like_or_bytes: Any,
    max_rows: int = 366,
    max_bytes: int = 1_000_000,
) -> pd.DataFrame:
    """
    Parse and validate an uploaded multi-day usage history CSV.

    Parameters
    ----------
    file_like_or_bytes : file-like object, bytes, or str
        Uploaded CSV data.
    max_rows : int, default=366
        Maximum allowed row count.
    max_bytes : int, default=1_000_000
        Maximum allowed raw file size in bytes (1 MB).

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with lowercased column names and parsed dates if present.

    Raises
    ------
    ValueError
        If file exceeds size, is empty, has missing columns, non-numeric or non-positive
        values, invalid dates, or exceeds max_rows.
    """
    if isinstance(file_like_or_bytes, bytes):
        raw_bytes = file_like_or_bytes
        stream = io.BytesIO(raw_bytes)
    elif hasattr(file_like_or_bytes, "read"):
        raw_bytes = file_like_or_bytes.read()
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-8")
        stream = io.BytesIO(raw_bytes)
    elif isinstance(file_like_or_bytes, str):
        raw_bytes = file_like_or_bytes.encode("utf-8")
        stream = io.BytesIO(raw_bytes)
    else:
        raise ValueError(f"Unsupported input type for parse_history_csv: {type(file_like_or_bytes)}")

    if len(raw_bytes) == 0:
        raise ValueError("Uploaded CSV file is empty.")
    if len(raw_bytes) > max_bytes:
        raise ValueError(f"File size ({len(raw_bytes)} bytes) exceeds maximum limit of {max_bytes} bytes.")

    try:
        df = pd.read_csv(stream)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file: {exc}") from exc

    if len(df) == 0:
        raise ValueError("Uploaded CSV contains no data rows.")
    if len(df) > max_rows:
        raise ValueError(f"Uploaded CSV has {len(df)} rows, exceeding maximum allowed limit of {max_rows} rows.")

    # Normalize column names: strip and lowercase
    raw_col_map = {col: str(col).strip().lower() for col in df.columns}
    df = df.rename(columns=raw_col_map)

    required_numeric = ["app_usage_min", "screen_on_hours", "battery_mah", "data_mb"]
    missing = [c for c in required_numeric if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}. Required columns: {required_numeric}.")

    # Validate and convert required numeric columns
    for col in required_numeric:
        try:
            converted = pd.to_numeric(df[col], errors="raise")
        except Exception as exc:
            raise ValueError(f"Column '{col}' contains non-numeric values that could not be converted to numbers.") from exc

        if converted.isna().any() or np.isinf(converted).any():
            raise ValueError(f"Column '{col}' contains NaN or infinite values.")
        if (converted <= 0).any():
            raise ValueError(f"Column '{col}' contains non-positive values (<= 0). All daily usage metrics must be strictly positive.")
        df[col] = converted

    # Check for optional date column
    date_cols = [c for c in df.columns if c in ("date", "timestamp", "datetime", "day")]
    if date_cols:
        d_col = date_cols[0]
        try:
            df[d_col] = pd.to_datetime(df[d_col], errors="raise")
        except Exception as exc:
            raise ValueError(f"Column '{d_col}' contains invalid or unparseable date values.") from exc

    return df

