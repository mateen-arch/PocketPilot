"""
Data cleaning and validation module for Dataset 1 (Mobile Device Usage and User Behavior).

Design Note on Outliers:
Outlier removal is deliberately NOT performed on Dataset 1. The dataset originates from
a synthetic/simulated user behavior model where all values already reside within plausible
daily human operational bounds (e.g. daily app time <= 10 hours, screen time <= 12 hours).
Removing points based on generic statistical heuristics (such as IQR or z-scores) would
artificially truncate heavy-usage user personas (e.g., intensive gamers or media power users)
that PocketPilot specifically aims to forecast and optimize.
Scaling and encoding are also excluded here to avoid premature transformation before splitting.
"""

import logging
import pandas as pd

from pocketpilot.config import COLUMN_RENAME

logger = logging.getLogger(__name__)


def clean_ds1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, standardize, cast, and validate Dataset 1.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Dataset 1 DataFrame loaded from CSV.

    Returns
    -------
    pd.DataFrame
        New cleaned DataFrame with snake_case column names, enforced dtypes,
        and validated physical ranges. The original input DataFrame is never modified.

    Raises
    ------
    ValueError
        If any data validation rule or physical range check is violated,
        listing each offending column and the number of violating rows.
    """
    # 1. Work on a copy to ensure input immutability
    cleaned = df.copy()

    # 1. Rename columns to snake_case
    cleaned = cleaned.rename(columns=COLUMN_RENAME)

    # 2. Drop exact duplicate rows and log
    initial_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    dropped_duplicates = initial_rows - len(cleaned)
    if dropped_duplicates > 0:
        logger.info("Dropped %d exact duplicate row(s).", dropped_duplicates)

    # 3. Drop rows with nulls in any column and log
    rows_before_nulls = len(cleaned)
    cleaned = cleaned.dropna()
    dropped_nulls = rows_before_nulls - len(cleaned)
    if dropped_nulls > 0:
        logger.info("Dropped %d row(s) containing missing values.", dropped_nulls)

    # 4. Cast numeric columns to proper dtypes
    integer_columns = [
        "user_id",
        "app_usage_min",
        "battery_mah",
        "apps_installed",
        "data_mb",
        "age",
        "behavior_class",
    ]
    for col in integer_columns:
        cleaned[col] = cleaned[col].astype(int)

    cleaned["screen_on_hours"] = cleaned["screen_on_hours"].astype(float)

    # 5. Validate plausible physical ranges
    validation_errors: list[str] = []

    # 0 < app_usage_min <= 1440 (minutes in a 24-hour day)
    bad_app = cleaned[~((cleaned["app_usage_min"] > 0) & (cleaned["app_usage_min"] <= 1440))]
    if len(bad_app) > 0:
        validation_errors.append(
            f"app_usage_min (expected 0 < min <= 1440): {len(bad_app)} violating row(s)"
        )

    # 0 < screen_on_hours <= 24 (hours in a day)
    bad_screen = cleaned[~((cleaned["screen_on_hours"] > 0) & (cleaned["screen_on_hours"] <= 24))]
    if len(bad_screen) > 0:
        validation_errors.append(
            f"screen_on_hours (expected 0 < hours <= 24): {len(bad_screen)} violating row(s)"
        )

    # battery_mah > 0
    bad_battery = cleaned[~(cleaned["battery_mah"] > 0)]
    if len(bad_battery) > 0:
        validation_errors.append(
            f"battery_mah (expected > 0): {len(bad_battery)} violating row(s)"
        )

    # data_mb >= 0
    bad_data = cleaned[~(cleaned["data_mb"] >= 0)]
    if len(bad_data) > 0:
        validation_errors.append(
            f"data_mb (expected >= 0): {len(bad_data)} violating row(s)"
        )

    # apps_installed >= 0
    bad_apps = cleaned[~(cleaned["apps_installed"] >= 0)]
    if len(bad_apps) > 0:
        validation_errors.append(
            f"apps_installed (expected >= 0): {len(bad_apps)} violating row(s)"
        )

    # 10 <= age <= 100
    bad_age = cleaned[~((cleaned["age"] >= 10) & (cleaned["age"] <= 100))]
    if len(bad_age) > 0:
        validation_errors.append(
            f"age (expected 10 <= age <= 100): {len(bad_age)} violating row(s)"
        )

    # behavior_class in {1, 2, 3, 4, 5}
    bad_class = cleaned[~cleaned["behavior_class"].isin({1, 2, 3, 4, 5})]
    if len(bad_class) > 0:
        validation_errors.append(
            f"behavior_class (expected in {{1, 2, 3, 4, 5}}): {len(bad_class)} violating row(s)"
        )

    # user_id unique
    if not cleaned["user_id"].is_unique:
        duplicate_users = cleaned["user_id"].duplicated().sum()
        validation_errors.append(
            f"user_id (must be unique): {duplicate_users} duplicate user ID(s) detected"
        )

    if validation_errors:
        error_msg = "Data validation checks failed:\n" + "\n".join(f" - {err}" for err in validation_errors)
        raise ValueError(error_msg)

    # 6. Outlier removal, scaling, and categorical encoding are deliberately excluded.
    return cleaned
