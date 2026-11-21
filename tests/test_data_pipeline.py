"""
Tests for data loading, cleaning, and splitting pipelines.
"""

import pytest
import pandas as pd

from pocketpilot.data.loader import load_ds1, load_ds2
from pocketpilot.data.cleaning import clean_ds1
from pocketpilot.data.splitting import split_ds1, save_splits
from pocketpilot.config import RANDOM_SEED, TEST_SIZE


def test_data_loaders_dimensions():
    """Verify load_ds1 returns 700x11 and load_ds2 returns 10,000x34."""
    df1 = load_ds1()
    assert df1.shape == (700, 11), f"Expected (700, 11) for Dataset 1, got {df1.shape}"

    df2 = load_ds2()
    assert df2.shape == (10000, 34), f"Expected (10000, 34) for Dataset 2, got {df2.shape}"


def test_load_ds2_preserves_none_category():
    """Verify load_ds2 keeps literal string 'None' and has 0 NaN in sleep_disruption_from_phone."""
    df2 = load_ds2()
    col = "sleep_disruption_from_phone"
    assert col in df2.columns, f"Column '{col}' not found in Dataset 2"

    # Exactly 2,767 rows must equal literal "None"
    none_count = (df2[col] == "None").sum()
    assert none_count == 2767, f"Expected 2767 'None' occurrences, found {none_count}"

    # Zero NaN values in this column
    nan_count = df2[col].isna().sum()
    assert nan_count == 0, f"Expected 0 NaN values in '{col}', found {nan_count}"


def test_clean_ds1_standards_and_immutability():
    """Verify clean_ds1 returns snake_case, no nulls, unique user_id, and preserves input."""
    raw = load_ds1()
    raw_copy = raw.copy()

    cleaned = clean_ds1(raw)

    # Immutability check: raw DataFrame must be untouched
    pd.testing.assert_frame_equal(raw, raw_copy)

    # Cleaned assertions
    expected_cols = {
        "user_id",
        "device_model",
        "os",
        "app_usage_min",
        "screen_on_hours",
        "battery_mah",
        "apps_installed",
        "data_mb",
        "age",
        "gender",
        "behavior_class",
    }
    assert set(cleaned.columns) == expected_cols
    assert cleaned.isna().sum().sum() == 0, "Cleaned dataset contains unexpected null values"
    assert cleaned["user_id"].is_unique, "Cleaned user_id is not unique"
    assert len(cleaned) == 700


def test_clean_ds1_validation_failure():
    """Verify clean_ds1 raises ValueError when corrupted values violate physical ranges."""
    raw = load_ds1()

    # Corrupt a copy with negative battery drain
    corrupt_df = raw.copy()
    corrupt_df.loc[0, "Battery Drain (mAh/day)"] = -5
    with pytest.raises(ValueError, match="battery_mah"):
        clean_ds1(corrupt_df)

    # Corrupt app usage beyond 1440 min
    corrupt_df2 = raw.copy()
    corrupt_df2.loc[0, "App Usage Time (min/day)"] = 2000
    with pytest.raises(ValueError, match="app_usage_min"):
        clean_ds1(corrupt_df2)

    # Corrupt age
    corrupt_df3 = raw.copy()
    corrupt_df3.loc[0, "Age"] = 5
    with pytest.raises(ValueError, match="age"):
        clean_ds1(corrupt_df3)


def test_split_ds1_properties(tmp_path):
    """Verify split_ds1 partitions (560/140), no leakage, stratified classes, and deterministic seed."""
    raw = load_ds1()
    clean = clean_ds1(raw)

    train_df, test_df = split_ds1(clean, test_size=TEST_SIZE, seed=RANDOM_SEED)

    # Check partition sizes
    assert len(train_df) == 560, f"Expected 560 train rows, got {len(train_df)}"
    assert len(test_df) == 140, f"Expected 140 test rows, got {len(test_df)}"

    # Check zero overlap in user IDs
    assert set(train_df["user_id"]).isdisjoint(
        set(test_df["user_id"])
    ), "Detected overlapping user_id between train and test splits!"

    # Behavior class stratification within 2 percentage points (0.02)
    train_props = train_df["behavior_class"].value_counts(normalize=True).sort_index()
    test_props = test_df["behavior_class"].value_counts(normalize=True).sort_index()
    max_diff = (train_props - test_props).abs().max()
    assert (
        max_diff <= 0.02
    ), f"Stratification deviation {max_diff:.4f} exceeded tolerance of 0.02"

    # Determinism: identical results for identical seed
    train_df2, test_df2 = split_ds1(clean, test_size=TEST_SIZE, seed=RANDOM_SEED)
    pd.testing.assert_frame_equal(train_df, train_df2)
    pd.testing.assert_frame_equal(test_df, test_df2)

    # Verify save_splits writes valid CSVs without touching data/processed
    train_file = tmp_path / "test_train.csv"
    test_file = tmp_path / "test_test.csv"
    save_splits(train_df, test_df, train_path=train_file, test_path=test_file)
    assert train_file.exists() and test_file.exists()
    assert len(pd.read_csv(train_file)) == 560
    assert len(pd.read_csv(test_file)) == 140
