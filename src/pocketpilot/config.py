"""
Configuration constants, dataset schemas, file paths, and random seeds.
"""

from pathlib import Path

# Project root = two levels above this file (src/pocketpilot/config.py), so paths work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DS1_PATH = RAW_DIR / "user_behavior_dataset.csv"
DS2_PATH = RAW_DIR / "mobile_app_usage_screen_time.csv"

# In Dataset 2, the text "None" is a real category of the column sleep_disruption_from_phone.
# Pandas would turn it into NaN by default, so always load Dataset 2 with these kwargs.
DS2_READ_KWARGS = {"keep_default_na": False}

RANDOM_SEED = 42

# Column renaming mapping from raw Dataset 1 column names to clean snake_case
COLUMN_RENAME: dict[str, str] = {
    "User ID": "user_id",
    "Device Model": "device_model",
    "Operating System": "os",
    "App Usage Time (min/day)": "app_usage_min",
    "Screen On Time (hours/day)": "screen_on_hours",
    "Battery Drain (mAh/day)": "battery_mah",
    "Number of Apps Installed": "apps_installed",
    "Data Usage (MB/day)": "data_mb",
    "Age": "age",
    "Gender": "gender",
    "User Behavior Class": "behavior_class",
}

# Regression and estimation targets
TARGET_BATTERY: str = "battery_mah"
TARGET_DATA: str = "data_mb"
TARGET_SCREEN: str = "screen_on_hours"

# Train/test split ratio
TEST_SIZE: float = 0.2

# Supported device models in Dataset 1
KNOWN_DEVICES: list[str] = [
    "Google Pixel 5",
    "OnePlus 9",
    "Samsung Galaxy S21",
    "Xiaomi Mi 11",
    "iPhone 12",
]

# Device to Operating System mapping (used later to display the OS in the Streamlit app)
DEVICE_OS_MAP: dict[str, str] = {
    "Google Pixel 5": "Android",
    "OnePlus 9": "Android",
    "Samsung Galaxy S21": "Android",
    "Xiaomi Mi 11": "Android",
    "iPhone 12": "iOS",
}

# Output split paths for cleaned and split Dataset 1
PROCESSED_TRAIN_PATH: Path = PROCESSED_DIR / "ds1_train.csv"
PROCESSED_TEST_PATH: Path = PROCESSED_DIR / "ds1_test.csv"
