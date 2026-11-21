"""
Configuration constants, dataset schemas, file paths, and random seeds.
"""

from pathlib import Path
from typing import Any

# Project root = two levels above this file (src/pocketpilot/config.py), so paths work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

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

# Clustering model path and evaluation ranges
CLUSTERING_MODEL_PATH: Path = MODELS_DIR / "clustering_model.joblib"
CLUSTER_K_REPORT_RANGE: list[int] = list(range(2, 9))
# Documented design decision: selection is restricted to 3 to 6 profiles so that
# persona labels remain granular yet readily interpretable for non-expert users.
CLUSTER_K_SELECT_RANGE: tuple[int, int] = (3, 6)

# Saved regression model locations
MODEL_PATHS: dict[str, Path] = {
    "battery": MODELS_DIR / "battery_model.joblib",
    "data": MODELS_DIR / "data_usage_model.joblib",
    "screen_time": MODELS_DIR / "screen_time_model.joblib",
}

# Budget overrun risk constants
RISK_MODEL_PATH: Path = MODELS_DIR / "risk_model.joblib"
WATCH_FRACTION: float = 0.8
INTERVAL_LEVELS: tuple[float, float] = (0.8, 0.9)
BUDGET_PERCENTILES: tuple[int, ...] = (20, 35, 50, 65, 80)
HETEROSCEDASTICITY_RHO_THRESHOLD: float = 0.2
OVERRUN_PROB_DECISION: float = 0.5

# What-if planner levers and parameters
PLANNER_APP_CUTS: tuple[int, ...] = (0, 15, 30, 45, 60, 90, 120)   # minutes of app use removed
PLANNER_SCREEN_CUTS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)         # additional hours of screen-on time removed
PLANNER_WIFI_SHARES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)          # ASSUMPTION: share of mobile data moved to Wi-Fi
POWER_SAVING_BATTERY_MULTIPLIER: float = 0.85              # ASSUMPTION: not learned from data, configurable

# Action-cost weights: design choices reflecting user convenience penalty, not learned from data
W1_MINUTES: float = 1.0
W2_POWER_SAVING: float = 20.0
W3_WIFI_PER_PCT_POINT: float = 0.5

PLANNER_TOP_N: int = 5
PLANNER_BUDGET_SETTINGS: dict[str, int] = {"tight": 35, "moderate": 50, "loose": 65}   # train-target percentiles used for evaluation only

# Battery survival estimator constants (Feature F8)
WAKING_HOURS: float = 16.0  # ASSUMPTION: daily drain is spread evenly over waking hours
DEFAULT_MIN_BATTERY_PCT: int = 15
CAPACITY_RANGE_MAH: tuple[int, int] = (1000, 10000)
UNCONSTRAINED_BUDGET: float = 1e9  # used to leave a resource unconstrained in the planner

# Example user persona presets for interactive app pages
PERSONAS: dict[str, dict[str, Any]] = {
    "light": {
        "label": "Light user (Pixel 5)",
        "device_model": "Google Pixel 5",
        "app_usage_min": 100,
        "screen_on_hours": 2.5,
        "apps_installed": 30,
        "battery_capacity_mah": 3000,
    },
    "typical": {
        "label": "Typical user (Galaxy S21)",
        "device_model": "Samsung Galaxy S21",
        "app_usage_min": 250,
        "screen_on_hours": 5.0,
        "apps_installed": 50,
        "battery_capacity_mah": 4000,
    },
    "heavy": {
        "label": "Heavy user (iPhone 12)",
        "device_model": "iPhone 12",
        "app_usage_min": 480,
        "screen_on_hours": 9.0,
        "apps_installed": 85,
        "battery_capacity_mah": 3000,
    },
}

