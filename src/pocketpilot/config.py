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
