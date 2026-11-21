"""
Configuration constants, dataset schemas, file paths, and random seeds.
"""

from pathlib import Path

# Paths relative to the project root
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DS1_PATH = RAW_DIR / "user_behavior_dataset.csv"
DS2_PATH = RAW_DIR / "mobile_app_usage_screen_time.csv"

# "None" is a real category in Dataset 2 (e.g., subscription type, notification settings)
DS2_READ_KWARGS = {"keep_default_na": False}

RANDOM_SEED = 42
