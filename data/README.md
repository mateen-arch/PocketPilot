# Datasets

> "PocketPilot is a research and educational prototype trained on aggregated public mobile-usage records. It estimates daily behavior patterns and should not be interpreted as an operating-system-level battery monitor."

## Dataset 1: Mobile Device Usage and User Behavior Dataset (main training source)

- **Source**: https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset (license: TODO, check Kaggle page)
- **File**: `data/raw/user_behavior_dataset.csv`, 700 rows x 11 columns, no null values, no duplicate rows.
- **Columns**: User ID, Device Model, Operating System, App Usage Time (min/day), Screen On Time (hours/day), Battery Drain (mAh/day), Number of Apps Installed, Data Usage (MB/day), Age, Gender, User Behavior Class (1 to 5).
- **Provenance**: The Kaggle page states the data is generated from simulated user behavior.
- **Targets**: Battery Drain (mAh/day), Data Usage (MB/day), Screen On Time (hours/day, estimation model).
- **Known characteristics**: 5 device models only (Google Pixel 5, OnePlus 9, Samsung Galaxy S21, Xiaomi Mi 11 are Android; iPhone 12 is iOS), so Operating System is redundant with Device Model. App time, screen time, app count, battery and data are highly correlated with each other (r about 0.93 to 0.98). Age is uncorrelated with every other numeric column (r about 0). User Behavior Class is a banding of the usage columns (for example class 1 = app time 30 to 89 min, class 5 = 481 to 598 min), so it must NOT be used as an input in Forecast mode.
- **Preprocessing**: TODO (to be filled once the pipeline exists).
- **Train/test split**: TODO.

## Dataset 2: Mobile App Usage & Screen Time 2022-2024 (supplementary, optional)

- **Source**: https://www.kaggle.com/datasets/hamnamunir/mobile-app-usage-and-screen-time-2022-2024 (license: CC0 1.0 per its metadata)
- **File**: `data/raw/mobile_app_usage_screen_time.csv`, 10,000 rows x 34 columns.
- **Documentation**: `docs/datasets/`.
- **Provenance**: Fully synthetic (stated by the author).
- One row is one user-app usage
