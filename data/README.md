# Datasets

> "PocketPilot is a research and educational prototype trained on aggregated public mobile-usage records. It estimates daily behavior patterns and should not be interpreted as an operating-system-level battery monitor."

## Dataset 1: Mobile Device Usage and User Behavior Dataset (main training source)

- **Source**: https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset (license: TODO, check Kaggle page)
- **File**: `data/raw/user_behavior_dataset.csv`, 700 rows x 11 columns, no null values, no duplicate rows.
- **Columns**: User ID, Device Model, Operating System, App Usage Time (min/day), Screen On Time (hours/day), Battery Drain (mAh/day), Number of Apps Installed, Data Usage (MB/day), Age, Gender, User Behavior Class (1 to 5).
- **Provenance**: The Kaggle page states the data is generated from simulated user behavior.
- **Targets**: Battery Drain (mAh/day), Data Usage (MB/day), Screen On Time (hours/day, estimation model).
- **Known characteristics**: 5 device models only (Google Pixel 5, OnePlus 9, Samsung Galaxy S21, Xiaomi Mi 11 are Android; iPhone 12 is iOS), so Operating System is redundant with Device Model. App time, screen time, app count, battery and data are highly correlated with each other (r about 0.93 to 0.98). Age is uncorrelated with every other numeric column (r about 0). User Behavior Class is a banding of the usage columns (for example class 1 = app time 30 to 89 min, class 5 = 481 to 598 min), so it must NOT be used as an input in Forecast mode.
- **Preprocessing**: Columns renamed to snake_case; duplicates and nulls checked (none found); physical range validation; no outlier removal or scaling at cleaning time. Device Model is one-hot encoded and numeric features are standardized inside the model pipelines, fitted on the training split only.
- **Train/test split**: 80/20 split stratified on User Behavior Class, random_state 42, giving 560 training and 140 test rows; the test set is used only for final evaluation.

## Dataset 2: Mobile App Usage & Screen Time 2022-2024 (supplementary, optional)

- **Source**: https://www.kaggle.com/datasets/hamnamunir/mobile-app-usage-and-screen-time-2022-2024 (license: CC0 1.0 per its metadata)
- **File**: `data/raw/mobile_app_usage_screen_time.csv`, 10,000 rows x 34 columns.
- **Documentation**: `docs/datasets/`.
- **Provenance**: Fully synthetic (stated by the author).
- **Row meaning**: one row is one user-app usage record. There is NO user ID column, so multi-day user history cannot be built from this dataset.
- **Loading warning**: the text value "None" in column `sleep_disruption_from_phone` (2,767 rows) is a real category, not a missing value. Pandas converts it to NaN by default, so always load this file with `keep_default_na=False` (or equivalent).
- **Known limitations found during profiling**:
  - `battery_drain_pct_per_session` is a percentage per session (not mAh) and is statistically unrelated to every other column, so it is NOT usable as a battery target.
  - `network_connection` and `notifications_received_per_day` show no relationship with data usage or screen time, so Wi-Fi-shift and notification effects cannot be learned from this dataset.
  - Its documentation claims about 180 minutes average daily screen time, but the actual mean is about 85.5 minutes.
  - App category carries real signal for data usage and screen time.
- **Role in PocketPilot**: not used to train the battery or data-usage models. Optional use only for a category-level data/screen-time estimate.

## Limitations

Both datasets are simulated daily aggregates. They contain no minute-by-minute telemetry, no per-app battery consumption and no individual time series, so results describe patterns in simulated data and not real-world phone behavior.