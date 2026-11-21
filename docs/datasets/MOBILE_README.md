# 📱 Mobile App Usage & Screen Time Dataset (2022–2024)

**Author:** Hamna Munir  
**Version:** 1.0.0  
**Last Updated:** May 2026  
**License:** CC0: Public Domain  
**Dataset Type:** Synthetic / Simulated  

---

## 📌 Overview

This dataset provides a comprehensive, synthetic snapshot of **global mobile app usage and screen time behavior** spanning 2022 to 2024. It contains 10,000 usage records across 15 countries covering 12 app categories, 80+ popular apps, user demographics, engagement metrics, digital wellbeing indicators, and mental health impact scores.

Designed to support research on digital behavior, screen time addiction, app monetization, notification overload, and mobile health interventions.

---

## 🎯 Use Cases

- **Screen time prediction** using regression models
- **App category preference** analysis by age, gender & country
- **Digital wellbeing** research and intervention modeling
- **In-app purchase prediction** using ML classification
- **Mental health impact** of screen time analysis
- **Notification overload** and its effect on engagement
- **User segmentation** by usage pattern and demographics
- **Sleep disruption** correlation with late-night screen time
- **Battery & data consumption** optimization research
- **Subscription vs free** model engagement comparison

---

## 📁 Files

| File | Description |
|------|-------------|
| `mobile_app_usage_screen_time.csv` | Main dataset — 10,000 rows × 34 columns |
| `dataset-metadata.json` | Kaggle metadata with all column descriptors |
| `README.md` | Full documentation (this file) |

---

## 📊 Dataset Schema

### 🔑 Identifier & Time
| Column | Type | Description |
|--------|------|-------------|
| `record_id` | string | Unique record ID (APP0000001–APP0010000) |
| `year` | int | Year: 2022, 2023, 2024 |
| `month` | int | Month: 1–12 |
| `quarter` | string | Quarter: Q1–Q4 |
| `day_of_week` | string | Day of the week |

### 🌍 User Demographics
| Column | Type | Description |
|--------|------|-------------|
| `country` | string | User country (15 countries) |
| `age_group` | string | Age bracket: 13-17 through 65+ |
| `gender` | string | Male, Female, Prefer not to say |

### 📱 Device & Network
| Column | Type | Description |
|--------|------|-------------|
| `device_brand` | string | Smartphone brand (10 brands) |
| `operating_system` | string | Android or iOS |
| `network_connection` | string | 5G, 4G/LTE, 3G, WiFi, Offline |

### 🔧 App Details
| Column | Type | Description |
|--------|------|-------------|
| `app_category` | string | App category (12 categories) |
| `app_name` | string | Specific app name (80+ apps) |
| `subscription_type` | string | Free, Freemium, Paid Monthly, Paid Yearly, One-time Purchase |

### ⏱️ Usage Metrics
| Column | Type | Description |
|--------|------|-------------|
| `daily_screen_time_minutes` | float | Total daily screen time in minutes |
| `session_duration_minutes` | float | Average session length in minutes |
| `sessions_per_day` | int | Number of app sessions per day |
| `app_opens_per_day` | int | Total app opens per day |
| `data_used_mb_per_day` | float | Mobile data consumed per day in MB |
| `battery_drain_pct_per_session` | float | Battery percentage drained per session |
| `primary_usage_time` | string | Time of day with highest usage |

### 🔔 Notifications & Settings
| Column | Type | Description |
|--------|------|-------------|
| `notifications_received_per_day` | int | Push notifications received daily |
| `notification_settings` | string | All, Selected Only, Silent, Completely Off |
| `dark_mode_preference` | string | Always, Sometimes, Never |
| `location_permission` | string | Always, While Using, Never, Not Applicable |

### 💰 Monetization
| Column | Type | Description |
|--------|------|-------------|
| `in_app_purchase` | string | Whether user made in-app purchases: Yes, No |
| `monthly_spend_usd` | float | Monthly in-app spending in USD (0 if none) |
| `app_rating_given` | int | Star rating given to app: 1–5 |
| `review_submitted` | string | Whether user wrote a review: Yes, No |

### 🧠 Digital Wellbeing
| Column | Type | Description |
|--------|------|-------------|
| `sleep_disruption_from_phone` | string | None, Mild, Moderate, Severe |
| `screen_time_concern` | string | Yes, No, Somewhat |
| `mental_health_impact` | string | Very Positive to Very Negative |
| `digital_wellbeing_feature_used` | string | Yes, No |
| `app_deleted_and_reinstalled` | string | Yes, No |

---

## 📈 Key Statistics

| Metric | Value |
|--------|-------|
| Total Records | 10,000 |
| Total Features | 34 |
| Years Covered | 2022–2024 |
| Countries | 15 |
| App Categories | 12 |
| Apps Covered | 80+ |
| Zero Null Values | ✅ |
| Avg Daily Screen Time | ~180 min |
| In-App Purchase Rate | ~28% |

---

## 🔬 Suggested Research Questions

1. Which age group has the highest daily screen time?
2. Does late-night phone use correlate with sleep disruption?
3. Can we predict in-app purchases based on usage behavior?
4. Which app category has the highest battery drain?
5. How does mental health impact differ by app category?
6. Do users with more notifications have higher screen time?
7. Which countries show the highest screen time concern?
8. Does dark mode usage correlate with late-night usage?
9. How has screen time changed from 2022 to 2024?
10. What factors drive users to delete and reinstall apps?

---

## ⚠️ Disclaimer

This dataset is **entirely synthetic**. No real user data was collected. For educational and research purposes only.

---

## 📜 Citation

```
Munir, H. (2026). Mobile App Usage & Screen Time Dataset 2022-2024 (Synthetic) [Data set]. Kaggle.
https://www.kaggle.com/datasets/hamnamunir/mobile-app-usage-screen-time
```

---

## 🤝 Author

**Hamna Munir** | 🌐 [kaggle.com/hamnamunir](https://www.kaggle.com/hamnamunir)

*CC0 Public Domain License.*
