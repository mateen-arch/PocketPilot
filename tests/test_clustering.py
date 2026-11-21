"""
Tests for usage-profile clustering pipeline, k-selection, labeling, and inference.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from pocketpilot.config import CLUSTER_K_SELECT_RANGE
from pocketpilot.data.cleaning import clean_ds1
from pocketpilot.data.loader import load_ds1
from pocketpilot.data.splitting import split_ds1
from pocketpilot.features.input_modes import get_feature_columns
from pocketpilot.models.clustering import (
    NEUTRAL_LABELS,
    evaluate_k_range,
    fit_profile_model,
    percentile_report,
    predict_profile,
    select_k,
)


@pytest.fixture(scope="module")
def memory_splits():
    """Build train and test splits in memory without reading from data/processed/."""
    raw = load_ds1()
    clean = clean_ds1(raw)
    train_df, test_df = split_ds1(clean)
    return clean, train_df, test_df


def test_clustering_feature_leakage_and_schema():
    """Verify clustering features match profile mode schema and strictly exclude behavior_class."""
    feature_cols = get_feature_columns("clustering", mode="profile")
    expected = ["app_usage_min", "screen_on_hours", "battery_mah", "data_mb"]
    assert feature_cols == expected, f"Feature list mismatch: {feature_cols}"
    assert "behavior_class" not in feature_cols, "behavior_class must never be an input to clustering!"


def test_fitted_scaler_uses_train_split_only(memory_splits):
    """Verify scaler mean matches training set mean and differs from full dataset mean."""
    clean_df, train_df, _ = memory_splits
    feature_cols = get_feature_columns("clustering", mode="profile")

    bundle = fit_profile_model(train_df, k=4)
    scaler = bundle["pipeline"].named_steps["scaler"]

    train_mean = train_df[feature_cols].mean().values
    full_mean = clean_df[feature_cols].mean().values

    assert np.allclose(scaler.mean_, train_mean), "Fitted scaler mean does not match train split mean!"
    assert not np.allclose(scaler.mean_, full_mean), "Fitted scaler mean leaked full dataset statistics!"


def test_select_k_bounds_and_criterion(memory_splits):
    """Verify select_k returns a value within CLUSTER_K_SELECT_RANGE matching highest KMeans silhouette."""
    _, train_df, _ = memory_splits
    results_df = evaluate_k_range(train_df, list(range(2, 9)))
    chosen_k = select_k(results_df)

    min_k, max_k = CLUSTER_K_SELECT_RANGE
    assert min_k <= chosen_k <= max_k, f"Chosen k {chosen_k} outside allowed range {CLUSTER_K_SELECT_RANGE}"

    km_sub = results_df[
        (results_df["method"] == "kmeans")
        & (results_df["k"] >= min_k)
        & (results_df["k"] <= max_k)
    ]
    expected_k = int(km_sub.loc[km_sub["silhouette"].idxmax()]["k"])
    assert chosen_k == expected_k, f"Expected k={expected_k} (highest silhouette), got {chosen_k}"


def test_label_clusters_intensity_ordering(memory_splits):
    """Verify cluster labels are unique and map monotonically to scaled centroid intensity."""
    _, train_df, _ = memory_splits
    k = 4
    bundle = fit_profile_model(train_df, k=k)
    label_map = bundle["label_map"]

    labels = list(label_map.values())
    assert len(set(labels)) == k, "Assigned labels are not all distinct"

    # Compute intensity order directly
    kmeans = bundle["pipeline"].named_steps["kmeans"]
    intensities = kmeans.cluster_centers_.mean(axis=1)
    sorted_cids = np.argsort(intensities)

    assert label_map[int(sorted_cids[0])] == "Light use", "Lowest intensity cluster must be 'Light use'"
    assert label_map[int(sorted_cids[-1])] == "Very heavy use", "Highest intensity cluster must be 'Very heavy use'"


def test_neutral_labels_terminology():
    """Verify that no persona label in NEUTRAL_LABELS contains stigmatizing terminology."""
    forbidden = ["addict", "problematic", "unhealthy"]
    for k, labels in NEUTRAL_LABELS.items():
        for lbl in labels:
            for bad_term in forbidden:
                assert bad_term not in lbl.lower(), (
                    f"Label '{lbl}' violates neutrality policy by containing '{bad_term}'"
                )


def test_predict_profile_validity_and_determinism(memory_splits):
    """Verify predict_profile assigns valid labels and is deterministic across identical seed fits."""
    _, train_df, test_df = memory_splits

    bundle1 = fit_profile_model(train_df, k=4)
    preds1 = predict_profile(bundle1, test_df)

    assert len(preds1) == len(test_df)
    assert set(preds1.columns) == {"cluster_id", "profile_label"}
    assert set(preds1["profile_label"]).issubset(set(bundle1["label_map"].values()))

    # Determinism
    bundle2 = fit_profile_model(train_df, k=4)
    preds2 = predict_profile(bundle2, test_df)
    pd.testing.assert_frame_equal(preds1, preds2)


def test_bundle_serialization_parity(memory_splits, tmp_path):
    """Verify that a saved-then-loaded bundle yields identical predictions."""
    _, train_df, test_df = memory_splits
    bundle = fit_profile_model(train_df, k=4)
    orig_preds = predict_profile(bundle, test_df)

    dump_file = tmp_path / "temp_clustering.joblib"
    joblib.dump(bundle, dump_file)

    loaded_bundle = joblib.load(dump_file)
    loaded_preds = predict_profile(loaded_bundle, test_df)

    pd.testing.assert_frame_equal(orig_preds, loaded_preds)


def test_predict_profile_validation_guards(memory_splits):
    """Verify that predict_profile raises ValueError on missing columns, NaN values, or negative inputs."""
    _, train_df, test_df = memory_splits
    bundle = fit_profile_model(train_df, k=4)

    # Missing column
    df_missing = test_df.drop(columns=["battery_mah"])
    with pytest.raises(ValueError, match="missing required clustering column"):
        predict_profile(bundle, df_missing)

    # NaN value
    df_nan = test_df.copy()
    df_nan.loc[0, "app_usage_min"] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        predict_profile(bundle, df_nan)

    # Negative value
    df_neg = test_df.copy()
    df_neg.loc[0, "data_mb"] = -100
    with pytest.raises(ValueError, match="negative values"):
        predict_profile(bundle, df_neg)


def test_percentile_report_range_and_extremes(memory_splits):
    """Verify percentile_report values reside in [0, 100] and values above max yield 100."""
    _, train_df, test_df = memory_splits
    eval_cols = ["app_usage_min", "battery_mah"]

    rep = percentile_report(train_df, test_df.head(20), eval_cols)
    assert (rep.values >= 0).all() and (rep.values <= 100).all()

    # Extreme value above maximum in training split
    above_max_df = pd.DataFrame([{
        "app_usage_min": train_df["app_usage_min"].max() + 500,
        "battery_mah": train_df["battery_mah"].max() + 1000,
    }])
    rep_max = percentile_report(train_df, above_max_df, eval_cols)
    assert rep_max["app_usage_min"].iloc[0] == 100.0
    assert rep_max["battery_mah"].iloc[0] == 100.0
