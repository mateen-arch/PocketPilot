"""
SHAP explanation utilities providing TreeExplainer computations, feature grouping,
global importance rankings, and local sample explanations for PocketPilot.
"""

from typing import Any
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

SUPPORTED_TREE_TYPES = (RandomForestRegressor, XGBRegressor, GradientBoostingRegressor)


class Contribution(tuple):
    """
    A 3-tuple representing (feature, value, shap_value) supporting both
    tuple indexing/unpacking and dictionary/attribute key access.
    """

    def __new__(cls, feature: str, value: Any, shap_value: float):
        return super().__new__(cls, (feature, value, shap_value))

    @property
    def feature(self) -> str:
        return self[0]

    @property
    def value(self) -> Any:
        return self[1]

    @property
    def shap_value(self) -> float:
        return self[2]

    def __getitem__(self, key: Any) -> Any:
        if key in ("feature", "name"):
            return self[0]
        if key in ("value", "input_value"):
            return self[1]
        if key in ("shap_value", "shap", "contribution"):
            return self[2]
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default


def get_final_estimator(pipeline: Pipeline) -> Any:
    """
    Extract and validate the final tree-based estimator from a scikit-learn Pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted regression pipeline containing a final 'model' step.

    Returns
    -------
    Any
        The fitted tree-based estimator (RandomForestRegressor, XGBRegressor, GradientBoostingRegressor).

    Raises
    ------
    ValueError
        If pipeline does not have a 'model' step or the estimator is not a supported tree-based model.
    """
    if not hasattr(pipeline, "named_steps") or "model" not in pipeline.named_steps:
        raise ValueError(
            "Pipeline must be a scikit-learn Pipeline with a final named step 'model'."
        )

    model = pipeline.named_steps["model"]
    if not isinstance(model, SUPPORTED_TREE_TYPES):
        raise ValueError(
            f"Estimator '{type(model).__name__}' is not a supported tree-based model for SHAP TreeExplainer. "
            f"Supported model types: {[t.__name__ for t in SUPPORTED_TREE_TYPES]}."
        )

    return model


def transform_for_explanation(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all pipeline preprocessing steps except the final estimator to raw input X.

    Returns a DataFrame with the transformed feature names from
    preprocessor.get_feature_names_out() and preserving X's index.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted regression pipeline.
    X : pd.DataFrame
        Raw input feature DataFrame.

    Returns
    -------
    pd.DataFrame
        Transformed feature DataFrame.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"Input X must be a pandas DataFrame, got {type(X)}.")

    # Slice pipeline up to the final model step
    pre_steps = pipeline[:-1]
    transformed = pre_steps.transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    preprocessor = pipeline.named_steps["preprocessor"]
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def compute_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Compute SHAP values using shap.TreeExplainer on the pipeline's final estimator.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted regression pipeline with a tree-based estimator.
    X : pd.DataFrame
        Raw input feature DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, float, pd.DataFrame]
        - shap_df: DataFrame of SHAP values matching X_transformed columns and index (in target units).
        - base_value: Expected value of the explainer (scalar float, in target units).
        - X_transformed: Transformed feature DataFrame.
    """
    estimator = get_final_estimator(pipeline)
    X_trans = transform_for_explanation(pipeline, X)

    explainer = shap.TreeExplainer(estimator)
    shap_vals = explainer(X_trans)

    # Determine base_value as scalar float
    if hasattr(shap_vals, "base_values"):
        base_arr = np.asarray(shap_vals.base_values)
        base_value = float(base_arr.flat[0])
    elif hasattr(explainer, "expected_value"):
        exp_arr = np.asarray(explainer.expected_value)
        base_value = float(exp_arr.flat[0])
    else:
        base_value = 0.0

    # Determine shap_arr
    if hasattr(shap_vals, "values"):
        shap_arr = np.asarray(shap_vals.values)
    else:
        shap_arr = np.asarray(shap_vals)

    shap_df = pd.DataFrame(shap_arr, columns=X_trans.columns, index=X_trans.index)
    return shap_df, base_value, X_trans


def group_device_features(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Sum one-hot device_model columns (names starting with 'device_model_') into one
    column named 'device_model', leaving other columns unchanged.

    Works on both signed SHAP values and absolute-value tables, and preserves row sums.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Input table or series with feature columns/indices.

    Returns
    -------
    pd.DataFrame or pd.Series
        Table with one consolidated 'device_model' column/entry.
    """
    if isinstance(df, pd.Series):
        device_idx = [idx for idx in df.index if str(idx).startswith("device_model_")]
        if not device_idx:
            return df.copy()
        other_idx = [idx for idx in df.index if not str(idx).startswith("device_model_")]
        res = df[other_idx].copy()
        res["device_model"] = df[device_idx].sum()
        return res

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Input must be a DataFrame or Series, got {type(df)}.")

    device_cols = [c for c in df.columns if str(c).startswith("device_model_")]
    if not device_cols:
        return df.copy()

    first_pos = df.columns.get_loc(device_cols[0])
    other_cols = [c for c in df.columns if not str(c).startswith("device_model_")]
    device_sum = df[device_cols].sum(axis=1)

    out = df[other_cols].copy()
    out.insert(first_pos, "device_model", device_sum)
    return out


def mean_abs_importance(shap_df: pd.DataFrame, grouped: bool = True) -> pd.Series:
    """
    Compute mean absolute SHAP importance per feature, sorted descending.

    Parameters
    ----------
    shap_df : pd.DataFrame
        DataFrame of SHAP values.
    grouped : bool, default=True
        If True, consolidates one-hot device columns into a single 'device_model' feature.

    Returns
    -------
    pd.Series
        Mean absolute SHAP value per feature, sorted descending.
    """
    target_df = group_device_features(shap_df) if grouped else shap_df
    importance = target_df.abs().mean(axis=0)
    return importance.sort_values(ascending=False)


def explain_one(
    pipeline: Pipeline,
    row_df: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Generate local prediction explanation for a single user row.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted regression pipeline with a tree-based model.
    row_df : pd.DataFrame
        Raw single-row DataFrame.
    top_k : int, default=5
        Number of top contributing features to return.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - 'prediction': float model prediction
        - 'base_value': float expected/baseline prediction
        - 'contributions': list of top_k Contribution(feature, input_value, shap_value)
          sorted by absolute SHAP value descending.
    """
    if not isinstance(row_df, pd.DataFrame) or len(row_df) != 1:
        raise ValueError(
            f"row_df must be a single-row pandas DataFrame, got {type(row_df)} with {len(row_df) if isinstance(row_df, pd.DataFrame) else 'N/A'} rows."
        )

    shap_df, base_val, X_trans = compute_shap(pipeline, row_df)
    pred = float(pipeline.predict(row_df)[0])
    grouped_shap = group_device_features(shap_df)

    contributions = []
    for col in grouped_shap.columns:
        shap_val = float(grouped_shap[col].iloc[0])
        if col in row_df.columns:
            val = row_df[col].iloc[0]
        elif col in X_trans.columns:
            val = X_trans[col].iloc[0]
        elif col == "device_model" and "device_model" in row_df.columns:
            val = row_df["device_model"].iloc[0]
        else:
            val = None
        contributions.append(Contribution(col, val, shap_val))

    # Sort descending by absolute SHAP value
    contributions.sort(key=lambda item: abs(item[2]), reverse=True)
    top_contribs = contributions[:top_k]

    return {
        "prediction": pred,
        "base_value": base_val,
        "contributions": top_contribs,
        "top_contributions": top_contribs,
    }
