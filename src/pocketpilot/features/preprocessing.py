"""
Preprocessing pipelines ensuring identical transformations during training and inference.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    feature_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Construct an unfitted scikit-learn ColumnTransformer for categorical encoding and numeric scaling.

    Design & Leakage Prevention:
    This preprocessor is completely unfitted. It is later placed inside one sklearn Pipeline
    together with the model and fitted on the TRAINING set only, so the identical transform
    is applied at inference in the Streamlit app. This guarantees that test data statistics
    never influence feature transformations.

    Parameters
    ----------
    feature_cols : list[str]
        List of feature column names to be included in the transformer.
    scale_numeric : bool, default=True
        Whether to scale numeric features using StandardScaler. If False, numeric
        features pass through untransformed (useful for tree-based models like Random Forest/XGBoost).

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer instance configured with:
        - OneHotEncoder(handle_unknown="ignore", sparse_output=False) for 'device_model' (if present)
        - StandardScaler() or 'passthrough' for numeric features
        - remainder="drop"
        - verbose_feature_names_out=False
    """
    transformers = []

    cat_cols = [c for c in feature_cols if c == "device_model"]
    num_cols = [c for c in feature_cols if c != "device_model"]

    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )

    if num_cols:
        transformers.append(
            (
                "num",
                StandardScaler() if scale_numeric else "passthrough",
                num_cols,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor
