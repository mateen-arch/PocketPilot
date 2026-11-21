"""Pipeline tracing tool for PocketPilot model inference.

Inspects each stage of an inference scikit-learn Pipeline step-by-step,
recording intermediate data types, shapes, feature names, and transformed values
to facilitate transparency, verification, and viva examination explanations.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def trace_pipeline(pipeline: Pipeline, X_raw: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Step sequentially through a fitted scikit-learn Pipeline and trace intermediate stages.

    For all intermediate steps (except the final estimator), executes `transform`.
    For the final step, executes `predict`.
    Verifies that the final traced prediction strictly matches `pipeline.predict(X_raw)`.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted scikit-learn pipeline (e.g. Feature Engineering -> Preprocessor -> Regressor).
    X_raw : pd.DataFrame
        Raw input feature DataFrame. The input is never modified.

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries, one per pipeline step, containing:
        - 'step_name': name assigned to the step in the pipeline
        - 'class_name': Python class name of the transformer or estimator
        - 'output_type': class name of the output (e.g., 'DataFrame', 'ndarray')
        - 'output_shape': dimensional shape tuple of the step output
        - 'feature_names': list of string feature names if available, else None
        - 'output_row': list of dicts (rounded to 4 decimal places) representing the first row

    Raises
    ------
    AssertionError
        If the traced final prediction does not equal `pipeline.predict(X_raw)`
        within a 1e-6 relative tolerance.
    """
    # Guard against mutating input DataFrame
    curr = X_raw.copy()
    trace_records: list[dict[str, Any]] = []

    steps = pipeline.steps
    n_steps = len(steps)

    for idx, (step_name, step_obj) in enumerate(steps):
        is_last = idx == n_steps - 1
        class_name = step_obj.__class__.__name__

        if not is_last:
            # Intermediate transformation step
            curr = step_obj.transform(curr)
            out_type = type(curr).__name__

            # Handle sparse matrix conversions if encountered
            if hasattr(curr, "toarray"):
                arr = curr.toarray()
            elif isinstance(curr, np.ndarray):
                arr = curr
            else:
                arr = None

            out_shape = tuple(curr.shape)

            # Extract feature names when supported
            feature_names: list[str] | None = None
            if hasattr(step_obj, "get_feature_names_out"):
                try:
                    feature_names = [str(f) for f in step_obj.get_feature_names_out()]
                except Exception:
                    feature_names = None
            elif isinstance(curr, pd.DataFrame):
                feature_names = [str(c) for c in curr.columns]

            # Format first row as list of dicts rounded to 4 decimals
            if isinstance(curr, pd.DataFrame):
                row_dict = curr.iloc[0:1].round(4).to_dict(orient="records")
            elif arr is not None:
                first_row_vals = [round(float(v), 4) for v in arr[0]]
                if feature_names and len(feature_names) == len(first_row_vals):
                    row_dict = [{k: v for k, v in zip(feature_names, first_row_vals)}]
                else:
                    row_dict = [{f"feature_{i}": v for i, v in enumerate(first_row_vals)}]
            else:
                row_dict = [{"value": round(float(curr[0]), 4)}]

            trace_records.append({
                "step_name": step_name,
                "class_name": class_name,
                "output_type": out_type,
                "output_shape": out_shape,
                "feature_names": feature_names,
                "output_row": row_dict,
            })
        else:
            # Final predictor step
            y_pred = step_obj.predict(curr)
            out_type = type(y_pred).__name__
            out_shape = tuple(getattr(y_pred, "shape", (len(y_pred),)))
            pred_val = round(float(y_pred[0]), 4)
            row_dict = [{"prediction": pred_val}]

            trace_records.append({
                "step_name": step_name,
                "class_name": class_name,
                "output_type": out_type,
                "output_shape": out_shape,
                "feature_names": ["prediction"],
                "output_row": row_dict,
            })

            # Parity verification against direct pipeline.predict
            expected_pred = pipeline.predict(X_raw)
            if not np.isclose(y_pred, expected_pred, rtol=1e-6).all():
                raise AssertionError(
                    f"Traced pipeline prediction {y_pred} does not match direct pipeline.predict {expected_pred}"
                )

    return trace_records
