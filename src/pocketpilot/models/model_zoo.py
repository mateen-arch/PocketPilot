"""
Model zoo defining candidate regression models, pipelines, and hyperparameter search grids.
"""

from typing import Any
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBRegressor

from pocketpilot.config import RANDOM_SEED
from pocketpilot.features.engineering import add_forecast_features
from pocketpilot.features.input_modes import get_feature_columns
from pocketpilot.features.preprocessing import build_preprocessor


def get_candidate_models(
    task: str,
    fast: bool = False,
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """
    Construct model pipelines and corresponding hyperparameter grids for a regression task.

    Pipeline Architecture:
    - For tasks 'battery' and 'data':
      1. 'engineer': FunctionTransformer(add_forecast_features) to derive 'app_share_of_screen'
         directly from raw inputs ['device_model', 'app_usage_min', 'screen_on_hours', 'apps_installed'].
      2. 'preprocessor': ColumnTransformer (OneHotEncoder for device_model, StandardScaler or passthrough for numeric).
      3. 'model': Regression estimator.
    - For task 'screen_time':
      1. 'preprocessor': ColumnTransformer on raw inputs ['device_model', 'app_usage_min', 'apps_installed'].
         (Note: FunctionTransformer is omitted because screen_on_hours is the target and cannot exist at inference).
      2. 'model': Regression estimator.

    Parameters
    ----------
    task : str
        Target regression task: 'battery', 'data', or 'screen_time'.
    fast : bool, default=False
        If True, returns single-value hyperparameter grids to accelerate unit tests and validation runs.

    Returns
    -------
    dict[str, tuple[Pipeline, dict[str, list[Any]]]]
        Mapping of model_name -> (Pipeline, param_grid).
    """
    feature_cols = get_feature_columns(task, mode="forecast")
    needs_engineering = task in {"battery", "data"}

    def _assemble_pipeline(estimator: Any, scale_numeric: bool) -> Pipeline:
        preprocessor = build_preprocessor(feature_cols=feature_cols, scale_numeric=scale_numeric)
        steps = []
        if needs_engineering:
            steps.append(("engineer", FunctionTransformer(add_forecast_features)))
        steps.append(("preprocessor", preprocessor))
        steps.append(("model", estimator))
        return Pipeline(steps)

    # 1. Linear Regression baseline (no search grid, standardizes numeric features)
    pipe_linear = _assemble_pipeline(LinearRegression(), scale_numeric=True)
    grid_linear: dict[str, list[Any]] = {}

    # 2. Random Forest Regressor (tree model, no numeric scaling)
    rf_est = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
    pipe_rf = _assemble_pipeline(rf_est, scale_numeric=False)
    if fast:
        grid_rf = {
            "model__n_estimators": [50],
            "model__max_depth": [4],
            "model__min_samples_leaf": [1],
        }
    else:
        grid_rf = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 3, 5],
        }

    # 3. XGBoost Regressor (gradient boosting, no numeric scaling)
    xgb_est = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1)
    pipe_xgb = _assemble_pipeline(xgb_est, scale_numeric=False)
    if fast:
        grid_xgb = {
            "model__n_estimators": [50],
            "model__max_depth": [2],
            "model__learning_rate": [0.1],
        }
    else:
        grid_xgb = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.1],
        }

    # 4. Multi-Layer Perceptron (neural net, standardized input and standardized target)
    mlp_base = MLPRegressor(random_state=RANDOM_SEED, max_iter=2000, early_stopping=True)
    mlp_ttr = TransformedTargetRegressor(regressor=mlp_base, transformer=StandardScaler())
    pipe_mlp = _assemble_pipeline(mlp_ttr, scale_numeric=True)
    if fast:
        grid_mlp = {
            "model__regressor__hidden_layer_sizes": [(32,)],
            "model__regressor__alpha": [1e-4],
        }
    else:
        grid_mlp = {
            "model__regressor__hidden_layer_sizes": [(32,), (64, 32)],
            "model__regressor__alpha": [1e-4, 1e-2],
        }

    # 5. Reference Mean Baseline (never selected for deployment)
    dummy_est = DummyRegressor(strategy="mean")
    pipe_dummy = _assemble_pipeline(dummy_est, scale_numeric=False)
    grid_dummy: dict[str, list[Any]] = {}

    candidates = {
        "linear": (pipe_linear, grid_linear),
        "random_forest": (pipe_rf, grid_rf),
        "xgboost": (pipe_xgb, grid_xgb),
        "mlp": (pipe_mlp, grid_mlp),
        "mean_baseline": (pipe_dummy, grid_dummy),
    }

    return candidates
