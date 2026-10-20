"""Train a complete pipeline from raw loan records."""

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from .features import build_features
from .preprocessing import make_preprocessor
from .schema import TARGET


def create_model(name: str = "random_forest", **parameters) -> ClassifierMixin:
    """Create a random forest, logistic regression, or XGBoost classifier."""
    if name == "random_forest":
        defaults = {
            "n_estimators": 100, "max_depth": 12, "min_samples_leaf": 10,
            "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": 42,
        }
        return RandomForestClassifier(**(defaults | parameters))
    if name == "logistic_regression":
        defaults = {
            "max_iter": 2000, "solver": "lbfgs", "class_weight": "balanced",
            "random_state": 42,
        }
        return LogisticRegression(**(defaults | parameters))
    if name == "xgboost":
        defaults = {
            "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "n_jobs": -1, "random_state": 42,
        }
        return XGBClassifier(**(defaults | parameters))
    raise ValueError(
        f"Unknown model {name!r}. Choose random_forest, logistic_regression, or xgboost."
    )


def train_model(
    frame: pd.DataFrame,
    estimator: ClassifierMixin | None = None,
    *,
    model: str = "random_forest",
    scale: bool | None = None,
    **model_parameters,
) -> Pipeline:
    """Fit feature creation, preprocessing, and a classifier on training rows.

    Select an algorithm by model name and override its constructor parameters,
    or pass a custom estimator. Logistic regression uses scaling by default.
    The returned pipeline accepts raw predictors for predict/predict_proba.
    """
    if not frame.columns.is_unique or TARGET not in frame:
        raise ValueError(f"Provide unique columns including {TARGET!r}.")
    target = frame[TARGET]
    if target.isna().any() or not target.isin([0, 1]).all() or target.nunique() != 2:
        raise ValueError("Training labels must contain both binary classes 0 and 1.")
    if estimator is not None and (model != "random_forest" or model_parameters):
        raise ValueError("Choose a model name and parameters, or pass a custom estimator.")
    if estimator is None:
        estimator = create_model(model, **model_parameters)
    if scale is None:
        scale = isinstance(estimator, LogisticRegression)
    pipeline = Pipeline(
        [
            ("features", FunctionTransformer(build_features)),
            ("preprocessing", make_preprocessor(scale=scale)),
            ("model", estimator),
        ]
    )
    return pipeline.fit(frame.drop(columns=TARGET), target)
