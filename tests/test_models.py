"""Verify selectable classifiers work with the complete raw-loan pipeline."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models import create_model, train_model
from src.schema import TARGET


@pytest.mark.parametrize(
    "name,parameters,expected_type",
    [
        ("random_forest", {"n_estimators": 5, "n_jobs": 1}, RandomForestClassifier),
        ("logistic_regression", {"C": 0.25}, LogisticRegression),
        (
            "xgboost",
            {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
            XGBClassifier,
        ),
    ],
)
def test_selected_model_fits_and_predicts_without_refitting(
    loan_frame, name, parameters, expected_type
):
    train = loan_frame.iloc[:32].copy()
    original = train.copy(deep=True)
    pipeline = train_model(train, model=name, **parameters)
    estimator = pipeline.named_steps["model"]
    preprocessing = pipeline.named_steps["preprocessing"]
    imputation_statistics = preprocessing.named_steps["impute"].statistics_.copy()
    scaler = preprocessing.named_steps.get("scale")
    scaling_mean = None if scaler is None else scaler.mean_.copy()
    raw = loan_frame.iloc[[39, 32, 37, 34]].drop(columns=TARGET).copy()
    raw.loc[raw.index[0], "annual_inc"] = np.nan
    raw.loc[raw.index[1], "loan_amnt"] = 1e8

    prepared = pipeline[:-1].transform(raw)
    probabilities = pipeline.predict_proba(raw)

    assert isinstance(estimator, expected_type)
    assert is_classifier(estimator)
    assert clone(estimator).get_params() == estimator.get_params()
    for parameter, value in parameters.items():
        assert estimator.get_params()[parameter] == value
    pd.testing.assert_frame_equal(train, original)
    assert prepared.index.equals(raw.index)
    assert probabilities.shape == (len(raw), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
    np.testing.assert_array_equal(estimator.classes_, [0, 1])
    np.testing.assert_array_equal(
        preprocessing.named_steps["impute"].statistics_, imputation_statistics
    )
    if scaler is not None:
        np.testing.assert_array_equal(scaler.mean_, scaling_mean)
    assert (scaler is not None) == (name == "logistic_regression")


@pytest.mark.parametrize(
    "name,scale,parameters",
    [
        ("logistic_regression", False, {"max_iter": 3000}),
        ("xgboost", True, {"n_estimators": 5, "n_jobs": 1}),
    ],
)
def test_scaling_can_be_overridden(loan_frame, name, scale, parameters):
    pipeline = train_model(loan_frame, model=name, scale=scale, **parameters)

    assert ("scale" in pipeline.named_steps["preprocessing"].named_steps) == scale


def test_create_model_accepts_classifier_parameters():
    estimator = create_model("logistic_regression", C=0.5, max_iter=250)

    assert isinstance(estimator, LogisticRegression)
    assert estimator.C == 0.5
    assert estimator.max_iter == 250


def test_unknown_model_name_is_rejected(loan_frame):
    with pytest.raises(ValueError, match="unsupported_model"):
        create_model("unsupported_model")
    with pytest.raises(ValueError, match="unsupported_model"):
        train_model(loan_frame, model="unsupported_model")


def test_custom_logistic_regression_uses_automatic_scaling(loan_frame):
    estimator = LogisticRegression(C=0.5, max_iter=500)
    pipeline = train_model(loan_frame, estimator=estimator)

    assert pipeline.named_steps["model"] is estimator
    assert "scale" in pipeline.named_steps["preprocessing"].named_steps


@pytest.mark.parametrize("selection", [{"model": "xgboost"}, {"max_depth": 3}])
def test_custom_estimator_rejects_conflicting_model_selection(loan_frame, selection):
    with pytest.raises(ValueError):
        train_model(loan_frame, estimator=LogisticRegression(), **selection)


def test_xgboost_serialization_preserves_raw_record_predictions(loan_frame, tmp_path):
    pipeline = train_model(
        loan_frame,
        model="xgboost",
        n_estimators=5,
        max_depth=2,
        n_jobs=1,
    )
    raw = loan_frame.drop(columns=TARGET)
    expected = pipeline.predict_proba(raw)
    path = tmp_path / "xgboost.joblib"

    joblib.dump(pipeline, path)
    restored = joblib.load(path)

    np.testing.assert_allclose(restored.predict_proba(raw), expected)
