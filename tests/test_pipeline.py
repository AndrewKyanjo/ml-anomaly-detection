"""Check training, inference, serialization, and binary metric boundaries."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import evaluate_predictions
from src.models import train_model
from src.schema import ROW_ID, TARGET


def test_training_keeps_labels_and_metadata_out_of_predictors(loan_frame):
    train = loan_frame.iloc[:32].copy()
    train[ROW_ID] = np.arange(len(train))
    original = train.copy(deep=True)
    estimator = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1)
    pipeline = train_model(train, estimator=estimator, scale=True)
    raw = loan_frame.iloc[32:].drop(columns=TARGET).copy()
    raw[ROW_ID] = np.arange(100, 100 + len(raw))
    preprocessing = pipeline.named_steps["preprocessing"]
    statistics = preprocessing.named_steps["impute"].statistics_.copy()
    means = preprocessing.named_steps["scale"].mean_.copy()

    prepared = pipeline[:-1].transform(raw)
    probabilities = pipeline.predict_proba(raw)

    pd.testing.assert_frame_equal(train, original)
    assert prepared.index.equals(raw.index)
    assert TARGET not in prepared and ROW_ID not in prepared
    assert TARGET not in pipeline.named_steps["model"].feature_names_in_
    assert ROW_ID not in pipeline.named_steps["model"].feature_names_in_
    assert probabilities.shape == (len(raw), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    raw["loan_amnt"] = 1e9
    pipeline.predict_proba(raw)
    np.testing.assert_equal(preprocessing.named_steps["impute"].statistics_, statistics)
    np.testing.assert_equal(preprocessing.named_steps["scale"].mean_, means)


def test_pipeline_rejects_target_during_prediction(loan_frame):
    pipeline = train_model(
        loan_frame,
        estimator=RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1),
    )
    with pytest.raises(ValueError, match=TARGET):
        pipeline.predict_proba(loan_frame)


@pytest.mark.parametrize("labels", [[0] * 40, [0, 1, 2, 0] * 10, [0, 1, None, 0] * 10])
def test_training_requires_complete_binary_labels_with_both_classes(loan_frame, labels):
    loan_frame[TARGET] = labels
    with pytest.raises(ValueError):
        train_model(loan_frame)


def test_saved_model_reproduces_probabilities(loan_frame, tmp_path):
    pipeline = train_model(
        loan_frame,
        estimator=RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1),
    )
    features = loan_frame.drop(columns=TARGET)
    expected = pipeline.predict_proba(features)
    path = tmp_path / "model.joblib"

    joblib.dump(pipeline, path)
    restored = joblib.load(path)

    np.testing.assert_allclose(restored.predict_proba(features), expected)


def test_evaluation_reports_metrics_at_requested_threshold():
    target = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.8, 0.9])

    metrics = evaluate_predictions(target, probabilities)

    assert set(metrics) == {
        "accuracy", "roc_auc", "average_precision", "balanced_accuracy",
        "precision", "recall", "f1",
    }
    assert all(value == pytest.approx(1.0) for value in metrics.values())
    assert evaluate_predictions(target, probabilities, threshold=0.85)["recall"] == 0.5


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([0.1, 0.3]),
        np.array([[0.1], [0.3], [0.8], [0.9]]),
        np.array([np.nan, 0.3, 0.8, 0.9]),
        np.array([np.inf, 0.3, 0.8, 0.9]),
        np.array([-0.1, 0.3, 0.8, 0.9]),
        np.array([1.1, 0.3, 0.8, 0.9]),
    ],
)
def test_evaluation_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        evaluate_predictions(pd.Series([0, 0, 1, 1]), probabilities)


@pytest.mark.parametrize("labels", [[0, 0, 0, 0], [0, 1, 2, 0], [0, 1, None, 0]])
def test_evaluation_requires_binary_labels_with_both_classes(labels):
    with pytest.raises(ValueError):
        evaluate_predictions(pd.Series(labels), np.array([0.1, 0.3, 0.8, 0.9]))


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_evaluation_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        evaluate_predictions(
            pd.Series([0, 0, 1, 1]), np.array([0.1, 0.3, 0.8, 0.9]), threshold=threshold
        )
