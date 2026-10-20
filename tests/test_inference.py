"""Verify saved classifiers serve raw loan records without training."""

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.inference import load_predictor
from src.models import train_model
from src.schema import FEATURE_COLUMNS, TARGET


@pytest.fixture
def artifact(loan_frame, tmp_path):
    bundle = {
        "pipelines": {
            "random_forest": train_model(loan_frame, n_estimators=5, n_jobs=1),
            "logistic_regression": train_model(loan_frame, model="logistic_regression"),
        },
        "selected_model": "random_forest",
        "threshold": 0.63,
        "feature_columns": list(FEATURE_COLUMNS),
    }
    path = tmp_path / "loan_models.joblib"
    joblib.dump(bundle, path)
    return bundle, path


def test_saved_models_match_direct_predictions_and_preserve_record_order(
    artifact, loan_frame, monkeypatch
):
    bundle, path = artifact
    predictor = load_predictor(path)
    raw = loan_frame.iloc[[19, 2, 35]].drop(columns=TARGET)
    original = raw.copy(deep=True)

    def forbid_training(*args, **kwargs):
        raise AssertionError("Serving must never fit a pipeline.")

    monkeypatch.setattr(Pipeline, "fit", forbid_training)
    for selection in (None, "logistic_regression"):
        name = selection or bundle["selected_model"]
        expected = bundle["pipelines"][name].predict_proba(raw)[:, 1]
        result = predictor.predict(raw.to_dict("records"), model=selection)

        assert result["model"] == name
        assert result["threshold"] == bundle["threshold"]
        np.testing.assert_allclose(
            [row["probability"] for row in result["predictions"]], expected
        )
        assert [row["label"] for row in result["predictions"]] == (
            expected >= bundle["threshold"]
        ).astype(int).tolist()

    assert raw.equals(original)


def test_unknown_or_empty_prediction_request_is_rejected(artifact, loan_frame):
    _, path = artifact
    predictor = load_predictor(path)
    records = loan_frame.iloc[:1].drop(columns=TARGET).to_dict("records")

    with pytest.raises(ValueError):
        predictor.predict(records, model="unknown")
    with pytest.raises(ValueError):
        predictor.predict([])


@pytest.mark.parametrize(
    "change",
    [
        {"pipelines": {}},
        {"pipelines": {"random_forest": RandomForestClassifier()}},
        {"selected_model": "missing"},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": float("nan")},
        {"feature_columns": ["loan_amnt"]},
    ],
)
def test_invalid_artifact_is_rejected_before_serving(artifact, change):
    bundle, path = artifact
    joblib.dump(bundle | change, path)

    with pytest.raises(ValueError):
        load_predictor(path)


def test_missing_artifact_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictor(tmp_path / "missing.joblib")


def test_corrupt_artifact_is_rejected(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_text("not a saved model", encoding="utf-8")

    with pytest.raises(ValueError):
        load_predictor(path)
