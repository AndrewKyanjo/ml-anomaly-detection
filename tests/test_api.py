"""Exercise HTTP validation and predictions with real fitted pipelines."""

import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
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
        "threshold": 0.5,
        "feature_columns": list(FEATURE_COLUMNS),
    }
    path = tmp_path / "loan_models.joblib"
    joblib.dump(bundle, path)
    return bundle, path


def test_health_and_batch_model_selection(artifact, loan_frame):
    bundle, path = artifact
    raw = loan_frame.iloc[[7, 0, 11]].drop(columns=TARGET)

    with TestClient(create_app(model_path=path, database_path=path.with_suffix(".sqlite3"))) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert set(health.json()["models"]) == set(bundle["pipelines"])
        assert health.json()["default_model"] == "random_forest"

        for selection in (None, "logistic_regression"):
            payload = {"records": raw.to_dict("records")}
            if selection is not None:
                payload["model"] = selection
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            result = response.json()
            name = selection or "random_forest"
            expected = bundle["pipelines"][name].predict_proba(raw)[:, 1]

            assert result["model"] == name
            assert result["threshold"] == 0.5
            np.testing.assert_allclose(
                [row["probability"] for row in result["predictions"]], expected
            )
            assert [row["label"] for row in result["predictions"]] == (
                expected >= 0.5
            ).astype(int).tolist()


@pytest.mark.parametrize(
    "case", ["missing", "target", "extra", "term", "employment", "empty", "model"]
)
def test_invalid_requests_return_422(artifact, loan_frame, case):
    _, path = artifact
    record = loan_frame.iloc[0].drop(labels=TARGET).to_dict()
    payload = {"records": [record]}
    if case == "missing":
        record.pop("loan_amnt")
    elif case == "target":
        record[TARGET] = 1
    elif case == "extra":
        record["unrecognized"] = 1
    elif case == "term":
        record["term"] = "48 months"
    elif case == "employment":
        record["emp_length"] = "30 years"
    elif case == "empty":
        payload["records"] = []
    else:
        payload["model"] = "unsupported_model"

    with TestClient(create_app(model_path=path, database_path=path.with_suffix(".sqlite3"))) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_nonfinite_number_is_rejected(artifact, loan_frame):
    _, path = artifact
    record = loan_frame.iloc[0].drop(labels=TARGET).to_dict()
    record["annual_inc"] = float("nan")

    with TestClient(create_app(model_path=path, database_path=path.with_suffix(".sqlite3"))) as client:
        response = client.post(
            "/predict",
            content=json.dumps({"records": [record]}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


def test_missing_model_keeps_health_available_and_rejects_prediction(tmp_path, loan_frame):
    records = loan_frame.iloc[:1].drop(columns=TARGET).to_dict("records")

    with TestClient(create_app(model_path=tmp_path / "missing.joblib", database_path=tmp_path / "app.sqlite3")) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json() == {
            "status": "unavailable", "models": [], "default_model": None
        }
        prediction = client.post("/predict", json={"records": records})
        assert prediction.status_code == 503
        assert prediction.json()["detail"]
