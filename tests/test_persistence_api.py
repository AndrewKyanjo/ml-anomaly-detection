"""Exercise durable prediction history and recorded-data inference over HTTP."""

import hashlib

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.models import train_model
from src.schema import FEATURE_COLUMNS, TARGET
from src.storage import export_feedback, import_csv


@pytest.fixture
def persisted_setup(tmp_path, loan_frame):
    bundle = {
        "pipelines": {
            "random_forest": train_model(loan_frame, n_estimators=5, n_jobs=1),
            "logistic_regression": train_model(loan_frame, model="logistic_regression"),
        },
        "selected_model": "random_forest",
        "threshold": 0.5,
        "feature_columns": list(FEATURE_COLUMNS),
    }
    artifact = tmp_path / "models.joblib"
    joblib.dump(bundle, artifact)
    source = tmp_path / "data.csv"
    loan_frame.iloc[:8].to_csv(source, index=False)
    database = tmp_path / "state" / "records.sqlite3"
    import_csv(database, source, chunksize=3)
    return bundle, artifact, database


def test_raw_predictions_survive_restart_with_model_provenance(persisted_setup, loan_frame):
    _, artifact, database = persisted_setup
    records = loan_frame.iloc[[3, 0]][list(FEATURE_COLUMNS)].to_dict("records")
    with TestClient(create_app(model_path=artifact, database_path=database)) as client:
        response = client.post("/predict", json={"records": records})
        assert response.status_code == 200
        result = response.json()
        assert result["model_version"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        identifiers = [prediction["id"] for prediction in result["predictions"]]
        assert len(set(identifiers)) == 2

    with TestClient(create_app(model_path=artifact, database_path=database)) as restarted:
        history = restarted.get("/predictions").json()
        assert history["total"] == 2
        assert [item["id"] for item in history["items"]] == identifiers
        for record, item, prediction in zip(records, history["items"], result["predictions"]):
            assert item["inputs"] == record
            assert item["record_id"] is None
            assert item["observed_label"] is None
            assert item["model_version"] == result["model_version"]
            assert item["probability"] == prediction["probability"]
            assert item["label"] == prediction["label"]


def test_stored_record_queries_return_filtered_pages_and_missing_ids(persisted_setup):
    _, artifact, database = persisted_setup
    with TestClient(create_app(model_path=artifact, database_path=database)) as client:
        response = client.get("/records", params={"loan_status": 1, "dataset_id": 1, "limit": 2, "offset": 1})
        assert response.status_code == 200
        page = response.json()
        assert page["total"] == 4
        assert [row["source_row"] for row in page["items"]] == [3, 5]
        for row in page["items"]:
            assert row[TARGET] == 1
            assert client.get(f"/records/{row['id']}").json() == row
        assert client.get("/records/99999").status_code == 404


def test_predict_stored_rows_preserves_order_models_and_excludes_metadata(persisted_setup, loan_frame):
    bundle, artifact, database = persisted_setup
    requested = [4, 1, 4]
    raw = loan_frame.iloc[[3, 0, 3]][list(FEATURE_COLUMNS)]
    expected = bundle["pipelines"]["logistic_regression"].predict_proba(raw)[:, 1]
    with TestClient(create_app(model_path=artifact, database_path=database)) as client:
        response = client.post("/predict/records", json={"record_ids": requested, "model": "logistic_regression"})
        assert response.status_code == 200
        result = response.json()
        assert result["model"] == "logistic_regression"
        np.testing.assert_allclose([row["probability"] for row in result["predictions"]], expected)
        history = client.get("/predictions").json()["items"]
        assert [item["record_id"] for item in history] == requested
        assert [item["id"] for item in history] == [row["id"] for row in result["predictions"]]
        assert all(set(item["inputs"]) == set(FEATURE_COLUMNS) for item in history)
        assert history[0]["inputs"] == history[2]["inputs"]
        assert history[0]["inputs"]["term"] == " 60 months"


def test_missing_records_and_unavailable_models_leave_no_history(persisted_setup, loan_frame):
    _, artifact, database = persisted_setup
    raw = loan_frame.iloc[:1][list(FEATURE_COLUMNS)].to_dict("records")
    with TestClient(create_app(model_path=artifact, database_path=database)) as client:
        assert client.post("/predict/records", json={"record_ids": [1, 99999]}).status_code == 404
        assert client.post("/predict/records", json={"record_ids": [1], "model": "xgboost"}).status_code == 400
        assert client.post("/predict", json={"records": raw, "model": "xgboost"}).status_code == 400
        assert client.get("/predictions").json()["total"] == 0


def test_verified_outcomes_are_separate_from_predictions_and_exported(persisted_setup, tmp_path):
    _, artifact, database = persisted_setup
    with TestClient(create_app(model_path=artifact, database_path=database)) as client:
        result = client.post("/predict/records", json={"record_ids": [1, 2]}).json()
        prediction = result["predictions"][0]
        outcome = 1 - prediction["label"]
        response = client.patch(f"/predictions/{prediction['id']}/outcome", json={"observed_label": outcome})
        assert response.status_code == 200
        updated = response.json()
        assert updated["label"] == prediction["label"]
        assert updated["observed_label"] == outcome
        assert updated["observed_at"] is not None
        assert client.get("/predictions", params={"observed": True}).json()["total"] == 1
        assert client.get("/predictions", params={"observed": False}).json()["total"] == 1
        assert client.patch("/predictions/99999/outcome", json={"observed_label": 1}).status_code == 404
    feedback = tmp_path / "feedback.csv"
    assert export_feedback(database, feedback) == 1
    exported = pd.read_csv(feedback)
    assert list(exported.columns) == [*FEATURE_COLUMNS, TARGET]
    assert exported[TARGET].tolist() == [outcome]


@pytest.mark.parametrize("route,parameters", [
    ("/records", {"limit": 0}), ("/records", {"limit": 1001}),
    ("/records", {"offset": -1}), ("/records", {"loan_status": 2}),
    ("/records", {"dataset_id": 0}), ("/predictions", {"limit": 0}),
    ("/predictions", {"offset": -1}), ("/predictions", {"observed": "unknown"}),
])
def test_invalid_query_parameters_return_422(persisted_setup, route, parameters):
    _, artifact, database = persisted_setup
    with TestClient(create_app(model_path=artifact, database_path=database)) as client:
        assert client.get(route, params=parameters).status_code == 422


@pytest.mark.parametrize("payload", [
    {"record_ids": []}, {"record_ids": [0]}, {"record_ids": [-1]},
    {"record_ids": [1], TARGET: 1}, {"record_ids": [1], "model": "unsupported"},
])
def test_stored_prediction_request_validation_creates_no_history(persisted_setup, payload):
    _, artifact, database = persisted_setup
    with TestClient(create_app(model_path=artifact, database_path=database)) as client:
        assert client.post("/predict/records", json=payload).status_code == 422
        assert client.get("/predictions").json()["total"] == 0


@pytest.mark.parametrize("payload", [
    {"observed_label": 2}, {"observed_label": -1},
    {"observed_label": "unknown"}, {"observed_label": 1, "extra": 1}, {},
])
def test_invalid_feedback_is_rejected(persisted_setup, payload):
    _, artifact, database = persisted_setup
    with TestClient(create_app(model_path=artifact, database_path=database)) as client:
        assert client.patch("/predictions/1/outcome", json=payload).status_code == 422


def test_database_unavailable_prevents_unrecorded_predictions(persisted_setup, tmp_path, loan_frame):
    _, artifact, _ = persisted_setup
    records = loan_frame.iloc[:1][list(FEATURE_COLUMNS)].to_dict("records")
    with TestClient(create_app(model_path=artifact, database_path=tmp_path)) as client:
        assert client.get("/health").status_code == 503
        for response in (client.get("/records"), client.post("/predict", json={"records": records})):
            assert response.status_code == 503
            assert str(tmp_path) not in response.text
