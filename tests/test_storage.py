"""SQLite import, prediction history, and verified feedback behavior."""

import csv
import hashlib
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.schema import FEATURE_COLUMNS, TARGET
from src.storage import (
    export_feedback,
    get_record,
    get_records,
    import_csv,
    initialize_database,
    list_predictions,
    query_records,
    record_outcome,
    save_predictions,
)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "state" / "records.sqlite3"
    initialize_database(path)
    return path


def _csv(tmp_path, frame, name="records.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _inputs(frame):
    return frame[list(FEATURE_COLUMNS)].astype(object).where(frame[list(FEATURE_COLUMNS)].notna(), None).to_dict("records")


def _result(size):
    return {"model": "logistic_regression", "threshold": 0.5,
            "predictions": [{"probability": 0.8, "label": 1} for _ in range(size)]}


def test_database_initialization_sets_version_and_preserves_records(database, tmp_path, loan_frame):
    import_csv(database, _csv(tmp_path, loan_frame.iloc[:2]))
    before = query_records(database)
    initialize_database(database)
    assert query_records(database) == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


def test_future_schema_version_is_rejected_without_changes(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE future_data (value TEXT)")
        connection.execute("INSERT INTO future_data VALUES ('preserve')")
        connection.execute("PRAGMA user_version = 2")
    with pytest.raises(ValueError, match="Unsupported database schema version"):
        initialize_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("SELECT value FROM future_data").fetchall() == [("preserve",)]
        assert connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall() == [("future_data",)]


def test_import_preserves_rows_nulls_and_is_content_idempotent(database, tmp_path, loan_frame):
    frame = loan_frame.iloc[:4].copy()
    frame.loc[frame.index[1], "dti"] = np.nan
    source = _csv(tmp_path, frame)
    imported = import_csv(database, source, chunksize=2)
    assert imported == {"dataset_id": 1, "rows": 4,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "already_imported": False}
    copy = tmp_path / "renamed.csv"
    copy.write_bytes(source.read_bytes())
    assert import_csv(database, copy)["already_imported"]
    page = query_records(database)
    assert page["total"] == 4
    assert [row["source_row"] for row in page["items"]] == [0, 1, 2, 3]
    assert page["items"][0]["term"] == " 36 months"
    assert page["items"][1]["dti"] is None
    assert page["items"][2]["emp_length"] is None
    assert get_record(database, 9999) is None


def test_query_filters_and_pagination(database, tmp_path, loan_frame):
    first = import_csv(database, _csv(tmp_path, loan_frame.iloc[:6]))
    import_csv(database, _csv(tmp_path, loan_frame.iloc[6:10], "second.csv"))
    page = query_records(database, loan_status=1, dataset_id=first["dataset_id"], limit=2, offset=1)
    assert page["total"] == 3
    assert [row["source_row"] for row in page["items"]] == [3, 5]
    assert get_record(database, page["items"][0]["id"]) == page["items"][0]


def test_record_batch_preserves_requested_order_and_duplicates(database, tmp_path, loan_frame):
    import_csv(database, _csv(tmp_path, loan_frame.iloc[:3]))
    rows = get_records(database, [3, 1, 3])
    assert [row["id"] for row in rows] == [3, 1, 3]
    with pytest.raises(LookupError, match="9999"):
        get_records(database, [1, 9999])
    with pytest.raises(ValueError):
        get_records(database, [])
    with pytest.raises(ValueError):
        get_records(database, ["1 OR 1=1"])


@pytest.mark.parametrize("changes", [
    {"limit": 1001}, {"limit": 0}, {"offset": -1}, {"offset": True},
    {"loan_status": "0 OR 1=1"}, {"loan_status": 2}, {"dataset_id": -1},
])
def test_queries_reject_invalid_filters(database, changes):
    with pytest.raises(ValueError):
        query_records(database, **changes)


@pytest.mark.parametrize("problem", ["target", "missing_target", "numeric", "infinite"])
def test_late_invalid_chunk_rolls_back_every_imported_row(database, tmp_path, loan_frame, problem):
    frame = loan_frame.iloc[:4].copy()
    if problem == "target":
        frame.loc[frame.index[-1], TARGET] = 2
    elif problem == "missing_target":
        frame.loc[frame.index[-1], TARGET] = np.nan
    elif problem == "infinite":
        frame.loc[frame.index[-1], "dti"] = np.inf
    else:
        frame["dti"] = frame["dti"].astype(object)
        frame.loc[frame.index[-1], "dti"] = "broken"
    with pytest.raises(ValueError):
        import_csv(database, _csv(tmp_path, frame), chunksize=2)
    assert query_records(database)["total"] == 0
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "empty"])
def test_import_rejects_wrong_schema_and_empty_data(database, tmp_path, loan_frame, problem):
    frame = loan_frame.iloc[:2].copy()
    if problem == "missing":
        frame = frame.drop(columns="term")
    elif problem == "extra":
        frame["untrusted"] = 1
    elif problem == "empty":
        frame = frame.iloc[:0]
    else:
        frame.columns = ["term" if name == "dti" else name for name in frame.columns]
    with pytest.raises(ValueError):
        import_csv(database, _csv(tmp_path, frame))
    assert query_records(database)["total"] == 0


def test_predictions_capture_model_inputs_linked_records_and_real_outcomes(database, tmp_path, loan_frame):
    import_csv(database, _csv(tmp_path, loan_frame.iloc[:2]))
    inputs = _inputs(loan_frame.iloc[:2])
    identifiers = save_predictions(database, inputs, _result(2), model_version="a" * 64, record_ids=[1, None])
    assert len(identifiers) == 2
    history = list_predictions(database)
    assert history["total"] == 2
    item = history["items"][0]
    assert item["record_id"] == 1
    assert item["inputs"] == inputs[0]
    assert item["model_version"] == "a" * 64
    assert item["observed_label"] is None
    assert "input_data" not in item
    updated = record_outcome(database, identifiers[0], 0)
    assert updated["label"] == 1 and updated["observed_label"] == 0
    assert updated["observed_at"] is not None
    assert list_predictions(database, observed=True)["total"] == 1
    assert list_predictions(database, observed=False)["total"] == 1
    assert record_outcome(database, 9999, 1) is None


def test_foreign_key_failure_rolls_back_entire_prediction_batch(database, loan_frame):
    with pytest.raises(sqlite3.IntegrityError):
        save_predictions(database, _inputs(loan_frame.iloc[:2]), _result(2),
                         model_version="b" * 64, record_ids=[None, 9999])
    assert list_predictions(database)["total"] == 0


@pytest.mark.parametrize("problem", ["alignment", "identifiers", "nonfinite", "label", "target", "version"])
def test_invalid_prediction_batches_are_never_partially_saved(database, loan_frame, problem):
    inputs, result, ids, version = _inputs(loan_frame.iloc[:2]), _result(2), None, "b" * 64
    if problem == "alignment":
        result["predictions"].pop()
    elif problem == "identifiers":
        ids = [None]
    elif problem == "nonfinite":
        inputs[1]["annual_inc"] = float("nan")
    elif problem == "label":
        result["predictions"][1]["label"] = 0
    elif problem == "target":
        inputs[1][TARGET] = 0
    else:
        version = "unversioned"
    with pytest.raises(ValueError):
        save_predictions(database, inputs, result, model_version=version, record_ids=ids)
    assert list_predictions(database)["total"] == 0


def test_feedback_export_contains_only_verified_targets_and_raw_features(database, tmp_path, loan_frame):
    identifiers = save_predictions(database, _inputs(loan_frame.iloc[:3]), _result(3), model_version="c" * 64)
    destination = tmp_path / "feedback" / "verified.csv"
    assert export_feedback(database, destination) == 0
    record_outcome(database, identifiers[1], 0)
    assert export_feedback(database, destination) == 1
    with destination.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [*FEATURE_COLUMNS, TARGET]
        rows = list(reader)
    assert len(rows) == 1 and rows[0][TARGET] == "0"
    assert rows[0]["term"] == " 60 months"
    assert "model" not in rows[0]
    with pytest.raises(ValueError):
        export_feedback(database, database)


@pytest.mark.parametrize("outcome", [2, -1, "1", True])
def test_outcome_rejects_nonbinary_values(database, outcome):
    with pytest.raises(ValueError):
        record_outcome(database, 1, outcome)
