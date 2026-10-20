"""Store source records, predictions, and verified outcomes in SQLite."""

import csv
import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from ml_gearbox import audit as ga

from .schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET

_COLUMNS = (*FEATURE_COLUMNS, TARGET)
_SELECT_RECORD = "SELECT * FROM records"


@contextmanager
def _connection(path: str | Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_database(path: str | Path) -> None:
    """Create tables and enable WAL for concurrent readers."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    columns = ",\n".join(
        f'"{name}" ' + ("TEXT" if name in CATEGORICAL_COLUMNS else "REAL")
        for name in FEATURE_COLUMNS
    )
    with _connection(path) as connection:
        if connection.execute("PRAGMA user_version").fetchone()[0] > 1:
            raise ValueError("Unsupported database schema version")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(f"""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY, source_name TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE, row_count INTEGER NOT NULL,
                imported_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL REFERENCES datasets(id),
                source_row INTEGER NOT NULL, {columns},
                {TARGET} INTEGER NOT NULL CHECK ({TARGET} IN (0, 1)),
                UNIQUE (dataset_id, source_row)
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY, record_id INTEGER REFERENCES records(id),
                model TEXT NOT NULL, model_version TEXT NOT NULL,
                probability REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
                label INTEGER NOT NULL CHECK (label IN (0, 1)),
                threshold REAL NOT NULL CHECK (threshold BETWEEN 0 AND 1),
                input_data TEXT NOT NULL, created_at TEXT NOT NULL,
                observed_label INTEGER CHECK (observed_label IN (0, 1)),
                observed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS records_status ON records ({TARGET});
            CREATE INDEX IF NOT EXISTS predictions_observed ON predictions (observed_label);
            PRAGMA user_version = 1;
        """)


def import_csv(database_path: str | Path, csv_path: str | Path, *, chunksize: int = 10000) -> dict:
    """Import a CSV atomically; identical file contents are imported only once."""
    _positive(chunksize, "chunksize")
    source = Path(csv_path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    with source.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(_COLUMNS) or set(header) != set(_COLUMNS):
        raise ValueError(f"CSV columns must match {list(_COLUMNS)}.")
    initialize_database(database_path)
    with _connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT id, row_count FROM datasets WHERE sha256 = ?", (checksum,)
        ).fetchone()
        if existing:
            return {"dataset_id": existing["id"], "rows": existing["row_count"],
                    "sha256": checksum, "already_imported": True}
        dataset_id = connection.execute(
            "INSERT INTO datasets (source_name, sha256, row_count, imported_at) VALUES (?, ?, 0, ?)",
            (source.name, checksum, _now()),
        ).lastrowid
        names = ", ".join(f'"{column}"' for column in _COLUMNS)
        placeholders = ", ".join("?" for _ in range(len(_COLUMNS) + 2))
        count = 0
        for frame in pd.read_csv(source, chunksize=chunksize, dtype={c: object for c in CATEGORICAL_COLUMNS}):
            for column in (*NUMERIC_COLUMNS, TARGET):
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            if np.isinf(frame[[*NUMERIC_COLUMNS, TARGET]].to_numpy(dtype=float)).any():
                raise ValueError("CSV numeric values must be finite or missing.")
            target_check = ga.range_validation(frame, {TARGET: {"allowed": [0, 1]}})
            if frame[TARGET].isna().any() or target_check["violation_count"].sum():
                raise ValueError("CSV targets must be zero or one without missing values.")
            values = frame.loc[:, _COLUMNS].astype(object).where(frame.loc[:, _COLUMNS].notna(), None)
            rows = ((dataset_id, count + i, *row) for i, row in enumerate(values.itertuples(index=False, name=None)))
            connection.executemany(
                f"INSERT INTO records (dataset_id, source_row, {names}) VALUES ({placeholders})", rows
            )
            count += len(frame)
        if not count:
            raise ValueError("CSV must contain at least one record.")
        connection.execute("UPDATE datasets SET row_count = ? WHERE id = ?", (count, dataset_id))
    return {"dataset_id": dataset_id, "rows": count, "sha256": checksum, "already_imported": False}


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _binary(value: int, name: str) -> None:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be zero or one.")


def _page(limit: int, offset: int) -> None:
    _positive(limit, "limit")
    if limit > 1000 or type(offset) is not int or offset < 0:
        raise ValueError("Use limit between 1 and 1000 and a nonnegative offset.")


def query_records(path: str | Path, *, limit: int = 100, offset: int = 0,
                  loan_status: int | None = None, dataset_id: int | None = None) -> dict:
    """Return a bounded page of source records and its filtered total."""
    _page(limit, offset)
    conditions, parameters = [], []
    if loan_status is not None:
        _binary(loan_status, "loan_status")
        conditions.append(f"{TARGET} = ?")
        parameters.append(loan_status)
    if dataset_id is not None:
        _positive(dataset_id, "dataset_id")
        conditions.append("dataset_id = ?")
        parameters.append(dataset_id)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    with _connection(path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM records" + where, parameters).fetchone()[0]
        items = connection.execute(
            _SELECT_RECORD + where + " ORDER BY id LIMIT ? OFFSET ?", [*parameters, limit, offset]
        ).fetchall()
    return {"total": total, "items": [dict(row) for row in items]}


def get_record(path: str | Path, record_id: int) -> dict | None:
    """Retrieve a source record by its database identifier."""
    _positive(record_id, "record_id")
    with _connection(path) as connection:
        row = connection.execute(_SELECT_RECORD + " WHERE id = ?", (record_id,)).fetchone()
    return dict(row) if row else None


def get_records(path: str | Path, record_ids: list[int]) -> list[dict]:
    """Retrieve a batch in requested order, rejecting any missing identifier."""
    if not record_ids or len(record_ids) > 1000:
        raise ValueError("Provide between 1 and 1000 record identifiers.")
    for record_id in record_ids:
        _positive(record_id, "record_id")
    unique_ids = list(dict.fromkeys(record_ids))
    placeholders = ", ".join("?" for _ in unique_ids)
    with _connection(path) as connection:
        rows = connection.execute(_SELECT_RECORD + f" WHERE id IN ({placeholders})", unique_ids).fetchall()
    found = {row["id"]: dict(row) for row in rows}
    missing = [identifier for identifier in unique_ids if identifier not in found]
    if missing:
        raise LookupError(f"Record identifiers not found: {missing}")
    return [found[identifier] for identifier in record_ids]


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one.")
    return float(value)


def save_predictions(path: str | Path, records: list[dict], result: dict, *,
                     model_version: str, record_ids: list[int | None] | None = None) -> list[int]:
    """Persist an entire prediction batch or roll back every row."""
    predictions = result["predictions"]
    if not records or len(records) != len(predictions):
        raise ValueError("Inputs and predictions must have the same nonzero length.")
    ids = [None] * len(records) if record_ids is None else record_ids
    if len(ids) != len(records):
        raise ValueError("record_ids must align with inputs.")
    if not isinstance(model_version, str) or len(model_version) != 64 or any(c not in "0123456789abcdef" for c in model_version):
        raise ValueError("model_version must be a SHA-256 hexadecimal digest.")
    if not isinstance(result["model"], str) or not result["model"].strip():
        raise ValueError("A model name is required.")
    threshold = _probability(result["threshold"], "threshold")
    values = []
    for record, prediction, record_id in zip(records, predictions, ids):
        if set(record) != set(FEATURE_COLUMNS):
            raise ValueError("Prediction inputs must contain exactly the predictor columns.")
        if record_id is not None:
            _positive(record_id, "record_id")
        probability = _probability(prediction["probability"], "probability")
        _binary(prediction["label"], "label")
        if prediction["label"] != int(probability >= threshold):
            raise ValueError("Prediction labels must match the saved threshold.")
        values.append((record_id, result["model"], model_version, probability,
                       prediction["label"], threshold, json.dumps(record, allow_nan=False), _now()))
    with _connection(path) as connection:
        return [connection.execute(
            "INSERT INTO predictions (record_id, model, model_version, probability, label, threshold, input_data, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", row
        ).lastrowid for row in values]


def _prediction(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["inputs"] = json.loads(item.pop("input_data"))
    return item


def list_predictions(path: str | Path, *, limit: int = 100, offset: int = 0,
                     observed: bool | None = None) -> dict:
    """Return a page of predictions, optionally filtered by outcome availability."""
    _page(limit, offset)
    if observed is not None and type(observed) is not bool:
        raise ValueError("observed must be a boolean or None.")
    where = "" if observed is None else " WHERE observed_label IS " + ("NOT NULL" if observed else "NULL")
    with _connection(path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM predictions" + where).fetchone()[0]
        rows = connection.execute(
            "SELECT * FROM predictions" + where + " ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
    return {"total": total, "items": [_prediction(row) for row in rows]}


def record_outcome(path: str | Path, prediction_id: int, observed_label: int) -> dict | None:
    """Attach a verified outcome; never infer ground truth from a prediction."""
    _positive(prediction_id, "prediction_id")
    _binary(observed_label, "observed_label")
    with _connection(path) as connection:
        connection.execute("UPDATE predictions SET observed_label = ?, observed_at = ? WHERE id = ?",
                           (observed_label, _now(), prediction_id))
        row = connection.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    return _prediction(row) if row else None


def export_feedback(path: str | Path, output_path: str | Path) -> int:
    """Export raw predictors and verified labels for a future training review."""
    destination = Path(output_path)
    if destination.resolve() == Path(path).resolve():
        raise ValueError("Export destination must differ from the database.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _connection(path) as connection, destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in connection.execute("SELECT input_data, observed_label FROM predictions WHERE observed_label IS NOT NULL ORDER BY id"):
            writer.writerow({**json.loads(row["input_data"]), TARGET: row["observed_label"]})
            count += 1
    return count
