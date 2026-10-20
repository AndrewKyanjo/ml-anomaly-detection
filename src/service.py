"""Combine saved inference with durable prediction records."""

from pathlib import Path

from .inference import LoanPredictor
from .schema import FEATURE_COLUMNS
from .storage import get_records, save_predictions


def predict_and_store(
    predictor: LoanPredictor,
    database_path: str | Path,
    records: list[dict],
    model: str | None = None,
    record_ids: list[int] | None = None,
) -> dict:
    """Return predictions only after the complete batch has been recorded."""
    result = predictor.predict(records, model=model)
    identifiers = save_predictions(
        database_path, records, result,
        model_version=predictor.model_version, record_ids=record_ids,
    )
    return {
        **result,
        "model_version": predictor.model_version,
        "predictions": [
            {**prediction, "id": identifier}
            for prediction, identifier in zip(result["predictions"], identifiers)
        ],
    }


def predict_saved_records(
    predictor: LoanPredictor,
    database_path: str | Path,
    record_ids: list[int],
    model: str | None = None,
) -> dict:
    """Predict stored rows without including their labels or database metadata."""
    stored = get_records(database_path, record_ids)
    inputs = [{column: record[column] for column in FEATURE_COLUMNS} for record in stored]
    return predict_and_store(predictor, database_path, inputs, model, record_ids)
