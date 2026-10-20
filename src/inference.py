"""Load saved pipelines once and predict from raw loan records."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .schema import FEATURE_COLUMNS


@dataclass(frozen=True)
class LoanPredictor:
    """Keep fitted pipelines and their saved decision threshold in memory."""

    pipelines: dict[str, Pipeline]
    default_model: str
    threshold: float
    model_version: str = ""

    def predict(self, records: list[dict], model: str | None = None) -> dict:
        """Return class-1 probabilities and labels without fitting any step."""
        name = self.default_model if model is None else model
        if name not in self.pipelines:
            raise ValueError(f"Model {name!r} is unavailable; choose from {list(self.pipelines)}.")
        if not records:
            raise ValueError("Provide at least one loan record.")
        probabilities = np.asarray(
            self.pipelines[name].predict_proba(pd.DataFrame.from_records(records)),
            dtype=float,
        )
        if probabilities.shape != (len(records), 2):
            raise ValueError("The pipeline must return two probabilities per record.")
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)
        ).any():
            raise ValueError("The pipeline returned invalid probabilities.")
        return {
            "model": name,
            "threshold": self.threshold,
            "predictions": [
                {"probability": float(p), "label": int(p >= self.threshold)}
                for p in probabilities[:, 1]
            ],
        }


def load_predictor(path: str | Path) -> LoanPredictor:
    """Load a trusted local notebook-05 artifact and validate its fitted state."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model artifact not found: {source}")
    try:
        artifact_bytes = source.read_bytes()
        bundle = joblib.load(BytesIO(artifact_bytes))
    except Exception as exc:
        raise ValueError("Could not load the saved model artifact.") from exc
    if not isinstance(bundle, dict):
        raise ValueError("The model artifact must be a dictionary.")
    pipelines = bundle.get("pipelines")
    if not isinstance(pipelines, dict) or not pipelines:
        raise ValueError("The model artifact must contain fitted pipelines.")
    default_model = bundle.get("selected_model")
    if not isinstance(default_model, str) or default_model not in pipelines:
        raise ValueError("The selected model is not in the saved pipelines.")
    if bundle.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("The artifact predictor columns do not match the source schema.")
    threshold = bundle.get("threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The saved threshold must be between zero and one.")
    for name, pipeline in pipelines.items():
        if not isinstance(name, str) or not isinstance(pipeline, Pipeline):
            raise ValueError("Every saved model must be a named sklearn Pipeline.")
        try:
            check_is_fitted(pipeline)
            if not callable(getattr(pipeline, "predict_proba", None)):
                raise ValueError("The pipeline does not support probability prediction.")
            if not np.array_equal(pipeline.classes_, [0, 1]):
                raise ValueError("The pipeline must predict classes zero and one.")
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Saved pipeline {name!r} must be a fitted binary classifier.") from exc
    version = hashlib.sha256(artifact_bytes).hexdigest()
    return LoanPredictor(pipelines, default_model, float(threshold), version)
