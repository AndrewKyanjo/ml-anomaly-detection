"""Exercise local data commands and the saved training artifact contract."""

import json
import sqlite3

import pandas as pd
import pytest

from src import manage
from src.inference import load_predictor
from src.models import train_model
from src.schema import FEATURE_COLUMNS


def test_database_commands_are_idempotent_and_export_only_verified_outcomes(
    tmp_path, loan_frame, monkeypatch, capsys,
):
    database = tmp_path / "app.sqlite3"
    source, feedback = tmp_path / "source.csv", tmp_path / "feedback.csv"
    loan_frame.to_csv(source, index=False)
    monkeypatch.setenv("DATABASE_PATH", str(database))
    assert manage.main(["init-db"]) == 0
    assert json.loads(capsys.readouterr().out)["initialized"]
    assert manage.main(["import-csv", str(source)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["rows"] == len(loan_frame)
    assert not first["already_imported"]
    assert manage.main(["import-csv", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["already_imported"]
    assert manage.main(["export-feedback", str(feedback)]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 0
    assert pd.read_csv(feedback).empty
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == len(loan_frame)


def test_missing_csv_returns_an_error_without_importing(tmp_path, capsys):
    database = tmp_path / "app.sqlite3"
    with pytest.raises(SystemExit) as error:
        manage.main(["--database", str(database), "import-csv", str(tmp_path / "missing.csv")])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not database.exists()


def test_training_command_saves_switchable_models_and_split_reports(
    tmp_path, loan_frame, monkeypatch, capsys,
):
    csv_path, artifact = tmp_path / "source.csv", tmp_path / "models.joblib"
    splits, reports = tmp_path / "splits", tmp_path / "reports"
    loan_frame.to_csv(csv_path, index=False)
    fitted_names = []

    def fit_small(frame, *, model):
        fitted_names.append(model)
        parameters = {"max_iter": 100} if model == "logistic_regression" else {
            "n_estimators": 3, "max_depth": 2, "n_jobs": 1,
        }
        return train_model(frame, model=model, **parameters)

    monkeypatch.setattr(manage, "train_model", fit_small)
    assert manage.main([
        "train", str(csv_path), "--output", str(artifact),
        "--splits-dir", str(splits), "--reports-dir", str(reports),
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    predictor = load_predictor(artifact)
    assert fitted_names == list(manage.MODEL_NAMES)
    assert set(predictor.pipelines) == set(manage.MODEL_NAMES)
    assert result["training_rows"] + result["validation_rows"] == result["refit_rows"]
    assert result["refit_rows"] + result["test_rows"] == len(loan_frame)
    validation = pd.read_csv(reports / "validation_metrics.csv")
    assert predictor.default_model == validation.loc[validation.average_precision.idxmax(), "model"]
    assert set(pd.read_csv(reports / "test_model_comparison.csv").model) == set(manage.MODEL_NAMES)
    records = loan_frame.loc[:, FEATURE_COLUMNS].head(2).to_dict(orient="records")
    for name in predictor.pipelines:
        assert len(predictor.predict(records, name)["predictions"]) == 2
    assert (splits / "split.parquet").is_file()
    assert (splits / "validation_split.parquet").is_file()
