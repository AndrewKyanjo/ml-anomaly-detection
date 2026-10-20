"""Run database operations and reproducible training from the command line."""

import argparse
import json
import os
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from .data import get_or_create_split
from .evaluation import evaluate_predictions
from .models import train_model
from .schema import FEATURE_COLUMNS, TARGET
from .storage import export_feedback, import_csv, initialize_database

MODEL_NAMES = ("random_forest", "logistic_regression", "xgboost")


def train_models(
    csv_path: str | Path,
    output: str | Path = "models/loan_models.joblib",
    splits_dir: str | Path = "data/processed",
    reports_dir: str | Path = "reports",
) -> dict:
    """Select on validation, refit all candidates, and report reserved test scores."""
    frame = pd.read_csv(csv_path)
    if set(frame.columns) != {*FEATURE_COLUMNS, TARGET}:
        raise ValueError("Training CSV must contain exactly the predictors and target.")
    splits_path = Path(splits_dir)
    outer = get_or_create_split(frame, splits_path / "split.parquet")
    development, test = outer["train"], outer["test"]
    inner = get_or_create_split(development, splits_path / "validation_split.parquet")
    training, validation = inner["train"], inner["test"]
    candidates, validation_metrics = {}, []
    for name in MODEL_NAMES:
        pipeline = train_model(training, model=name)
        probabilities = pipeline.predict_proba(validation.drop(columns=TARGET))[:, 1]
        validation_metrics.append({
            "model": name, **evaluate_predictions(validation[TARGET], probabilities),
        })
        candidates[name] = pipeline
    selected = max(validation_metrics, key=lambda row: row["average_precision"])["model"]
    pipelines = {
        name: clone(pipeline).fit(development.drop(columns=TARGET), development[TARGET])
        for name, pipeline in candidates.items()
    }
    test_metrics = [
        {"model": name, **evaluate_predictions(
            test[TARGET], pipeline.predict_proba(test.drop(columns=TARGET))[:, 1],
        )}
        for name, pipeline in pipelines.items()
    ]
    bundle = {
        "pipelines": pipelines,
        "selected_model": selected,
        "threshold": 0.5,
        "feature_columns": list(FEATURE_COLUMNS),
        "validation_metrics": validation_metrics,
        "selection_metric": "average_precision",
    }
    destination, reports = Path(output), Path(reports_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    pd.DataFrame(validation_metrics).to_csv(reports / "validation_metrics.csv", index=False)
    pd.DataFrame(test_metrics).to_csv(reports / "test_model_comparison.csv", index=False)
    return {
        "selected_model": selected, "artifact": str(destination),
        "training_rows": len(training), "validation_rows": len(validation),
        "refit_rows": len(development), "test_rows": len(test), "reports": str(reports),
    }


def main(argv: list[str] | None = None) -> int:
    """Parse commands and print one JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.getenv("DATABASE_PATH", "data/app.sqlite3"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create database tables without replacing records.")
    importer = commands.add_parser("import-csv", help="Import a CSV without duplicate imports.")
    importer.add_argument("csv_path", type=Path)
    exporter = commands.add_parser("export-feedback", help="Export verified outcomes for review.")
    exporter.add_argument("output", type=Path)
    trainer = commands.add_parser("train", help="Compare and save all supported classifiers.")
    trainer.add_argument("csv_path", type=Path)
    trainer.add_argument("--output", type=Path, default=Path("models/loan_models.joblib"))
    trainer.add_argument("--splits-dir", type=Path, default=Path("data/processed"))
    trainer.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = train_models(args.csv_path, args.output, args.splits_dir, args.reports_dir)
        elif args.command == "init-db":
            initialize_database(args.database)
            result = {"database": args.database, "initialized": True}
        elif args.command == "import-csv":
            result = import_csv(args.database, args.csv_path)
        else:
            if not Path(args.database).is_file():
                raise FileNotFoundError(f"Database not found: {args.database}")
            result = {"rows": export_feedback(args.database, args.output), "output": str(args.output)}
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
