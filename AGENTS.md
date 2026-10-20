# Project guidance

Build a small, reproducible loan classification project. Keep code simple enough
to follow from the notebooks through training to prediction. Explicit user
instructions take precedence over this guidance.

## Development order

1. Explore and validate a step in `notebooks/`.
2. Extract each validated step into a focused function in `src/`.
3. Change the notebook to import that function; remove duplicate implementations.
4. Check the source code with small, meaningful tests.
5. Fit, compare, and save complete pipelines before building or changing `api/`.
6. Keep the API limited to HTTP input validation and calls into `src/`.

Use the notebooks in this order:

| Notebook | Responsibility |
| --- | --- |
| `01_audit.ipynb` | Validate the raw extract with `ml-gearbox`. |
| `02_eda.ipynb` | Profile and explore training data with `ml-gearbox`. |
| `03_feature_engineering.ipynb` | Validate signals implemented in `src/features.py`. |
| `04_data_preprocessing.ipynb` | Check cleaning and train-fitted preprocessing. |
| `05_model.ipynb` | Compare candidates on validation data; save fitted pipelines. |
| `06_evaluation.ipynb` | Report reserved test performance for saved models. |

## Directory boundaries

- `notebooks/`: experiments, plots, comparisons, and calls to reusable source code.
- `src/`: audit, splits, features, preprocessing, training, evaluation, artifact
  loading, and prediction. No HTTP framework imports.
- `api/`: FastAPI routes, Pydantic request/response schemas, application startup,
  and HTTP error handling. No training, DataFrame construction, feature
  calculations, imputation, or model-selection calculations.
- `tests/`: small independent fixtures and behavior checks. Tests must not need
  the private dataset or existing local model artifacts.

## Patterns without unnecessary classes

Prioritize Factory, Strategy, and Template patterns using the existing functions
and scikit-learn interfaces:

- **Factory:** `src.models.create_model(name, **parameters)` constructs a
  classifier. Model creation belongs here, separate from fitting and HTTP code.
  Source inference code loads and selects already fitted pipelines for serving.
- **Strategy:** a classifier or preprocessing component is interchangeable
  through the same estimator interface. `train_model()` accepts a supported
  model name or a compatible estimator. Add models without copying the workflow.
- **Template:** `train_model()` fixes the order in a scikit-learn `Pipeline`:
  feature engineering, preprocessing, then modeling. Run
  `evaluate_predictions()` on its probabilities afterward. Keep this lifecycle
  in source code; notebooks supply data, choices, and visualizations.

Prefer a direct function to a custom wrapper class. Keep fitted state inside
scikit-learn transformers and pipelines. If several custom implementations
actually require a shared stateful interface, define the small interface with
Python's `abc` module. Do not add inheritance or a separate framework to rename
the existing estimator interfaces.

## Data and model rules

- Use `ml-gearbox` first for EDA, profiling, audits, and preprocessing diagnostics
  in both notebooks and source code. Use its persisted split support.
- The pinned `ml-gearbox==0.1.1` has no fitted imputer or scaler. Keep Gearbox
  audits and use scikit-learn for training-fitted imputation and scaling. Check
  actual package capabilities before changing this arrangement.
- Preserve saved train/validation/test membership. Reject mismatched data rather
  than silently recreating an existing split.
- Fit imputers, scalers, feature-selection rules, and classifiers on training
  rows only. Use validation results for model and threshold decisions. Use the
  reserved test set for final reporting; its comparison table must not select
  the production model.
- Exclude `loan_status` and `_row_id` from predictors. Reuse the allowed columns
  in `src/schema.py`. Do not infer undocumented business meanings for classes
  `0` and `1`.
- Persist the complete fitted pipeline and its decision threshold. Predict from
  raw input records using that pipeline, so serving follows training exactly.
- Keep random seeds explicit. Preserve row order and avoid modifying caller
  DataFrames.

## Serving rules

Use FastAPI and Pydantic. Validate raw fields, finite numeric values, supported
model names, batch limits, and forbidden extra fields at the HTTP boundary.
The target is never a prediction input.

Load trusted local artifacts through `src/` at startup. Let source inference
functions construct DataFrames, select saved models, and produce probabilities
and predictions. API routes only pass validated values and return responses.
Do not retrain at startup or expose training endpoints. Do not accept artifact
paths or serialized models from HTTP clients. Report missing artifacts clearly;
prediction requires running the model notebook first.

## Code and validation

Use small functions, clear names, type hints, and short docstrings. Keep imports
explicit and avoid extra configuration layers. Explain behavior through code;
keep project instructions and setup commands in Markdown.

Test changed behavior: leakage prevention, persisted splits, model switching,
artifact loading, request validation, and prediction agreement with the saved
pipeline. Use small fitted fixtures rather than retraining the full dataset.
Do not add tests that merely repeat the implementation.

Run the relevant tests for each feature and the full suite before integration:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run affected notebooks when changing their executable workflow. Do not claim
notebooks or checks passed unless they were run successfully.

## Git workflow

Use this local branch hierarchy:

```text
main
  staging
    development
      feature/<purpose>
```

Create `staging` from `main`, then `development` from `staging`. Create every
feature branch from the latest `development`, including documentation and
configuration work. Keep each ordinary commit limited to one file. Stage exact
paths with `git add -- <path>`; avoid `git add .`.

After relevant checks, merge each feature into `development` using `--no-ff`.
After the full suite passes, merge `development` into `staging`, then `staging`
into `main`, also using `--no-ff`. Preserve these merge commits and the individual
file commits; do not squash them. Merge commits integrate branch history and
are exempt from the one-file rule for ordinary implementation commits.

For a repository with no commits, establish `main` with a single `.gitignore`
commit before creating the branch hierarchy. Put all other project files on
their appropriate feature branches. See `COMMIT_PLAN.md` for this project's
initial file-by-file plan.

Never force-add ignored `data/`, `models/`, `reports/`, `.venv/`, caches, or
generated package metadata. Do not discard existing work. Push, deployment,
history rewriting, and deleting branches are separate actions that require
explicit user instructions.
