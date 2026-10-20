# Loan classification

Python 3.10 or newer. Use the project `.venv` interpreter for the notebooks.

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

## Source files

| File | Responsibility | Main function |
| --- | --- | --- |
| `src/schema.py` | Target name and allowed predictor columns | Constants only |
| `src/audit.py` | Gearbox checks for missingness, duplicates, ranges, and counts | `run_loan_audit()` |
| `src/data.py` | Create or restore the same stratified train/test split | `get_or_create_split()` |
| `src/features.py` | Encode term/employment and calculate income ratios | `build_features()` |
| `src/preprocessing.py` | Audit, clean, median-impute, and optionally scale numeric features | `audit_preprocessing()`, `clean_numeric()`, `make_preprocessor()` |
| `src/models.py` | Select a classifier and fit its complete pipeline | `create_model()`, `train_model()` |
| `src/evaluation.py` | Calculate metrics from class-1 probabilities | `evaluate_predictions()` |
| `src/inference.py` | Load saved pipelines and predict without training | `load_predictor()` |
| `src/__init__.py` | Mark `src` as a Python package | No execution logic |

Custom abstract classes, strategy wrappers, and the factory were removed.
Scikit-learn's `Pipeline` connects ordinary functions with fitted transformers
and a classifier. Preprocessing statistics stay inside the fitted pipeline.

`ml-gearbox==0.1.1` provides auditing, EDA, and persistent splits. It has no
fitted imputer or scaler, so scikit-learn provides those operations.

## Notebooks

Run in order:

1. `01_audit.ipynb`: check the raw extract.
2. `02_eda.ipynb`: explore outer training data with Gearbox.
3. `03_feature_engineering.ipynb`: inspect numeric features.
4. `04_data_preprocessing.ipynb`: fit and save a preprocessing transformer.
5. `05_model.ipynb`: compare a prior baseline, random forest, logistic regression, and XGBoost on validation data; refit and save every pipeline.
6. `06_evaluation.ipynb`: compare all saved pipelines on the reserved test set and export results for the active model.

The outer split reserves 20% for the final test. Notebook 05 splits the other
80% again into training and validation rows, selects by validation average
precision, and refits on all outer training rows. Its candidate pipelines fit
their own preprocessing; they do not reuse notebook 04's fitted transformer.
The decision threshold stays at 0.5. Automatic model selection uses validation
average precision. The test comparison does not change that selection.

## Choose a model

Supported names: `random_forest`, `logistic_regression`, and `xgboost`.
Logistic regression automatically receives training-fitted scaling. Tree
models use unscaled features by default. Pass `scale=True` or `scale=False`
to override this, and pass classifier parameters to `train_model()` directly.

```python
from pathlib import Path
import pandas as pd

from src.data import get_or_create_split
from src.models import train_model

data = pd.read_csv("data/raw/data.csv")
split = get_or_create_split(data, Path("data/processed/split.parquet"))
pipeline = train_model(split["train"], model="logistic_regression", C=0.5)
```

In notebook 05, set `MODEL_TO_USE` to `"auto"` or a supported model name.
In notebook 06, leave `MODEL_TO_USE = None` to use the saved choice, or set
a name to inspect that model. All models remain in the comparison table.
After notebook 05 has run, switching between saved pipelines requires no retraining:

```python
import joblib

bundle = joblib.load("models/loan_models.joblib")
pipeline = bundle["pipelines"]["xgboost"]
```

## FastAPI

Run notebook 05 to create `models/loan_models.joblib`, then start the API from
the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- Interactive docs: `http://127.0.0.1:8000/docs`.
- `GET /health`: loaded model names and the default model; returns 503 if the artifact is missing or invalid.
- `POST /predict`: validate 1–1,000 raw loan records and return probabilities and labels in input order.

The API loads the artifact once at startup. Set `MODEL_PATH` to use another
trusted local artifact. Restart after replacing it. Omit `model` to use the
saved selection, or choose a model listed by `/health`.

```powershell
$body = @{
    model = "xgboost"
    records = @(@{
        loan_amnt = 10000; term = "36 months"; int_rate = 12.0
        installment = 332.14; emp_length = "5 years"; annual_inc = 60000
        dti = 18.5; delinq_2yrs = 0; fico_range_low = 700
        inq_last_6mths = 1; pub_rec = 0; revol_util = 42.0
        total_acc = 15; mort_acc = 1; pub_rec_bankruptcies = 0
    })
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -Body $body
```

Pydantic rejects missing required fields, extra fields, unsupported categories,
and nonfinite numbers. Invalid requests return 422. A valid model name absent
from the loaded artifact returns 400; unavailable inference returns 503.
The probability refers to class 1; the label uses the artifact's saved threshold.

`api/schemas.py` owns HTTP schemas, `api/main.py` owns routes and startup, and
`src/inference.py` owns loading and inference. The API contains no training or
data transformations.

## Inputs and outputs

- Input: `data/raw/data.csv`.
- Membership: `data/processed/split.parquet`, `data/processed/validation_split.parquet`.
- Fitted artifacts: `models/preparation.joblib`, `models/loan_models.joblib` (all models), `models/loan_model.joblib` (selected model).
- Comparisons: `reports/validation_metrics.csv`, `reports/test_model_comparison.csv`.
- Results: `reports/test_metrics.json`, `reports/test_predictions.csv`, `reports/test_confusion_matrix.csv`, plus metrics and predictions for all models.

The saved model includes the full pipeline, threshold, selected estimator name,
and validation results. Existing `train.csv` and `test.csv` files are preserved.

## Data rules

Employment below one year maps to 0; ten or more years maps to 10. Missing
employment has an indicator. Negative DTI and infinities become missing.
Zero income produces missing income ratios. Other large finite values remain
available for review. All-missing training features receive zero.

`loan_status` codes have no documented business meaning in this extract.
The content fingerprint `_row_id` is excluded from predictors. This random
split cannot establish performance on future loans or unseen borrowers.

## Development and Git

Read `AGENTS.md` for the notebook → source → API workflow and the practical
Factory, Strategy, and Template patterns. `COMMIT_PLAN.md` lists every file
commit and feature branch.

Branches follow `main → staging → development → feature/<purpose>`.
Commit one file at a time on each feature, merge it into `development`, then
promote `development → staging → main` with merge commits. Data and generated
artifacts stay ignored. This workflow does not push or deploy automatically.
