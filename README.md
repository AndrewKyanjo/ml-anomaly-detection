# Loan classification

Compare random forest, logistic regression, and XGBoost pipelines, serve them
with FastAPI and a React dashboard, and store source records and prediction
outcomes in SQLite.

## Setup

Use Python 3.14 to match the container and saved artifact dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements.lock.txt -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Place the CSV at `data/raw/data.csv`. It must contain the 15 predictors in
`src/schema.py` and binary target `loan_status`. Data, databases, models, reports,
and environment files are ignored by Git.

## Notebooks and source

Validate each step in a notebook, extract it into a focused `src/` function,
then import that function from the notebook. The API delegates to source code.

| Notebook | Responsibility |
| --- | --- |
| `01_audit.ipynb` | Audit the raw extract with `ml-gearbox`. |
| `02_eda.ipynb` | Profile and explore outer training rows with `ml-gearbox`. |
| `03_feature_engineering.ipynb` | Validate derived signals. |
| `04_data_preprocessing.ipynb` | Check cleaning and training-fitted preprocessing. |
| `05_model.ipynb` | Compare candidates on validation data and save fitted pipelines. |
| `06_evaluation.ipynb` | Report reserved test performance for saved models. |

| Source file | Responsibility |
| --- | --- |
| `schema.py` | Define target and allowed predictors. |
| `audit.py` | Check missingness, duplicates, ranges, and counts with Gearbox. |
| `data.py` | Create or restore stratified split membership. |
| `features.py` | Encode categories and calculate income ratios. |
| `preprocessing.py` | Audit, clean, impute, and optionally scale features. |
| `models.py` | Construct a classifier and fit its complete pipeline. |
| `evaluation.py` | Calculate metrics from probabilities. |
| `inference.py` | Load trusted fitted artifacts and predict. |
| `storage.py` | Import CSV records; persist predictions and observed outcomes. |
| `service.py` | Coordinate inference and prediction storage. |
| `manage.py` | Provide database, feedback export, and training commands. |

`create_model()` supplies the Factory. The common estimator interface supplies
interchangeable Strategies. `train_model()` fixes the Template sequence:
features, preprocessing, classification. Fitted state stays in the pipeline.

`ml-gearbox==0.1.1` provides auditing, profiling, and persisted splits. It has no
fitted imputer or scaler; scikit-learn provides those steps after Gearbox audits.

## Training and model selection

Run notebooks in order, or train from the command line:

```powershell
.\.venv\Scripts\python.exe -m src.manage train data/raw/data.csv
```

Training preserves saved split membership, selects by validation average
precision, refits on outer training rows, and reports reserved test performance.
The test comparison never changes the selection. The threshold is 0.5.

```python
from src.models import train_model

pipeline = train_model(training_frame, model="logistic_regression", C=0.5)
```

Supported names: `random_forest`, `logistic_regression`, `xgboost`. Logistic
regression receives training-fitted scaling automatically. In notebook 05,
`MODEL_TO_USE = "auto"` selects the validation winner; a model name overrides it.
In notebook 06, `None` uses the saved choice. All candidates are saved in
`models/loan_models.joblib`; requests can switch between them without retraining.

## Docker

Create `models/loan_models.joblib` before building. The image includes the
compiled React dashboard, trained pipelines, source, and notebooks with outputs
removed. A Node 24 build stage compiles the frontend; the final Python image
serves the dashboard and API on one port. CSV and SQLite files stay outside the
image. Execute notebooks in the local development environment.

```powershell
docker compose build
docker compose up -d
docker compose run --rm api python -m src.manage import-csv /input/data.csv
```

Compose mounts `./data/raw` read-only at `/input` and the named `app-data` volume
at `/app/data`. SQLite uses `/app/data/app.sqlite3`. Identical CSV contents are
imported once, identified by SHA-256. The API runs as a non-root user with one
worker. Open `http://127.0.0.1:8000/` for the dashboard or
`http://127.0.0.1:8000/docs` for the API reference.

`docker compose down` preserves the database volume. Inspect startup with
`docker compose logs api`. The health check requires both models and storage.

## Local API

```powershell
.\.venv\Scripts\python.exe -m src.manage import-csv data/raw/data.csv
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The local database defaults to `data/app.sqlite3`; override it with
`DATABASE_PATH`. `MODEL_PATH` selects another trusted local bundle. Restart after
replacing it. Startup creates database tables; import and training are explicit.

## React dashboard

Use Node.js 24. Start the API with the command above, then run these commands
in another terminal:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://127.0.0.1:5173`. Development requests target
`http://127.0.0.1:8000`; production requests use the dashboard's origin. Override
the API address with `VITE_API_BASE_URL` when building for another deployment.
FastAPI permits the local Vite origins on port 5173. `CORS_ORIGINS` replaces that
allowlist with comma-separated origins when needed.

The dashboard provides loaded-model health, paginated records and record details,
a 15-field prediction form, batch predictions from stored IDs, and prediction
history with independently observed outcomes. Prediction probabilities refer to
class 1; the dataset does not establish a business meaning for either class.

| Browser route | Page |
| --- | --- |
| `/` | Dashboard and model health. |
| `/app/records` | Records explorer and details. |
| `/app/predict` | Raw prediction form. |
| `/app/batch` | Stored-record batch prediction. |
| `/app/history` | Prediction history and outcome feedback. |

Browser routes support direct navigation and refresh. API routes such as
`GET /records` retain their JSON responses. UI components live in
`frontend/src/components`, HTTP calls in `frontend/src/services/api.ts`, and
contracts in `frontend/src/types`.

To check or build the dashboard locally:

```powershell
cd frontend
npm run lint
npm run build
```

After building, restart FastAPI and open `http://127.0.0.1:8000/`. It serves
`frontend/dist` by default; `FRONTEND_PATH` selects another compiled directory.
Without a local frontend build, `/` redirects to `/docs`.

## API endpoints

| Method and path | Behavior |
| --- | --- |
| `GET /health` | Report loaded models and the default. |
| `GET /records` | Query with `limit`, `offset`, optional `loan_status` and `dataset_id`. |
| `GET /records/{id}` | Retrieve one imported record. |
| `POST /predict` | Predict 1 to 1,000 raw records and store the batch. |
| `POST /predict/records` | Predict imported records using `record_ids`. |
| `GET /predictions` | Query history; `observed=true` selects predictions with outcomes. |
| `PATCH /predictions/{id}/outcome` | Attach an independently observed binary label. |

Omit `model` to use the saved default. Prediction responses contain `model`,
`model_version`, `threshold`, and ordered `predictions`. Each prediction has
`id`, `probability`, and `label`. The version is the artifact's SHA-256 digest.
Probability refers to class 1; the saved threshold determines the binary label.

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/records?limit=10&offset=0&dataset_id=1'
$body = @{ record_ids = @(1, 2); model = "xgboost" } | ConvertTo-Json
$result = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict/records -ContentType application/json -Body $body
$result
```

For a new record, submit raw predictors:

```powershell
$body = @{
    model = "logistic_regression"
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

Invalid inputs return 422; missing records return 404. A supported model absent
from the artifact returns 400; unavailable models or storage return 503.
Predictions succeed only after the entire batch is stored.

## Outcomes and feedback

SQLite stores imported data in `records`, and prediction input snapshots, model
identity, probabilities, labels, thresholds, and timestamps in `predictions`.
Observed outcomes remain empty until explicitly submitted.

After independently obtaining the outcome for a prediction ID:

```powershell
$body = @{ observed_label = 1 } | ConvertTo-Json
Invoke-RestMethod -Method Patch -Uri http://127.0.0.1:8000/predictions/1/outcome -ContentType application/json -Body $body
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predictions?observed=true&limit=100'

docker compose exec api python -m src.manage export-feedback /app/data/feedback.csv
docker compose cp api:/app/data/feedback.csv data/feedback.csv
```

Local export:

```powershell
.\.venv\Scripts\python.exe -m src.manage --database data/app.sqlite3 export-feedback data/feedback.csv
```

Exports contain predictors and observed labels only. Review and deduplicate
feedback, audit the proposed training revision, and preserve evaluation
membership before retraining. Predicted labels are never ground truth. Feedback
does not retrain or replace the running model automatically.

## Docker Hub and automation

The default image is `local/ml-anomaly-detection:latest`. After testing, publish
under your own Docker Hub account with these manual commands:

```powershell
docker login
docker tag local/ml-anomaly-detection:latest your-dockerhub-username/ml-anomaly-detection:0.1.0
docker push your-dockerhub-username/ml-anomaly-detection:0.1.0
```

GitHub Actions runs Python tests and frontend lint/build checks on hosted runners.
Training and publishing use a
Windows self-hosted runner on the computer containing your local CSV and models.
Set it up once:

1. Open **Settings -> Actions -> Runners -> New self-hosted runner**, select
   **Windows / x64**, and follow GitHub's install and registration commands.
2. Install it outside this project, for example `C:/actions-runner`, and add the
   custom label `ml-project` during registration.
3. Install Python 3.14. Run the runner under the same Windows account as Docker
   Desktop, keep it online, and use Linux container mode.
4. Under **Settings -> Secrets and variables -> Actions -> Variables**, set
   `ML_PROJECT_PATH` to the absolute project directory, such as
   `C:/projects/ml-project`. This directory must contain
   `data/raw/data.csv` for training and `models/loan_models.joblib` for publishing.

Under the same settings page, add these repository secrets:

| Secret | Value |
| --- | --- |
| `DOCKERHUB_USERNAME` | Your Docker Hub username. |
| `DOCKERHUB_TOKEN` | A Docker Hub token with permission to push images. |

Push the updated workflows, then open **Actions -> Publish Docker image -> Run
workflow** and select the branch. No dispatch fields are required. Publishing
copies the existing local fitted model bundle into a clean checkout, builds the
image including the dashboard, verifies model loading and API health, and pushes
`<username>/ml-anomaly-detection:sha-<commit>` and `:latest`.

To retrain, run **Train and evaluate models**. It reads the local CSV and writes
the updated fitted bundle back to `ML_PROJECT_PATH`; review the
`training-metrics` artifact before publishing. This workflow also requires no
dispatch fields. CSV and SQLite files remain ignored and outside the image.

Neither workflow needs a dataset URL. These manual workflows execute trusted
repository code on your computer; restrict dispatch access to trusted
maintainers. Pull-request tests continue to use hosted runners.

See [workflows](docs/WORKFLOWS.md) for the operational sequence and future options.

## Outputs and development

- Splits: `data/processed/split.parquet` and `validation_split.parquet`.
- Fitted models: `models/loan_models.joblib`; notebooks also save preparation and selected-model artifacts.
- Reports: `reports/validation_metrics.csv`, `reports/test_model_comparison.csv`, and notebook evaluation exports.
- Local database: `data/app.sqlite3`; container database: the `app-data` volume.

The target and `_row_id` are excluded from predictors. Class codes 0 and 1 have
no documented business meaning in this extract. Fit preprocessing on training
rows only. Random splits do not establish future or unseen-borrower performance.

Create features from `development`, commit one file at a time, merge into
`development`, then promote through `staging` to `main` with merge commits after
checks pass. Local agent guidance and commit plans are ignored and untracked;
they are not required for repository setup.
