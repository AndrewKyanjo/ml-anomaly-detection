# Project workflows

## Implemented

| Workflow | Trigger | Result |
| --- | --- | --- |
| Test CI | Pull request or push to `development`, `staging`, or `main` | Python tests without private data or artifacts; frontend lint/build checks; Compose configuration. |
| Dashboard | Browser navigation | React pages call the API for records, predictions, history, outcomes, and model health. |
| CSV import | CLI command | Transactional SQLite import identified by file SHA-256. |
| Prediction | API request | Saved-pipeline inference with a durable input and prediction snapshot. |
| Outcome collection | API update | An independently observed outcome attached to a prediction. |
| Feedback export | CLI command | Predictors and observed labels exported for review. |
| Training and evaluation | Manual CLI or GitHub dispatch on a Windows self-hosted runner | Validation-based selection, fitted pipelines, and reserved test reports. |
| Docker Hub publishing | Manual GitHub dispatch on a Windows self-hosted runner without input fields | Local model compatibility check, image build, API health check, and publishing. |

## Feature development

1. Prototype and audit in `notebooks/`, using `ml-gearbox` for data work.
2. Extract validated functions to `src/` and import them from notebooks.
3. Check changed behavior and execute affected notebooks.
4. Save and evaluate complete fitted pipelines before changing serving.
5. Keep HTTP validation in `api/`; keep model and database operations in `src/`.
6. Keep dashboard components in `frontend/src/components`, HTTP calls in
   `frontend/src/services/api.ts`, and request/response types in `frontend/src/types`.

Use `create_model()` for construction, compatible estimators for interchangeable
strategies, and the fixed pipeline sequence for training. Add custom interfaces
only when existing functions and estimator interfaces cannot express the work.

Branch from the latest `development` into `feature/<purpose>`. Keep ordinary
commits to one file. Merge features with `--no-ff` into `development`, then merge
`development -> staging -> main` after checks pass. Publishing is separate.

## Dashboard development

Start FastAPI on `127.0.0.1:8000`, then use Node.js 24 in a second terminal:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://127.0.0.1:5173`. The development client calls the API at
`http://127.0.0.1:8000`. Run `npm run lint` and `npm run build` before integration;
CI performs both checks on a hosted runner.

The Docker build compiles the dashboard and embeds its static files in the same
Python image as the API and trained models. Open `http://127.0.0.1:8000/` after
starting Compose. Pages use `/app/records`, `/app/predict`, `/app/batch`, and
`/app/history`; these paths support direct navigation and refresh. API paths
such as `/records` continue to return JSON. Production requests use the same
origin, with no separate frontend container or Node server.

## Import, predict, and collect feedback

```powershell
docker compose up -d
docker compose run --rm api python -m src.manage import-csv /input/data.csv
```

The CSV comes from local `data/raw/`, mounted read-only at `/input`. Invalid
imports roll back. Identical file contents return the existing dataset ID;
changed contents create a new dataset. Use `dataset_id` to query that import.
The named database volume survives container replacement.

Submit raw inputs to `POST /predict` or imported IDs to `POST /predict/records`.
Retain returned prediction IDs. Each stored prediction includes the exact input,
artifact fingerprint, model name, probability, threshold, label, and timestamp.
After obtaining the actual outcome, attach it through
`PATCH /predictions/{id}/outcome` with `{"observed_label": 0}` or
`{"observed_label": 1}`. Query labeled predictions using
`GET /predictions?observed=true`.

```powershell
docker compose exec api python -m src.manage export-feedback /app/data/feedback.csv
docker compose cp api:/app/data/feedback.csv data/feedback.csv
```

Review label provenance, repeated predictions, duplicate entities, missingness,
and sampling bias before assembling a training revision. Split reviewed data
before fitting and preserve existing evaluation membership. The export is a
candidate dataset; do not blindly append it to training data. A predicted label
is never ground truth. Collection does not retrain or replace models.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe -m src.manage train data/raw/data.csv
```

Optional arguments: `--output`, `--splits-dir`, `--reports-dir`. Training selects
using validation average precision and reports test performance separately.
Review both reports before accepting a bundle for serving.

For GitHub Actions, complete the self-hosted runner setup below. Open **Actions
-> Train and evaluate models -> Run workflow** and select the branch. There are
no dispatch input fields or dataset URLs. Training reads
`ML_PROJECT_PATH/data/raw/data.csv` and saves the updated model bundle back to
`ML_PROJECT_PATH/models/loan_models.joblib`. Review the `training-metrics`
artifact before publishing. `trained-models` contains `loan_models.joblib`;
artifacts last seven days.

Training uses the local project's persisted split files to preserve evaluation
membership. Use the same versioned CSV and seed for reproducibility. Training
does not publish an image or replace a running container automatically.

## Self-hosted runner setup

1. Open repository **Settings -> Actions -> Runners -> New self-hosted runner**
   and select **Windows / x64**.
2. Follow the displayed download, configuration, and startup commands. Install
   outside the project, for example `C:/actions-runner`, so runner checkout
   cleanup cannot remove your local dataset or models.
3. Add custom label `ml-project`; the workflows select
   `[self-hosted, Windows, X64, ml-project]`.
4. Install Python 3.14. Start the runner under the Windows account that runs
   Docker Desktop and can read your project files. Keep it online and use Linux
   containers when publishing.
5. Under **Settings -> Secrets and variables -> Actions -> Variables**, create
   repository variable `ML_PROJECT_PATH` with your absolute project directory,
   for example `C:/projects/ml-project`. Keep `data/raw/data.csv` and the fitted
   `models/loan_models.joblib` inside that directory.

The runner executes code from the selected repository branch on your computer.
Allow only trusted maintainers to dispatch these workflows. They run manually;
pull-request CI remains on hosted runners. No dataset URL is required; the CSV
is read locally and is not uploaded as an artifact. Local CSV and SQLite files
stay ignored and outside the image.

## Build and publish

First create `models/loan_models.joblib`, then run:

```powershell
docker compose build
docker compose up -d
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

The image embeds the compiled React dashboard, trained pipelines, and notebook
code with outputs removed. Open `/` for the dashboard and `/docs` for the API
reference. CSV and SQLite data are mounted outside the image. Existing database
volumes survive image upgrades.

For GitHub publishing:

1. Complete the runner setup and configure repository secrets
   `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` once. The token needs permission to
   push to your Docker Hub repository.
2. Ensure `ML_PROJECT_PATH/models/loan_models.joblib` exists, Docker Desktop is
   running, and the updated workflows have been pushed.
3. Open **Actions -> Publish Docker image -> Run workflow** and select the branch.

No run ID, CSV URL, or image tag is entered at dispatch. Publishing copies the
existing local model bundle into a clean checkout of the selected commit. It
builds a Linux AMD64 image and starts the API with a temporary SQLite volume to
verify model loading with the declared dependencies. After `/health` succeeds,
it pushes `<username>/ml-anomaly-detection:sha-<commit>` and `:latest`. Publishing does not
retrain; run the separate training workflow when a new bundle is needed.
Commits and merges do not publish images automatically.

For a direct local push, train and build using local files,
run `docker login`, tag the tested image under your account, then run
`docker push`. See the commands in the [README](../README.md#docker-hub-and-automation).
For rollback, retain the registry image digest. Repeated publication at the
same code commit can change its tag when the local model bundle changes.

## Possible next workflows

| Workflow | Purpose | Required decision |
| --- | --- | --- |
| Batch inference | Predict CSV rows in bounded chunks. | Resumption and duplicate-request handling. |
| Input drift report | Compare serving inputs with a versioned training reference. | Features, windows, and review thresholds. |
| Outcome monitoring | Report labeled performance by artifact version. | Minimum observed outcomes and reporting window. |
| Scheduled retraining | Train a candidate from reviewed, versioned feedback. | Label review, leakage checks, retained evaluation membership, and promotion criteria. |
| Database backup | Keep consistent SQLite backups outside the volume. | Retention and a tested restoration procedure. |

These extensions are not enabled. Current operations keep collection, training,
evaluation, and publication separate and explicit.
