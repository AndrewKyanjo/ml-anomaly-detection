# Project workflows

## Implemented

| Workflow | Trigger | Result |
| --- | --- | --- |
| Test CI | Pull request or push to `development`, `staging`, or `main` | Tests without private data or existing artifacts; checks Compose configuration. |
| CSV import | CLI command | Transactional SQLite import identified by file SHA-256. |
| Prediction | API request | Saved-pipeline inference with a durable input and prediction snapshot. |
| Outcome collection | API update | An independently observed outcome attached to a prediction. |
| Feedback export | CLI command | Predictors and observed labels exported for review. |
| Training and evaluation | Manual CLI or GitHub dispatch | Validation-based selection, fitted pipelines, and reserved test reports. |
| Docker Hub publishing | Manual GitHub dispatch | An image containing trained pipelines, checked before publishing. |

## Feature development

1. Prototype and audit in `notebooks/`, using `ml-gearbox` for data work.
2. Extract validated functions to `src/` and import them from notebooks.
3. Check changed behavior and execute affected notebooks.
4. Save and evaluate complete fitted pipelines before changing serving.
5. Keep HTTP validation in `api/`; keep model and database operations in `src/`.

Use `create_model()` for construction, compatible estimators for interchangeable
strategies, and the fixed pipeline sequence for training. Add custom interfaces
only when existing functions and estimator interfaces cannot express the work.

Branch from the latest `development` into `feature/<purpose>`. Keep ordinary
commits to one file. Merge features with `--no-ff` into `development`, then merge
`development -> staging -> main` after checks pass. Publishing is separate.

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

For GitHub Actions:

1. Push the source and workflow revision.
2. Open **Actions -> Train and evaluate models -> Run workflow**.
3. Select the branch and provide `csv_url`, an HTTPS URL accessible to the runner.
   Dispatch inputs appear in workflow history; choose the data-access mechanism
   accordingly.
4. Review the `training-metrics` artifact and retain the successful run ID.
   `trained-models` contains `loan_models.joblib`. Artifacts last seven days.

Local CLI runs reuse saved split files. A fresh GitHub runner creates its splits
from the supplied CSV because local ignored splits are not uploaded. Reproduce
with the same versioned CSV and seed. Promotion remains explicit.

## Build and publish

First create `models/loan_models.joblib`, then run:

```powershell
docker compose build
docker compose up -d
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

The image embeds trained pipelines and notebook code with outputs removed. CSV
and SQLite data are mounted outside the image. Existing database volumes survive
image upgrades.

For GitHub publishing:

1. Set repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` with write
   access to the intended Docker Hub repository.
2. Complete and review training for the exact commit to publish.
3. Open **Actions -> Publish Docker image -> Run workflow** on that commit.
4. Supply `training_run_id` and a release tag such as `0.1.0`.

Publishing verifies the selected training run succeeded in this repository at
the same commit. It downloads the artifact, builds a Linux AMD64 image, and
starts the API with a SQLite volume. After `/health` succeeds, it pushes
`<username>/ml-anomaly-detection:<tag>`. Commits and merges never publish images.

For a local push, run `docker login`, tag the tested image under your account,
then run `docker push`. Retain release tags so a previous tested image can be
selected during rollback.

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
