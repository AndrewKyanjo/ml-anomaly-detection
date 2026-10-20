# Local commit and integration plan

This project starts in a repository with no commits. Establish `main` with one
`.gitignore` commit, then use this hierarchy for all remaining files:

```text
main -> staging -> development -> feature/<purpose>
```

Every ordinary commit below contains exactly one file. Each feature starts from
the latest `development` and returns through a merge commit. Merge commits
preserve branch history and are exempt from the one-file implementation rule.
All operations are local; this plan includes no push or deployment.

## Establish the branch hierarchy

On the unborn `main` branch:

```powershell
git add -- .gitignore
git commit -m "chore: ignore local data and generated artifacts"
git switch -c staging main
git switch -c development staging
```

Keep raw data, fitted models, generated reports, environments, and caches
ignored. Do not force-add them.

## Feature commits

Apply the groups in order. Create each branch from `development`, commit its
files individually, check its behavior, then merge it back before starting the
next group.

### `feature/project-setup`

| File | Commit subject |
| --- | --- |
| `pyproject.toml` | `build: declare ML API and development dependencies` |

Check: install the editable project with development dependencies in `.venv`.

### `feature/data-audit`

| File | Commit subject |
| --- | --- |
| `src/__init__.py` | `chore: initialize the ML source package` |
| `src/schema.py` | `feat: define loan predictors and target` |
| `src/audit.py` | `feat: audit loan records with ml-gearbox` |
| `src/data.py` | `feat: persist reproducible loan data splits` |
| `notebooks/01_audit.ipynb` | `notebook: audit the loan dataset` |
| `notebooks/02_eda.ipynb` | `notebook: explore training data with ml-gearbox` |
| `notebooks/lending_club_columns.txt` | `docs: record source dataset columns` |
| `tests/conftest.py` | `test: provide independent loan fixtures` |
| `tests/test_audit_and_split.py` | `test: verify auditing and split persistence` |

Check: `tests/test_audit_and_split.py`.

### `feature/data-preparation`

| File | Commit subject |
| --- | --- |
| `src/features.py` | `feat: build numeric loan features` |
| `src/preprocessing.py` | `feat: audit and fit loan preprocessing` |
| `notebooks/03_feature_engineering.ipynb` | `notebook: validate loan feature engineering` |
| `notebooks/04_data_preprocessing.ipynb` | `notebook: validate training-only preprocessing` |
| `tests/test_preparation.py` | `test: verify feature rules and prevent preprocessing leakage` |

Check: `tests/test_preparation.py` and the audit/split tests.

### `feature/model-comparison`

| File | Commit subject |
| --- | --- |
| `src/models.py` | `feat: create and train selectable loan classifiers` |
| `src/evaluation.py` | `feat: evaluate binary classification probabilities` |
| `notebooks/05_model.ipynb` | `notebook: compare and save fitted loan classifiers` |
| `notebooks/06_evaluation.ipynb` | `notebook: evaluate saved models on reserved test data` |
| `tests/test_pipeline.py` | `test: verify complete training and evaluation pipelines` |
| `tests/test_models.py` | `test: verify classifier selection and model switching` |

Check: `tests/test_pipeline.py`, `tests/test_models.py`, and earlier source tests.

### `feature/prediction-api`

| File | Commit subject |
| --- | --- |
| `src/inference.py` | `feat: load saved pipelines and predict raw loan records` |
| `api/__init__.py` | `chore: initialize the HTTP serving package` |
| `api/schemas.py` | `feat: validate loan prediction requests and responses` |
| `api/main.py` | `feat: serve loan predictions with FastAPI` |
| `tests/test_inference.py` | `test: verify saved model inference and selection` |
| `tests/test_api.py` | `test: verify prediction routes and request validation` |

Check: inference/API tests and the full suite. Check startup and prediction
against the saved local artifact. Keep training and transformations in `src/`.

### `feature/project-guidance`

| File | Commit subject |
| --- | --- |
| `AGENTS.md` | `docs: guide simple notebook source and API development` |
| `COMMIT_PLAN.md` | `docs: define file-by-file commits and branch integration` |
| `README.md` | `docs: explain model selection and FastAPI usage` |

Check: documented commands, file references, and the full test suite.

## Commit and merge commands

Repeat this sequence for each feature group, substituting its branch, file,
and subject. Repeat only the `add` and `commit` pair for the remaining files in
that group:

```powershell
git switch development
git switch -c feature/project-setup development
git add -- pyproject.toml
git commit -m "build: declare ML API and development dependencies"
# Run the group's relevant checks before merging.
git switch development
git merge --no-ff feature/project-setup -m "merge: integrate project setup"
```

Before each commit, inspect `git diff --cached --name-only` and verify it lists
one path. Other untracked project files can remain in the working tree until
their feature is committed. Do not stage them together. If validation requires
a fix, keep that fix on the owning feature in another single-file commit.

## Promote the integrated result

After all feature groups are merged into `development`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git status --short
git switch staging
git merge --no-ff development -m "merge: promote development to staging"
git switch main
git merge --no-ff staging -m "merge: release staging to main"
git log --graph --oneline --all
```

Verify that `main`, `staging`, and `development` point to the same project tree,
that the working tree is clean, and that every non-merge commit changes exactly
one file. Retain the feature branches and merge history for review.
