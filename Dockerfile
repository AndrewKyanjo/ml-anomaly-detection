# syntax=docker/dockerfile:1
FROM python:3.14-slim AS notebook-source

WORKDIR /source
COPY notebooks ./notebooks
RUN python - <<'PY'
import json
from pathlib import Path

for path in Path("notebooks").rglob("*.ipynb"):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook.get("metadata", {}).pop("widgets", None)
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
PY

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/loan_models.joblib \
    DATABASE_PATH=/app/data/app.sqlite3

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/models \
    && chown app:app /app/data

COPY requirements.lock.txt ./
RUN python -m pip install -r requirements.lock.txt

COPY pyproject.toml ./
COPY src ./src
COPY api ./api
RUN python -m pip install --no-deps .

COPY --from=notebook-source /source/notebooks ./notebooks
COPY models/loan_models.joblib ./models/loan_models.joblib

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
