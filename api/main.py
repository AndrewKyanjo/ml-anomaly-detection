"""Serve saved loan predictions through FastAPI."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from sqlite3 import Error as DatabaseError

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as PathParameter
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.inference import load_predictor
from src.service import predict_and_store, predict_saved_records
from src.storage import (
    get_record, initialize_database, list_predictions, query_records, record_outcome,
)

from .schemas import (
    HealthResponse, OutcomeRequest, PredictRequest, PredictResponse,
    PredictionPage, RecordPage, SavedPrediction, StoredPredictRequest,
)
from .frontend import serve_frontend


def create_app(
    model_path: str | Path | None = None,
    database_path: str | Path | None = None,
    frontend_path: str | Path | None = None,
) -> FastAPI:
    """Load one local artifact at startup; routes only validate and delegate."""
    default_path = Path(__file__).resolve().parents[1] / "models/loan_models.joblib"
    artifact = (
        model_path
        if model_path is not None
        else os.getenv("MODEL_PATH", str(default_path))
    )
    database = Path(database_path or os.getenv("DATABASE_PATH", str(default_path.parent.parent / "data/app.sqlite3")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.database_ready = False
        try:
            initialize_database(database)
            app.state.database_ready = True
        except (DatabaseError, OSError, ValueError) as exc:
            logging.getLogger(__name__).warning("Storage unavailable: %s", exc)
        try:
            app.state.predictor = load_predictor(artifact)
        except (FileNotFoundError, ValueError) as exc:
            logging.getLogger(__name__).warning("Predictions unavailable: %s", exc)
            app.state.predictor = None
        yield
        app.state.predictor = None

    app = FastAPI(title="Loan Prediction API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip() for origin in os.getenv(
                "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",") if origin.strip()
        ],
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type"],
    )
    serve_frontend(
        app,
        Path(frontend_path or os.getenv(
            "FRONTEND_PATH", str(default_path.parent.parent / "frontend/dist")
        )),
    )

    def require_database():
        if not getattr(app.state, "database_ready", False):
            raise HTTPException(status_code=503, detail="Storage is unavailable.")

    def require_predictor():
        require_database()
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(status_code=503, detail="No model is loaded. Run model training and restart the API.")
        return predictor

    @app.exception_handler(DatabaseError)
    async def database_error(request: Request, exc: DatabaseError):
        logging.getLogger(__name__).error("Storage operation failed: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "Storage is unavailable."})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Invalid inputs can contain NaN, which cannot appear in a JSON response.
        details = [
            {key: value for key, value in error.items() if key != "input"}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422, content=jsonable_encoder({"detail": details})
        )

    @app.get("/health", response_model=HealthResponse)
    def health(response: Response):
        predictor = getattr(app.state, "predictor", None)
        if predictor is None or not getattr(app.state, "database_ready", False):
            response.status_code = 503
            return {"status": "unavailable", "models": [], "default_model": None}
        return {
            "status": "ok",
            "models": list(predictor.pipelines),
            "default_model": predictor.default_model,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest):
        predictor = require_predictor()
        try:
            return predict_and_store(
                predictor, database,
                [record.model_dump() for record in payload.records], model=payload.model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/predict/records", response_model=PredictResponse)
    def predict_records(payload: StoredPredictRequest):
        predictor = require_predictor()
        try:
            return predict_saved_records(predictor, database, payload.record_ids, payload.model)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/records", response_model=RecordPage)
    def records(
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        loan_status: int | None = Query(default=None, ge=0, le=1),
        dataset_id: int | None = Query(default=None, ge=1),
    ):
        require_database()
        return query_records(database, limit=limit, offset=offset, loan_status=loan_status, dataset_id=dataset_id)

    @app.get("/records/{record_id}")
    def record(record_id: int = PathParameter(ge=1)):
        require_database()
        row = get_record(database, record_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Record not found.")
        return row

    @app.get("/predictions", response_model=PredictionPage)
    def predictions(
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        observed: bool | None = None,
    ):
        require_database()
        return list_predictions(database, limit=limit, offset=offset, observed=observed)

    @app.patch("/predictions/{prediction_id}/outcome", response_model=SavedPrediction)
    def outcome(payload: OutcomeRequest, prediction_id: int = PathParameter(ge=1)):
        require_database()
        row = record_outcome(database, prediction_id, payload.observed_label)
        if row is None:
            raise HTTPException(status_code=404, detail="Prediction not found.")
        return row

    return app


app = create_app()
