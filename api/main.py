"""Serve saved loan predictions through FastAPI."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.inference import load_predictor

from .schemas import HealthResponse, PredictRequest, PredictResponse


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Load one local artifact at startup; routes only validate and delegate."""
    default_path = Path(__file__).resolve().parents[1] / "models/loan_models.joblib"
    artifact = (
        model_path
        if model_path is not None
        else os.getenv("MODEL_PATH", str(default_path))
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = load_predictor(artifact)
        except (FileNotFoundError, ValueError) as exc:
            logging.getLogger(__name__).warning("Predictions unavailable: %s", exc)
            app.state.predictor = None
        yield
        app.state.predictor = None

    app = FastAPI(title="Loan Prediction API", version="0.1.0", lifespan=lifespan)

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
        if predictor is None:
            response.status_code = 503
            return {"status": "unavailable", "models": [], "default_model": None}
        return {
            "status": "ok",
            "models": list(predictor.pipelines),
            "default_model": predictor.default_model,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest):
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="No model is loaded. Run notebook 05 and restart the API.",
            )
        try:
            return predictor.predict(
                [record.model_dump() for record in payload.records], model=payload.model
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
