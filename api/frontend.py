"""Serve the compiled dashboard without changing API routes."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles


def serve_frontend(app: FastAPI, directory: Path) -> None:
    """Expose compiled assets and the dashboard's explicit browser routes."""
    directory = directory.resolve()
    assets = directory / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    def dashboard() -> Response:
        index = directory / "index.html"
        if not index.is_file():
            return RedirectResponse("/docs")
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    for route in ("/", "/app", "/app/records", "/app/predict", "/app/batch", "/app/history"):
        app.add_api_route(route, dashboard, methods=["GET"], include_in_schema=False)
