"""Verify browser routing and development access preserve API contracts."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def dashboard(tmp_path):
    directory = tmp_path / "frontend"
    assets = directory / "assets"
    assets.mkdir(parents=True)
    (directory / "index.html").write_text(
        '<!doctype html><html><body><div id="root">Dashboard</div></body></html>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text('console.log("dashboard");', encoding="utf-8")
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    app = create_app(
        model_path=tmp_path / "missing.joblib",
        database_path=tmp_path / "app.sqlite3",
        frontend_path=directory,
    )
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("route", ["/", "/app", "/app/records", "/app/predict", "/app/batch", "/app/history"])
def test_browser_routes_serve_dashboard(dashboard, route):
    response = dashboard.get(route)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"
    assert 'id="root"' in response.text


def test_compiled_assets_are_served(dashboard):
    response = dashboard.get("/assets/app.js")
    assert response.status_code == 200
    assert 'console.log("dashboard")' in response.text


@pytest.mark.parametrize("route", ["/assets/missing.js", "/assets/%2e%2e%2f%2e%2e%2fsecret.txt", "/app/unknown", "/unknown-api"])
def test_unknown_routes_do_not_receive_dashboard(dashboard, route):
    response = dashboard.get(route)
    assert response.status_code == 404
    assert "private" not in response.text
    assert 'id="root"' not in response.text


def test_records_stays_a_json_api(dashboard):
    response = dashboard.get("/records")
    assert response.status_code == 200
    assert response.json() == {"total": 0, "items": []}


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:5173"])
def test_vite_development_preflight(dashboard, origin):
    response = dashboard.options("/predict", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_unconfigured_origin_is_rejected(dashboard):
    response = dashboard.options("/predict", headers={
        "Origin": "https://untrusted.example",
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_missing_local_build_redirects_to_api_docs(tmp_path):
    app = create_app(
        model_path=tmp_path / "missing.joblib",
        database_path=tmp_path / "app.sqlite3",
        frontend_path=tmp_path / "unbuilt",
    )
    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_cors_origins_can_be_replaced(monkeypatch, tmp_path):
    monkeypatch.setenv("CORS_ORIGINS", "https://dashboard.example")
    app = create_app(
        model_path=tmp_path / "missing.joblib",
        database_path=tmp_path / "app.sqlite3",
        frontend_path=tmp_path / "unbuilt",
    )
    with TestClient(app) as client:
        for origin, expected in [("https://dashboard.example", 200), ("http://localhost:5173", 400)]:
            response = client.options("/records", headers={
                "Origin": origin, "Access-Control-Request-Method": "GET",
            })
            assert response.status_code == expected
