"""Unit tests de la API FastAPI — routers, DI override y ciclo de jobs."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cop_fx.api import create_app
from cop_fx.api.dependencies import (
    get_pipeline_service,
    get_predictions_service,
)
from cop_fx.api.errors import DataUnavailable
from cop_fx.api.schemas import (
    MetricsResponse,
    PredictionItem,
    PredictionList,
)
from cop_fx.api.services.pipeline_service import JobRegistry, PipelineService


class _FakePredictions:
    """Doble del servicio de predicciones: cero sqlite, datos controlados."""

    def __init__(self, *, has_data: bool = True) -> None:
        self._has_data = has_data

    async def latest(self) -> PredictionItem:
        if not self._has_data:
            raise DataUnavailable("vacío")
        return PredictionItem(run_date="2026-06-18", direction="down", confidence=0.7)

    async def history(self, limit: int | None = None) -> PredictionList:
        item = PredictionItem(run_date="2026-06-18", direction="up", confidence=0.6)
        return PredictionList(count=1, items=[item])

    async def metrics(self) -> MetricsResponse:
        return MetricsResponse(n_total=3, n_evaluated=2, n_decided=2, hit_rate=0.5)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.unit()
def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.unit()
def test_predictions_latest_uses_injected_service() -> None:
    app = create_app()
    app.dependency_overrides[get_predictions_service] = lambda: _FakePredictions()
    client = TestClient(app)

    resp = client.get("/api/v1/predictions/latest")

    assert resp.status_code == 200
    assert resp.json() == {
        "run_date": "2026-06-18",
        "direction": "down",
        "confidence": 0.7,
        **{
            k: None
            for k in (
                "horizon_days",
                "reconciliation",
                "dominant_signal",
                "news_direction",
                "ts_direction",
                "market_direction",
                "latest_rate",
                "rationale",
                "devils_advocate",
                "top_story_title",
                "top_story_source",
                "actual_direction",
                "final_hit",
            )
        },
    }


@pytest.mark.unit()
def test_predictions_latest_503_when_empty() -> None:
    app = create_app()
    app.dependency_overrides[get_predictions_service] = lambda: _FakePredictions(has_data=False)
    client = TestClient(app)

    resp = client.get("/api/v1/predictions/latest")

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "data_unavailable"


@pytest.mark.unit()
def test_report_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/api/v1/reports/1999-01-01")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


@pytest.mark.unit()
def test_report_bad_date_returns_422(client: TestClient) -> None:
    resp = client.get("/api/v1/reports/not-a-date")
    assert resp.status_code == 422


@pytest.mark.unit()
def test_pipeline_job_lifecycle() -> None:
    """POST encola y el background runner (fake pipeline) deja el job succeeded."""

    def fake_run_pipeline(*, run_date: str) -> dict[str, object]:
        return {
            "directional_call": {"direction": "up", "confidence": 0.81},
            "report_path": f"/tmp/report_{run_date}.md",
            "errors": [],
        }

    registry = JobRegistry()
    service = PipelineService(registry, run_pipeline_fn=fake_run_pipeline)

    app = create_app()
    app.dependency_overrides[get_pipeline_service] = lambda: service
    client = TestClient(app)

    started = client.post("/api/v1/pipeline/runs", json={"run_date": "2026-06-20"})
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    assert started.json()["status"] in {"queued", "running", "succeeded"}

    # TestClient corre las BackgroundTasks antes de devolver la respuesta.
    final = client.get(f"/api/v1/pipeline/runs/{job_id}")
    assert final.status_code == 200
    body = final.json()
    assert body["status"] == "succeeded"
    assert body["direction"] == "up"
    assert body["confidence"] == 0.81


@pytest.mark.unit()
def test_pipeline_job_not_found(client: TestClient) -> None:
    resp = client.get("/api/v1/pipeline/runs/deadbeef")
    assert resp.status_code == 404
