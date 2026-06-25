"""Contratos de entrada/salida de la API (Pydantic v2).

Son DTOs de la capa HTTP, deliberadamente separados de los contratos del
dominio (``cop_fx.contracts``): la API puede evolucionar su forma de respuesta
sin arrastrar el modelo interno, y viceversa.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["down", "up", "neutral"]


class HealthResponse(BaseModel):
    """Prueba de vida de la API."""

    status: Literal["ok"] = "ok"
    service: str = "cop-fx-intelligence"
    version: str


class PredictionItem(BaseModel):
    """Un call direccional histórico tal como vive en el store."""

    run_date: str
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    horizon_days: int | None = None
    reconciliation: str | None = None
    dominant_signal: str | None = None
    news_direction: str | None = None
    ts_direction: str | None = None
    market_direction: str | None = None
    latest_rate: float | None = None
    rationale: str | None = None
    devils_advocate: str | None = None
    top_story_title: str | None = None
    top_story_source: str | None = None
    actual_direction: str | None = None
    final_hit: int | None = None


class PredictionList(BaseModel):
    count: int
    items: list[PredictionItem]


class HitRateBucket(BaseModel):
    bucket: str
    hit_rate: float | None = None
    n: int


class MetricsResponse(BaseModel):
    """Métricas direccionales del track record."""

    n_total: int
    n_evaluated: int
    n_decided: int
    n_abstained: int = 0
    hit_rate: float | None = None
    final_hit_rate: float | None = None
    news_hit_rate: float | None = None
    ts_hit_rate: float | None = None
    market_hit_rate: float | None = None
    hit_rate_by_confidence: list[HitRateBucket] = Field(default_factory=list)


class ForecastPoint(BaseModel):
    ds: datetime
    yhat: float


class ModelDelta(BaseModel):
    model: str
    yhat_final: float
    delta_pct: float


class ForecastResponse(BaseModel):
    """Forecast del ensemble Prophet + ARIMA y el signo implícito."""

    latest_rate: float
    horizon_days: int
    direction: Direction
    models: list[ModelDelta]
    ensemble: list[ForecastPoint]


class NewsItem(BaseModel):
    title: str
    source: str | None = None
    published_at: datetime | None = None
    topic: str | None = None
    fx_channel: str | None = None
    fx_relevance: str | None = None
    severity: str | None = None
    bullish_cop: bool | None = None
    importance: float | None = None


class NewsList(BaseModel):
    count: int
    items: list[NewsItem]


class ReportSummary(BaseModel):
    run_date: str
    filename: str


class ReportList(BaseModel):
    count: int
    items: list[ReportSummary]


class ReportDetail(BaseModel):
    run_date: str
    filename: str
    markdown: str


class PipelineRunRequest(BaseModel):
    """Parámetros para disparar una corrida."""

    run_date: str | None = Field(
        default=None, description="Fecha ISO de la corrida (por defecto hoy)."
    )


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class PipelineJob(BaseModel):
    """Estado de una corrida lanzada en segundo plano."""

    job_id: str
    status: JobStatus
    run_date: str
    detail: str | None = None
    direction: Direction | None = None
    confidence: float | None = None
    report_path: str | None = None
