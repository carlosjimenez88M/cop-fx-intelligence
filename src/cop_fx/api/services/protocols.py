"""Interfaces (Protocols) que los routers consumen.

Definir protocolos —en vez de importar las clases concretas— permite que los
routers dependan de una abstracción estable (Dependency Inversion / Interface
Segregation): cada router pide solo el protocolo que usa, y los tests inyectan
dobles sin levantar sqlite ni LangGraph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cop_fx.api.schemas import (
        ForecastResponse,
        MetricsResponse,
        NewsList,
        PipelineJob,
        PredictionItem,
        PredictionList,
        ReportDetail,
        ReportList,
    )


class PredictionsServiceProtocol(Protocol):
    async def latest(self) -> PredictionItem: ...
    async def history(self, limit: int | None = None) -> PredictionList: ...
    async def metrics(self) -> MetricsResponse: ...


class ForecastServiceProtocol(Protocol):
    async def compute(self, horizon_days: int | None = None) -> ForecastResponse: ...


class NewsServiceProtocol(Protocol):
    async def latest(self, limit: int, *, material_only: bool) -> NewsList: ...


class ReportsServiceProtocol(Protocol):
    async def list_reports(self) -> ReportList: ...
    async def get_report(self, run_date: str) -> ReportDetail: ...


class PipelineServiceProtocol(Protocol):
    async def enqueue(self, run_date: str | None) -> PipelineJob: ...
    async def get_job(self, job_id: str) -> PipelineJob: ...
