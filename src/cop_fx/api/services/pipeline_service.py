"""Servicio de pipeline: dispara corridas en segundo plano y reporta su estado.

El pipeline es caro (varias llamadas a LLM): no se puede ejecutar de forma
síncrona dentro de un request. El patrón es fire-and-forget con seguimiento:

    POST  /pipeline/runs        → encola, responde 202 + job_id (status="queued")
    GET   /pipeline/runs/{id}   → consulta el estado (running/succeeded/failed)

El estado vive en un registro en memoria, protegido por un lock (las corridas
de fondo viven en threads). Para producción multi-réplica habría que mover el
registro a un store compartido; la interfaz no cambiaría (Open/Closed).
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import date
from typing import Any

from fastapi.concurrency import run_in_threadpool

from cop_fx.agents.graph import run_pipeline
from cop_fx.api.errors import ResourceNotFound
from cop_fx.api.schemas import PipelineJob
from cop_fx.logger import get_logger

logger = get_logger(__name__)

RunPipelineFn = Callable[..., Any]


class JobRegistry:
    """Registro thread-safe de jobs en memoria."""

    def __init__(self) -> None:
        self._jobs: dict[str, PipelineJob] = {}
        self._lock = threading.Lock()

    def create(self, run_date: str) -> PipelineJob:
        job = PipelineJob(job_id=uuid.uuid4().hex, status="queued", run_date=run_date)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> PipelineJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                self._jobs[job_id] = current.model_copy(update=fields)


class PipelineService:
    """Encola corridas del pipeline y expone su estado."""

    def __init__(
        self,
        registry: JobRegistry,
        run_pipeline_fn: RunPipelineFn = run_pipeline,
    ) -> None:
        self._registry = registry
        self._run_pipeline = run_pipeline_fn

    async def enqueue(self, run_date: str | None) -> PipelineJob:
        return self._registry.create(run_date or date.today().isoformat())

    async def get_job(self, job_id: str) -> PipelineJob:
        job = self._registry.get(job_id)
        if job is None:
            raise ResourceNotFound(f"No existe el job {job_id}")
        return job

    async def run_job(self, job_id: str) -> None:
        """Ejecuta la corrida (lo invoca BackgroundTasks tras responder 202)."""
        job = self._registry.get(job_id)
        if job is None:
            return
        self._registry.update(job_id, status="running")
        try:
            state = await run_in_threadpool(self._run_pipeline, run_date=job.run_date)
            call = state.get("directional_call") or {}
            errors = state.get("errors") or []
            if errors:
                self._registry.update(
                    job_id, status="failed", detail="; ".join(str(e) for e in errors)
                )
                return
            self._registry.update(
                job_id,
                status="succeeded",
                direction=call.get("direction"),
                confidence=call.get("confidence"),
                report_path=state.get("report_path"),
                detail="Pipeline completado",
            )
        except Exception as exc:  # registramos cualquier fallo en el job
            logger.exception("Job %s falló: %s", job_id, exc)
            self._registry.update(job_id, status="failed", detail=str(exc))
