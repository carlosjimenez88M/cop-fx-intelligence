"""Endpoints para disparar y monitorear corridas del pipeline.

El POST encola y responde **202 Accepted** de inmediato (la corrida vive en
segundo plano); el GET consulta el estado por ``job_id``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, status

from cop_fx.api.dependencies import get_pipeline_service
from cop_fx.api.schemas import PipelineJob, PipelineRunRequest
from cop_fx.api.services.pipeline_service import PipelineService

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

ServiceDep = Annotated[PipelineService, Depends(get_pipeline_service)]


@router.post(
    "/runs",
    response_model=PipelineJob,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Disparar una corrida del pipeline (asíncrona)",
)
async def start_run(
    service: ServiceDep,
    background_tasks: BackgroundTasks,
    request: PipelineRunRequest | None = None,
) -> PipelineJob:
    payload = request or PipelineRunRequest()
    job = await service.enqueue(payload.run_date)
    background_tasks.add_task(service.run_job, job.job_id)
    return job


@router.get("/runs/{job_id}", response_model=PipelineJob, summary="Estado de una corrida")
async def get_run(service: ServiceDep, job_id: str) -> PipelineJob:
    return await service.get_job(job_id)
