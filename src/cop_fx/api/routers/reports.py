"""Endpoints de reportes diarios en Markdown."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path

from cop_fx.api.dependencies import get_reports_service
from cop_fx.api.schemas import ReportDetail, ReportList
from cop_fx.api.services.protocols import ReportsServiceProtocol

router = APIRouter(prefix="/reports", tags=["reports"])

ServiceDep = Annotated[ReportsServiceProtocol, Depends(get_reports_service)]

_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


@router.get("", response_model=ReportList, summary="Listado de reportes")
async def list_reports(service: ServiceDep) -> ReportList:
    return await service.list_reports()


@router.get("/{run_date}", response_model=ReportDetail, summary="Reporte por fecha")
async def get_report(
    service: ServiceDep,
    run_date: Annotated[str, Path(pattern=_DATE_PATTERN, examples=["2026-06-18"])],
) -> ReportDetail:
    return await service.get_report(run_date)
