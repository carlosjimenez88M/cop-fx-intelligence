"""Liveness/readiness."""

from __future__ import annotations

from fastapi import APIRouter

from cop_fx.api.config import API_VERSION
from cop_fx.api.schemas import HealthResponse

router = APIRouter(tags=["liveness"])


@router.get("/health", response_model=HealthResponse, summary="Prueba de vida")
async def health() -> HealthResponse:
    return HealthResponse(version=API_VERSION)
