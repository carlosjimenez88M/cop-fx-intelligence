"""Endpoint de noticias clasificadas (capa GOLD)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from cop_fx.api.dependencies import get_news_service
from cop_fx.api.schemas import NewsList
from cop_fx.api.services.protocols import NewsServiceProtocol

router = APIRouter(prefix="/news", tags=["news"])

ServiceDep = Annotated[NewsServiceProtocol, Depends(get_news_service)]


@router.get("", response_model=NewsList, summary="Noticias clasificadas por importancia")
async def news(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    material_only: Annotated[bool, Query(description="Solo noticias con relevancia FX")] = True,
) -> NewsList:
    return await service.latest(limit, material_only=material_only)
