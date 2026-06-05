"""Contratos de datos para la ingestión de noticias (Fase 0).

`Article` es el contrato único que viaja por todo el sistema:
ingestión → almacenamiento bronze → análisis agentic. Si esto cambia,
cambia de forma explícita en un solo lugar.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Country = Literal["CO", "GLOBAL"]


class Article(BaseModel):
    """Una noticia normalizada, lista para persistir o analizar."""

    id: str = Field(description="Hash estable de la URL — clave de deduplicación")
    title: str
    summary: str = ""
    url: str
    source: str = Field(description="Nombre legible de la fuente")
    country: Country = Field(description="CO = local, GLOBAL = lado USD/macro")
    published_at: datetime
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("published_at", "fetched_at")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        """Toda fecha vive en UTC con tzinfo — evita comparaciones naive/aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @staticmethod
    def make_id(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_raw(
        cls,
        *,
        title: str,
        summary: str,
        url: str,
        source: str,
        country: Country,
        published_at: datetime,
    ) -> "Article":
        """Construye un Article calculando el id a partir de la URL."""
        return cls(
            id=cls.make_id(url),
            title=(title or "").strip(),
            summary=(summary or "").strip(),
            url=url.strip(),
            source=source,
            country=country,
            published_at=published_at,
        )
