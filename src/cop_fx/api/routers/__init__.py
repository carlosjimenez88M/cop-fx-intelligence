"""Routers HTTP. Cada módulo agrupa los endpoints de un recurso."""

from __future__ import annotations

from cop_fx.api.routers import forecast, health, news, pipeline, predictions, reports

__all__ = ["forecast", "health", "news", "pipeline", "predictions", "reports"]
