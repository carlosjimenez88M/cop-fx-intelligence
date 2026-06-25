"""Servicio de predicciones: envuelve el ``PredictionStore`` (sqlite/pandas).

El store es síncrono y toca disco; cada método público corre en un threadpool
(`run_in_threadpool`) para no bloquear el event loop de FastAPI.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pandas as pd
from fastapi.concurrency import run_in_threadpool

from cop_fx.api.errors import DataUnavailable
from cop_fx.api.schemas import (
    HitRateBucket,
    MetricsResponse,
    PredictionItem,
    PredictionList,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cop_fx.tracking.predictions import PredictionStore


def _clean(value: Any) -> Any:
    """Normaliza NaN/NaT de pandas a None para que Pydantic no falle."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    return value


def _row_to_item(row: dict[Hashable, Any]) -> PredictionItem:
    cleaned = {str(key): _clean(val) for key, val in row.items()}
    return PredictionItem.model_validate(cleaned)


class PredictionsService:
    """Lecturas sobre el track record de calls direccionales."""

    def __init__(self, store: PredictionStore) -> None:
        self._store = store

    async def latest(self) -> PredictionItem:
        df = await run_in_threadpool(self._store.all)
        if df.empty:
            raise DataUnavailable("No hay predicciones guardadas todavía")
        row = df.sort_values("run_date", ascending=False).iloc[0].to_dict()
        return _row_to_item(row)

    async def history(self, limit: int | None = None) -> PredictionList:
        df = await run_in_threadpool(self._store.all)
        if df.empty:
            return PredictionList(count=0, items=[])
        df = df.sort_values("run_date", ascending=False)
        if limit is not None:
            df = df.head(limit)
        items = [_row_to_item(row) for row in df.to_dict(orient="records")]
        return PredictionList(count=len(items), items=items)

    async def metrics(self) -> MetricsResponse:
        raw = await run_in_threadpool(self._store.metrics)
        buckets = self._buckets(raw.get("hit_rate_by_confidence"))
        return MetricsResponse(
            n_total=int(raw.get("n_total", 0)),
            n_evaluated=int(raw.get("n_evaluated", 0)),
            n_decided=int(raw.get("n_decided", 0)),
            n_abstained=int(raw.get("n_abstained", 0)),
            hit_rate=raw.get("hit_rate"),
            final_hit_rate=raw.get("final_hit_rate"),
            news_hit_rate=raw.get("news_hit_rate"),
            ts_hit_rate=raw.get("ts_hit_rate"),
            market_hit_rate=raw.get("market_hit_rate"),
            hit_rate_by_confidence=buckets,
        )

    @staticmethod
    def _buckets(frame: Any) -> list[HitRateBucket]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return []
        out: list[HitRateBucket] = []
        for label, row in frame.iterrows():
            out.append(
                HitRateBucket(
                    bucket=str(label),
                    hit_rate=_clean(row.get("hit_rate")),
                    n=int(row.get("n", 0)),
                )
            )
        return out
