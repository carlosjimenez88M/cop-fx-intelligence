"""Servicio de noticias: lee la capa GOLD (sqlite) de la última corrida.

Mapea cada fila a ``NewsItem`` y calcula la importancia con la heurística
determinista del dominio (``topic_taxonomy.article_importance``), ordenando por
ella. Solo lectura; corre en threadpool.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi.concurrency import run_in_threadpool

from cop_fx.analysis.topic_taxonomy import article_importance
from cop_fx.api.schemas import NewsItem, NewsList
from cop_fx.paths import DATA_DIR

if TYPE_CHECKING:
    from pathlib import Path

_GOLD_DB = DATA_DIR / "cnn_articles.db"


class NewsService:
    """Lecturas sobre las noticias clasificadas en la capa GOLD."""

    def __init__(self, db_path: Path = _GOLD_DB) -> None:
        self._db_path = db_path

    async def latest(self, limit: int, *, material_only: bool) -> NewsList:
        return await run_in_threadpool(self._read_sync, limit, material_only)

    # ------------------------------------------------------------------

    def _read_sync(self, limit: int, material_only: bool) -> NewsList:
        if not self._db_path.exists():
            return NewsList(count=0, items=[])
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            if not self._table_exists(conn, "articles"):
                return NewsList(count=0, items=[])
            rows = [dict(r) for r in conn.execute("SELECT * FROM articles")]

        items = [self._to_item(row) for row in rows]
        if material_only:
            items = [item for item in items if item.fx_relevance != "none"]
        items.sort(key=lambda i: i.importance or 0.0, reverse=True)
        items = items[:limit]
        return NewsList(count=len(items), items=items)

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _to_item(row: dict[str, Any]) -> NewsItem:
        published_raw = str(row.get("published_at") or "")
        published: datetime | None = None
        if published_raw:
            try:
                published = datetime.fromisoformat(published_raw)
            except ValueError:
                published = None
        return NewsItem(
            title=str(row.get("title") or ""),
            source=row.get("source") or None,
            published_at=published,
            topic=row.get("topic") or None,
            fx_channel=row.get("fx_channel") or None,
            fx_relevance=row.get("fx_relevance") or None,
            severity=row.get("severity") or None,
            bullish_cop=bool(row["bullish_cop"]) if row.get("bullish_cop") is not None else None,
            importance=article_importance(row),
        )
