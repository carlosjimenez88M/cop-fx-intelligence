"""Extractor de noticias RSS (Fase 1) — determinista, sin LLM.

Flujo:
    fetch_all(sources)            # descarga + normaliza + dedup
        │
        ├─ persist_bronze(...)    # guarda crudo + normalizado en data/raw/news/<fecha>/
        │
        └─ devuelve list[Article] # listo para la capa de análisis agentic

Decisiones de diseño:
  - El crudo se guarda ANTES de procesar (reproducibilidad: si cambias el
    parser, reprocesas sin re-scrapear).
  - Un feed caído NO tumba la corrida; se registra y se sigue.
  - Cero LLM aquí: parsear, deduplicar y ordenar es determinista y barato.
  - Ruta bronze local = mismo layout que tendrá en GCS:
        local: data/raw/news/2026-06-04/...
        gcs:   gs://<bucket>/raw/news/2026-06-04/...
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import feedparser
import httpx

from cop_fx.ingestion.news.schemas import Article
from cop_fx.ingestion.news.sources import SOURCES, NewsSource
from cop_fx.logger import get_logger

logger = get_logger(__name__)

_USER_AGENT = "cop-fx-intelligence/0.1 (+https://github.com/carlosdaniel/cop-fx-intelligence)"
_TIMEOUT = httpx.Timeout(20.0)


# ---------------------------------------------------------------------------
# Descarga de una fuente
# ---------------------------------------------------------------------------

def _download(url: str) -> bytes | None:
    """GET con timeout y User-Agent. Devuelve None si falla (feed caído)."""
    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        logger.warning("feed no disponible %s: %s", url, exc)
        return None


def _parse_published(entry: feedparser.FeedParserDict) -> datetime:
    if getattr(entry, "published_parsed", None):
        return datetime(*entry.published_parsed[:6], tzinfo=UTC)
    if getattr(entry, "updated_parsed", None):
        return datetime(*entry.updated_parsed[:6], tzinfo=UTC)
    return datetime.now(tz=UTC)


def _parse_feed(raw: bytes, source: NewsSource) -> list[Article]:
    parsed = feedparser.parse(raw)
    articles: list[Article] = []
    for entry in parsed.entries:
        url = entry.get("link", "").strip()
        if not url:
            continue
        articles.append(
            Article.from_raw(
                title=entry.get("title", ""),
                summary=entry.get("summary", ""),
                url=url,
                source=source.name,
                country=source.country,
                published_at=_parse_published(entry),
            )
        )
    return articles


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def fetch_all(
    sources: list[NewsSource] | None = None,
    *,
    max_articles: int = 60,
    raw_sink: dict[str, bytes] | None = None,
) -> list[Article]:
    """Descarga todas las fuentes, normaliza, deduplica y ordena por fecha.

    Si `raw_sink` se pasa, se llena con {source_name: bytes_crudos} para
    poder persistir el bronze.
    """
    sources = sources or SOURCES
    collected: list[Article] = []

    for src in sources:
        raw = _download(src.url)
        if raw is None:
            continue
        if raw_sink is not None:
            raw_sink[src.name] = raw
        found = _parse_feed(raw, src)
        logger.info("%s → %d artículos", src.name, len(found))
        collected.extend(found)

    # Dedup por id (hash de la URL)
    by_id: dict[str, Article] = {}
    for art in collected:
        by_id.setdefault(art.id, art)

    unique = sorted(by_id.values(), key=lambda a: a.published_at, reverse=True)
    result = unique[:max_articles]
    logger.success("Total: %d únicos (de %d brutos)", len(result), len(collected))
    return result


# ---------------------------------------------------------------------------
# Persistencia bronze
# ---------------------------------------------------------------------------

def persist_bronze(
    articles: list[Article],
    raw_sink: dict[str, bytes],
    *,
    base_dir: str = "data/raw/news",
    run_date: str | None = None,
) -> Path:
    """Guarda el crudo por fuente + el normalizado JSON. Devuelve el directorio."""
    run_date = run_date or date.today().isoformat()
    out = Path(base_dir) / run_date
    out.mkdir(parents=True, exist_ok=True)

    # 1) crudo por fuente (para reproducibilidad)
    for name, raw in raw_sink.items():
        safe = name.lower().replace(" ", "_").replace("—", "-").replace("/", "_")
        (out / f"{safe}.xml").write_bytes(raw)

    # 2) normalizado
    payload = [json.loads(a.model_dump_json()) for a in articles]
    (out / "articles.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Bronze persistido en %s (%d artículos)", out, len(articles))
    return out


def run_news_ingestion(
    *,
    sources: list[NewsSource] | None = None,
    max_articles: int = 60,
    base_dir: str = "data/raw/news",
    run_date: str | None = None,
    persist: bool = True,
) -> list[Article]:
    """Punto de entrada de la Fase 1: descarga, persiste bronze y devuelve artículos."""
    raw_sink: dict[str, bytes] = {}
    articles = fetch_all(sources, max_articles=max_articles, raw_sink=raw_sink)
    if persist and articles:
        persist_bronze(articles, raw_sink, base_dir=base_dir, run_date=run_date)
    return articles


# ---------------------------------------------------------------------------
# Validación de fuentes (para que NO confíes en la lista a ciegas)
# ---------------------------------------------------------------------------

def validate_sources(sources: list[NewsSource] | None = None) -> dict[str, int]:
    """Devuelve {nombre: n_articulos}. 0 = feed caído o vacío."""
    sources = sources or SOURCES
    report: dict[str, int] = {}
    for src in sources:
        raw = _download(src.url)
        report[src.name] = len(_parse_feed(raw, src)) if raw else 0
    return report
