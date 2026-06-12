"""Extractor de noticias económicas (RSS) → capa bronze."""

from cop_fx.ingestion.news.extractor import (
    fetch_all,
    run_news_ingestion,
    validate_sources,
)
from cop_fx.ingestion.news.schemas import Article
from cop_fx.ingestion.news.sources import SOURCES, NewsSource

__all__ = [
    "SOURCES",
    "Article",
    "NewsSource",
    "fetch_all",
    "run_news_ingestion",
    "validate_sources",
]
