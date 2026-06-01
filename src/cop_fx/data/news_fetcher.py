"""Fetch economic news via RSS feeds and NewsAPI."""

from __future__ import annotations

from cop_fx.logger import get_logger
from dataclasses import dataclass, field
from datetime import datetime, timezone

import feedparser
import httpx

from cop_fx.config.settings import get_settings

logger = get_logger(__name__)


@dataclass
class Article:
    title: str
    summary: str
    url: str
    published_at: datetime
    source: str
    tags: list[str] = field(default_factory=list)


class NewsFetcher:
    """Aggregates news from configured RSS feeds and optionally NewsAPI."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def fetch(self) -> list[Article]:
        articles: list[Article] = []
        articles.extend(self._fetch_rss())
        if self._settings.newsapi_key:
            articles.extend(self._fetch_newsapi(self._settings.newsapi_key.get_secret_value()))

        # deduplicate by url
        seen: set[str] = set()
        unique: list[Article] = []
        for a in articles:
            if a.url not in seen:
                seen.add(a.url)
                unique.append(a)

        unique.sort(key=lambda a: a.published_at, reverse=True)
        result = unique[: self._settings.news_max_articles]
        logger.info("Fetched %d unique articles", len(result))
        return result

    # ------------------------------------------------------------------
    # RSS
    # ------------------------------------------------------------------

    def _fetch_rss(self) -> list[Article]:
        articles: list[Article] = []
        for feed_url in self._settings.news_rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    published = self._parse_rss_date(entry)
                    articles.append(
                        Article(
                            title=entry.get("title", ""),
                            summary=entry.get("summary", ""),
                            url=entry.get("link", ""),
                            published_at=published,
                            source=feed.feed.get("title", feed_url),
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("RSS feed %s failed: %s", feed_url, exc)
        return articles

    @staticmethod
    def _parse_rss_date(entry: feedparser.FeedParserDict) -> datetime:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            import time

            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return datetime.now(tz=timezone.utc)

    # ------------------------------------------------------------------
    # NewsAPI
    # ------------------------------------------------------------------

    def _fetch_newsapi(self, api_key: str) -> list[Article]:
        params = {
            "q": "dólar Colombia peso COP USD tasa de cambio",
            "language": "es",
            "sortBy": "publishedAt",
            "pageSize": min(self._settings.news_max_articles, 100),
            "apiKey": api_key,
        }
        try:
            with httpx.Client(timeout=20) as client:
                resp = client.get("https://newsapi.org/v2/everything", params=params)
                resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("NewsAPI failed: %s", exc)
            return []

        articles: list[Article] = []
        for item in data.get("articles", []):
            try:
                published = datetime.fromisoformat(
                    item["publishedAt"].replace("Z", "+00:00")
                )
            except (KeyError, ValueError):
                published = datetime.now(tz=timezone.utc)
            articles.append(
                Article(
                    title=item.get("title") or "",
                    summary=item.get("description") or "",
                    url=item.get("url") or "",
                    published_at=published,
                    source=item.get("source", {}).get("name", "NewsAPI"),
                )
            )
        return articles
