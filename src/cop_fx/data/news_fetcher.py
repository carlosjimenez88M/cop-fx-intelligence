"""Fetch economic news via RSS feeds, NewsAPI, and CNN Español Colombia."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import feedparser
import httpx

from cop_fx.config.settings import get_settings
from cop_fx.data.cnn_fetcher import CNNArticle, CNNColombiaFetcher
from cop_fx.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Article:
    title: str
    summary: str
    url: str
    published_at: datetime
    source: str
    author: str = ""
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
        articles.extend(self._fetch_cnn())

        # deduplicate by url
        seen: set[str] = set()
        unique: list[Article] = []
        for a in articles:
            if a.url not in seen:
                seen.add(a.url)
                unique.append(a)

        unique.sort(key=lambda a: a.published_at, reverse=True)
        result = unique[: self._settings.news_max_articles]
        logger.success("Fetched %d unique articles total", len(result))  # type: ignore[attr-defined]
        return result

    def _fetch_cnn(self) -> list[Article]:
        """Fetch articles from CNN Español Colombia and convert to Article."""
        try:
            cnn_articles: list[CNNArticle] = CNNColombiaFetcher(
                max_articles=self._settings.news_max_articles
            ).fetch()
            return [
                Article(
                    title=a.title,
                    summary=a.summary,
                    url=a.url,
                    published_at=a.published_at,
                    source=a.source,
                    author=a.author,
                    tags=a.tags,
                )
                for a in cnn_articles
            ]
        except Exception as exc:
            logger.error("CNNColombiaFetcher failed: %s", exc)
            return []

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
            except Exception as exc:
                logger.warning("RSS feed %s failed: %s", feed_url, exc)
        return articles

    @staticmethod
    def _parse_rss_date(entry: feedparser.FeedParserDict) -> datetime:
        if hasattr(entry, "published_parsed") and entry.published_parsed:

            return datetime(*entry.published_parsed[:6], tzinfo=UTC)
        return datetime.now(tz=UTC)

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
        except Exception as exc:
            logger.warning("NewsAPI failed: %s", exc)
            return []

        articles: list[Article] = []
        for item in data.get("articles", []):
            try:
                published = datetime.fromisoformat(
                    item["publishedAt"].replace("Z", "+00:00")
                )
            except (KeyError, ValueError):
                published = datetime.now(tz=UTC)
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
