"""CNN Español Colombia scraper — RSS primary, HTML fallback.

Colour conventions (via cop_fx.logger):
  green  (success) — articles fetched / parsed correctly
  yellow (warning) — fallback used, partial results, empty selectors
  red    (error)   — HTTP failure, timeout, no articles at all
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar

import feedparser
import httpx
from bs4 import BeautifulSoup, Tag

from cop_fx.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CNNArticle:
    """Single article scraped from CNN Español Colombia."""

    title: str
    url: str
    published_at: datetime
    author: str = ""
    summary: str = ""
    source: str = "CNN Español Colombia"
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RSS parser
# ---------------------------------------------------------------------------


class CNNRSSParser:
    """Parses CNNArticles from a feedparser-compatible RSS URL."""

    def __init__(self, feed_url: str) -> None:
        self._url = feed_url

    def parse(self) -> list[CNNArticle]:
        """Fetch the feed and return parsed articles."""
        feed = feedparser.parse(self._url)
        if not feed.entries:
            logger.warning("RSS feed returned 0 entries: %s", self._url)
            return []

        articles = [self._to_article(e) for e in feed.entries]
        logger.success("RSS: %d articles from %s", len(articles), self._url)  # type: ignore[attr-defined]
        return articles

    # ------------------------------------------------------------------

    def _to_article(self, entry: feedparser.FeedParserDict) -> CNNArticle:
        author = getattr(entry, "author", "")
        if not author:
            author = ", ".join(a.get("name", "") for a in getattr(entry, "authors", []))

        summary = BeautifulSoup(entry.get("summary", ""), "lxml").get_text(" ", strip=True)
        tags = [t.get("term", "") for t in entry.get("tags", [])]

        return CNNArticle(
            title=entry.get("title", "").strip(),
            url=entry.get("link", ""),
            published_at=self._parse_date(entry),
            author=author.strip(),
            summary=summary,
            tags=tags,
        )

    @staticmethod
    def _parse_date(entry: feedparser.FeedParserDict) -> datetime:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            parsed = entry.published_parsed
            return datetime(
                parsed.tm_year,
                parsed.tm_mon,
                parsed.tm_mday,
                parsed.tm_hour,
                parsed.tm_min,
                parsed.tm_sec,
                tzinfo=UTC,
            )
        return datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------


class CNNHTMLParser:
    """Scrapes CNNArticles by downloading and parsing the CNN Colombia page.

    CNN Español uses `.container__item` cards with:
    - title  → ``span.container__headline-text``
    - url    → ``data-open-link`` attribute on the card ``<li>``
    - date   → extracted from the URL path ``/YYYY/MM/DD/``
    - author → not present in the listing; use ``fetch_author()`` per article.
    """

    _BASE_URL = "https://cnnespanol.cnn.com"
    # Article-page selectors for author enrichment
    _AUTHOR_SELECTORS = (
        "span.byline__name",
        "span.vossi-byline__name",
        ".byline__names",
    )

    def __init__(self, page_url: str, headers: dict[str, str]) -> None:
        self._url = page_url
        self._headers = headers

    def fetch_and_parse(self) -> list[CNNArticle]:
        """Download the listing page and return parsed article cards."""
        html = self._download(self._url)
        if not html:
            return []
        return self._parse(html)

    def fetch_author(self, article_url: str) -> str:
        """Fetch an individual article page and extract the author name."""
        html = self._download(article_url)
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        for selector in self._AUTHOR_SELECTORS:
            tag = soup.select_one(selector)
            if tag:
                return re.sub(r"(?i)^por\s+", "", tag.get_text(strip=True)).strip()
        return ""

    # ------------------------------------------------------------------

    def _download(self, url: str) -> str:
        try:
            with httpx.Client(timeout=30, follow_redirects=True, headers=self._headers) as client:
                resp = client.get(url)
                resp.raise_for_status()
            logger.success(  # type: ignore[attr-defined]
                "HTML: fetched %d bytes from %s", len(resp.text), url
            )
            return resp.text
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP %d fetching %s", exc.response.status_code, url)
        except httpx.TimeoutException:
            logger.error("Timeout fetching %s", url)
        except Exception as exc:
            logger.error("Unexpected error fetching HTML: %s", exc)
        return ""

    def _parse(self, html: str) -> list[CNNArticle]:
        soup = BeautifulSoup(html, "lxml")
        cards = soup.select(".container__item[data-open-link]")
        if not cards:
            logger.warning(
                "No .container__item[data-open-link] cards found — "
                "the page structure may have changed"
            )
            return []
        logger.info("Found %d article cards in HTML", len(cards))
        articles = [a for card in cards if (a := self._card_to_article(card)) is not None]
        logger.success("HTML: extracted %d articles", len(articles))  # type: ignore[attr-defined]
        return articles

    def _card_to_article(self, card: Tag) -> CNNArticle | None:
        title_tag = card.find("span", class_="container__headline-text")
        if not title_tag:
            return None

        title = title_tag.get_text(strip=True)
        relative_url: str = card.get("data-open-link") or ""  # type: ignore[assignment]
        if not relative_url:
            return None
        url = self._BASE_URL + relative_url if not relative_url.startswith("http") else relative_url

        return CNNArticle(
            title=title,
            url=url,
            published_at=self._date_from_url(url),
            author="",  # enriched separately via fetch_author()
            summary="",
        )

    @staticmethod
    def _date_from_url(url: str) -> datetime:
        """Extract publication date from CNN URL path pattern /YYYY/MM/DD/."""
        match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
        if match:
            try:
                return datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    tzinfo=UTC,
                )
            except ValueError:
                pass
        return datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# Public fetcher — composes RSS + HTML
# ---------------------------------------------------------------------------


class CNNColombiaFetcher:
    """Aggregate CNN Español Colombia articles using RSS (primary) + HTML (fallback).

    Usage::

        fetcher = CNNColombiaFetcher(max_articles=100)
        articles: list[CNNArticle] = fetcher.fetch()
    """

    _RSS_URL = "https://cnnespanol.cnn.com/colombia/feed/"
    _HTML_URL = "https://cnnespanol.cnn.com/colombia/"
    _FALLBACK_RSS = "https://cnnespanol.cnn.com/feed/"
    _HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
    }

    def __init__(self, max_articles: int = 50) -> None:
        self._max = max_articles
        self._rss = CNNRSSParser(self._RSS_URL)
        self._fallback_rss = CNNRSSParser(self._FALLBACK_RSS)
        self._html = CNNHTMLParser(self._HTML_URL, self._HEADERS)

    def fetch(self, enrich_authors: bool = False) -> list[CNNArticle]:
        """Return deduplicated, newest-first articles from CNN Colombia.

        Args:
            enrich_authors: When True, fetch each article page to extract the
                author name.  Makes one additional HTTP request per article —
                use sparingly (e.g. top 10 only via ``max_articles``).
        """
        logger.info("CNNColombiaFetcher: starting (max=%d)", self._max)

        rss = self._rss.parse()
        if not rss:
            logger.warning("Colombia RSS empty — trying main CNN Español feed")
            rss = self._fallback_rss.parse()

        html = self._html.fetch_and_parse()
        merged = self._deduplicate(rss, html)

        if enrich_authors:
            merged = self._enrich_authors(merged)

        if merged:
            with_author = sum(1 for a in merged if a.author)
            logger.success(  # type: ignore[attr-defined]
                "CNNColombiaFetcher: %d unique articles ready | with author: %d/%d",
                len(merged),
                with_author,
                len(merged),
            )
        else:
            logger.error("CNNColombiaFetcher: 0 articles fetched — check connectivity")

        return merged

    def _enrich_authors(self, articles: list[CNNArticle]) -> list[CNNArticle]:
        """Fetch individual article pages to fill in missing authors."""
        to_enrich = [a for a in articles if not a.author]
        logger.info("Enriching authors for %d articles...", len(to_enrich))
        enriched = 0
        for article in to_enrich:
            author = self._html.fetch_author(article.url)
            if author:
                article.author = author
                enriched += 1
        if enriched:
            logger.success("Author enrichment: %d/%d filled", enriched, len(to_enrich))  # type: ignore[attr-defined]
        else:
            logger.warning("Author enrichment: no authors found in %d pages", len(to_enrich))
        return articles

    # ------------------------------------------------------------------

    def _deduplicate(self, rss: list[CNNArticle], html: list[CNNArticle]) -> list[CNNArticle]:
        seen: set[str] = set()
        merged: list[CNNArticle] = []
        for article in rss + html:
            key = article.url.rstrip("/")
            if key and key not in seen:
                seen.add(key)
                merged.append(article)
        merged.sort(key=lambda a: a.published_at, reverse=True)
        return merged[: self._max]
