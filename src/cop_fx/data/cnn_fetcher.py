"""Scraper de CNN Español Colombia — HTML primario, RSS como complemento.

Historia del bug que corrige este módulo
-----------------------------------------
El diseño anterior trataba el **RSS** (``/colombia/feed/``) como fuente
*primaria* y el scraping HTML como simple *fallback*. Pero esos feeds hoy
responden **404**: el camino primario estaba muerto y el sistema dependía de un
fallback que el código describía como "por si acaso".

La página de listado HTML, en cambio, **sí deja extraer información**: expone
~40 tarjetas ``.container__item`` con título, enlace y fecha embebida en la URL.
Así que aquí se invierte la jerarquía:

  - **HTML = fuente primaria** (``CNNHTMLParser``): lo que de verdad funciona.
  - **RSS = complemento de mejor esfuerzo** (``CNNRSSParser``): si algún día
    revive aporta autor/summary "gratis", pero su ausencia es silenciosa.

Convenciones de color (vía ``cop_fx.logger``):
  green  — artículos extraídos correctamente.
  yellow — complemento vacío / estructura sospechosa.
  red    — cero artículos (revisar conectividad o cambio de maquetado).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar

import feedparser
from bs4 import BeautifulSoup, Tag

from cop_fx.data.http import BROWSER_HEADERS, fetch_text
from cop_fx.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------


@dataclass
class CNNArticle:
    """Un artículo extraído de CNN Español Colombia."""

    title: str
    url: str
    published_at: datetime
    author: str = ""
    summary: str = ""
    source: str = "CNN Español Colombia"
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utilidades de parseo compartidas
# ---------------------------------------------------------------------------


def _date_from_struct_time(parsed: object) -> datetime | None:
    """Convierte el ``time.struct_time`` de feedparser a ``datetime`` UTC."""
    if not parsed:
        return None
    return datetime(
        parsed.tm_year,  # type: ignore[attr-defined]
        parsed.tm_mon,  # type: ignore[attr-defined]
        parsed.tm_mday,  # type: ignore[attr-defined]
        parsed.tm_hour,  # type: ignore[attr-defined]
        parsed.tm_min,  # type: ignore[attr-defined]
        parsed.tm_sec,  # type: ignore[attr-defined]
        tzinfo=UTC,
    )


# ---------------------------------------------------------------------------
# Fuente primaria — HTML
# ---------------------------------------------------------------------------


class CNNHTMLParser:
    """Extrae artículos del listado HTML de CNN Colombia (fuente primaria).

    Estructura de la página: tarjetas ``<... class="container__item"
    data-open-link="/YYYY/MM/DD/...">`` con:
      - título  → ``span.container__headline-text``
      - url     → atributo ``data-open-link`` de la tarjeta
      - fecha   → derivada del patrón ``/YYYY/MM/DD/`` en la URL
      - autor   → no está en el listado; se enriquece con :meth:`fetch_author`.
    """

    _BASE_URL = "https://cnnespanol.cnn.com"
    _CARD_SELECTOR = ".container__item[data-open-link]"
    _TITLE_SELECTOR = "span.container__headline-text"
    # Las URLs con /video/ o /gallery/ no tienen cuerpo de texto que clasificar.
    _NON_ARTICLE = ("/video/", "/gallery/", "/fotos/")
    _AUTHOR_SELECTORS: ClassVar[tuple[str, ...]] = (
        "span.byline__name",
        "span.vossi-byline__name",
        ".byline__names",
    )
    _DATE_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")
    _AUTHOR_PREFIX_RE = re.compile(r"(?i)^por\s+")

    def __init__(self, page_url: str, *, skip_non_articles: bool = True) -> None:
        self._url = page_url
        self._skip_non_articles = skip_non_articles

    def fetch_and_parse(self) -> list[CNNArticle]:
        """Descarga el listado y devuelve las tarjetas parseadas."""
        html = fetch_text(self._url)
        if html is None:
            return []
        return self._parse(html)

    def fetch_author(self, article_url: str) -> str:
        """Descarga la página del artículo y extrae el nombre del autor."""
        html = fetch_text(article_url)
        if html is None:
            return ""
        soup = BeautifulSoup(html, "lxml")
        for selector in self._AUTHOR_SELECTORS:
            tag = soup.select_one(selector)
            if tag:
                return self._AUTHOR_PREFIX_RE.sub("", tag.get_text(strip=True)).strip()
        return ""

    # ------------------------------------------------------------------

    def _parse(self, html: str) -> list[CNNArticle]:
        soup = BeautifulSoup(html, "lxml")
        cards = soup.select(self._CARD_SELECTOR)
        if not cards:
            logger.warning(
                "Ninguna tarjeta '%s' — el maquetado de CNN pudo cambiar",
                self._CARD_SELECTOR,
            )
            return []

        articles = [a for card in cards if (a := self._card_to_article(card)) is not None]
        logger.success("HTML: %d artículos de %d tarjetas", len(articles), len(cards))  # type: ignore[attr-defined]
        return articles

    def _card_to_article(self, card: Tag) -> CNNArticle | None:
        title_tag = card.select_one(self._TITLE_SELECTOR)
        relative_url = card.get("data-open-link") or ""
        if not title_tag or not isinstance(relative_url, str) or not relative_url:
            return None
        if self._skip_non_articles and any(p in relative_url for p in self._NON_ARTICLE):
            return None

        url = relative_url if relative_url.startswith("http") else self._BASE_URL + relative_url
        return CNNArticle(
            title=title_tag.get_text(strip=True),
            url=url,
            published_at=self._date_from_url(url),
            author="",  # se enriquece aparte vía fetch_author()
        )

    def _date_from_url(self, url: str) -> datetime:
        """Extrae la fecha del patrón ``/YYYY/MM/DD/`` en la URL de CNN."""
        match = self._DATE_RE.search(url)
        if match:
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day), tzinfo=UTC)
            except ValueError:
                pass
        return datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# Complemento — RSS (mejor esfuerzo)
# ---------------------------------------------------------------------------


class CNNRSSParser:
    """Parsea ``CNNArticle`` desde un feed RSS, si el feed existe.

    A la fecha los feeds de CNN Español responden 404; esta clase queda como
    complemento de bajo costo: si reviven, aportan autor y summary que el
    listado HTML no trae. Su fallo es silencioso (warning, nunca error).
    """

    def __init__(self, feed_url: str) -> None:
        self._url = feed_url

    def parse(self) -> list[CNNArticle]:
        """Descarga el feed y devuelve los artículos (``[]`` si no hay)."""
        feed = feedparser.parse(self._url)
        if not feed.entries:
            logger.warning("RSS sin entradas (probable 404): %s", self._url)
            return []
        articles = [self._to_article(e) for e in feed.entries]
        logger.success("RSS: %d artículos de %s", len(articles), self._url)  # type: ignore[attr-defined]
        return articles

    def _to_article(self, entry: feedparser.FeedParserDict) -> CNNArticle:
        author = getattr(entry, "author", "") or ", ".join(
            a.get("name", "") for a in getattr(entry, "authors", [])
        )
        summary = BeautifulSoup(entry.get("summary", ""), "lxml").get_text(" ", strip=True)
        published = _date_from_struct_time(getattr(entry, "published_parsed", None))
        return CNNArticle(
            title=entry.get("title", "").strip(),
            url=entry.get("link", ""),
            published_at=published or datetime.now(tz=UTC),
            author=author.strip(),
            summary=summary,
            tags=[t.get("term", "") for t in entry.get("tags", [])],
        )


# ---------------------------------------------------------------------------
# Fetcher público — compone HTML (primario) + RSS (complemento)
# ---------------------------------------------------------------------------


class CNNColombiaFetcher:
    """Agrega artículos de CNN Español Colombia: HTML primario + RSS complemento.

    Uso::

        fetcher = CNNColombiaFetcher(max_articles=100)
        articles: list[CNNArticle] = fetcher.fetch()
    """

    _HTML_URL = "https://cnnespanol.cnn.com/colombia/"
    # Conservados por si CNN revive sus feeds; hoy ambos devuelven 404.
    _RSS_URLS: ClassVar[tuple[str, ...]] = (
        "https://cnnespanol.cnn.com/colombia/feed/",
        "https://cnnespanol.cnn.com/feed/",
    )
    _HEADERS: ClassVar[dict[str, str]] = BROWSER_HEADERS

    def __init__(self, max_articles: int = 50, *, use_rss: bool = True) -> None:
        self._max = max_articles
        self._use_rss = use_rss
        self._html = CNNHTMLParser(self._HTML_URL)
        self._rss_parsers = [CNNRSSParser(u) for u in self._RSS_URLS]

    def fetch(self, *, enrich_authors: bool = False) -> list[CNNArticle]:
        """Devuelve artículos únicos, más recientes primero.

        Args:
            enrich_authors: si es ``True``, descarga cada página de artículo para
                completar el autor (una petición extra por artículo) — úsalo solo
                sobre un ``max_articles`` pequeño.
        """
        logger.info("CNNColombiaFetcher: inicio (max=%d)", self._max)

        html_articles = self._html.fetch_and_parse()
        rss_articles = self._fetch_rss() if self._use_rss else []

        # HTML primero: gana en el dedup y manda el orden base.
        merged = self._deduplicate(html_articles, rss_articles)

        if enrich_authors:
            self._enrich_authors(merged)

        if merged:
            with_author = sum(1 for a in merged if a.author)
            logger.success(  # type: ignore[attr-defined]
                "CNNColombiaFetcher: %d únicos | con autor: %d/%d",
                len(merged),
                with_author,
                len(merged),
            )
        else:
            logger.error("CNNColombiaFetcher: 0 artículos — revisar conectividad o maquetado")
        return merged

    # ------------------------------------------------------------------

    def _fetch_rss(self) -> list[CNNArticle]:
        """Primer feed con entradas gana; si todos fallan devuelve ``[]``."""
        for parser in self._rss_parsers:
            articles = parser.parse()
            if articles:
                return articles
        return []

    def _enrich_authors(self, articles: list[CNNArticle]) -> None:
        """Completa autores faltantes descargando cada página (muta in-place)."""
        to_enrich = [a for a in articles if not a.author]
        logger.info("Enriqueciendo autor en %d artículos...", len(to_enrich))
        enriched = 0
        for article in to_enrich:
            author = self._html.fetch_author(article.url)
            if author:
                article.author = author
                enriched += 1
        if enriched:
            logger.success("Autores completados: %d/%d", enriched, len(to_enrich))  # type: ignore[attr-defined]
        else:
            logger.warning("Sin autores en %d páginas", len(to_enrich))

    def _deduplicate(
        self, primary: list[CNNArticle], secondary: list[CNNArticle]
    ) -> list[CNNArticle]:
        """Une ambas listas por URL (sin barra final), ordena por fecha desc."""
        seen: set[str] = set()
        merged: list[CNNArticle] = []
        for article in primary + secondary:
            key = article.url.rstrip("/")
            if key and key not in seen:
                seen.add(key)
                merged.append(article)
        merged.sort(key=lambda a: a.published_at, reverse=True)
        return merged[: self._max]
