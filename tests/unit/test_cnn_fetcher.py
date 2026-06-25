"""Unit tests del scraper de CNN — HTML primario, RSS complemento."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cop_fx.data import cnn_fetcher
from cop_fx.data.cnn_fetcher import CNNArticle, CNNColombiaFetcher, CNNHTMLParser

# Listado HTML mínimo: 2 artículos reales + 1 video (debe filtrarse).
_LISTING_HTML = """
<html><body>
  <li class="container__item" data-open-link="/2026/06/24/colombia/dolar-sube-orix">
    <span class="container__headline-text">El dólar sube tras la decisión del Banco</span>
  </li>
  <li class="container__item" data-open-link="/2026/06/23/economia/inflacion-junio">
    <span class="container__headline-text">La inflación de junio sorprende al mercado</span>
  </li>
  <li class="container__item" data-open-link="/2026/06/24/colombia/video/entrevista">
    <span class="container__headline-text">Video: entrevista al ministro</span>
  </li>
  <li class="container__item">
    <span class="container__headline-text">Tarjeta sin enlace, se descarta</span>
  </li>
</body></html>
"""


@pytest.mark.unit()
def test_html_parser_extracts_cards_and_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cnn_fetcher, "fetch_text", lambda url, **_: _LISTING_HTML)
    parser = CNNHTMLParser("https://cnnespanol.cnn.com/colombia/")

    articles = parser.fetch_and_parse()

    # video y tarjeta-sin-enlace descartados → quedan 2
    assert len(articles) == 2
    first = articles[0]
    assert first.title.startswith("El dólar sube")
    assert first.url == "https://cnnespanol.cnn.com/2026/06/24/colombia/dolar-sube-orix"
    assert first.published_at == datetime(2026, 6, 24, tzinfo=UTC)


@pytest.mark.unit()
def test_html_parser_handles_missing_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cnn_fetcher, "fetch_text", lambda url, **_: "<html></html>")
    assert CNNHTMLParser("x").fetch_and_parse() == []


@pytest.mark.unit()
def test_html_parser_returns_empty_on_download_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cnn_fetcher, "fetch_text", lambda url, **_: None)
    assert CNNHTMLParser("x").fetch_and_parse() == []


@pytest.mark.unit()
def test_fetcher_prefers_html_and_dedupes_with_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    html = [
        CNNArticle("HTML nuevo", "https://cnn.com/a", datetime(2026, 6, 24, tzinfo=UTC)),
        CNNArticle("HTML viejo", "https://cnn.com/b", datetime(2026, 6, 20, tzinfo=UTC)),
    ]
    # 'a/' duplica al de HTML (misma URL salvo barra) → no se cuenta dos veces.
    rss = [
        CNNArticle("RSS dup", "https://cnn.com/a/", datetime(2026, 6, 24, tzinfo=UTC), author="Yo"),
        CNNArticle("RSS único", "https://cnn.com/c", datetime(2026, 6, 22, tzinfo=UTC)),
    ]
    fetcher = CNNColombiaFetcher(max_articles=10)
    monkeypatch.setattr(fetcher._html, "fetch_and_parse", lambda: html)
    monkeypatch.setattr(fetcher, "_fetch_rss", lambda: rss)

    result = fetcher.fetch()

    assert [a.url for a in result] == [
        "https://cnn.com/a",  # 24 jun (HTML gana el dedup)
        "https://cnn.com/c",  # 22 jun
        "https://cnn.com/b",  # 20 jun
    ]
    assert result[0].title == "HTML nuevo"  # la versión HTML, no la RSS


@pytest.mark.unit()
def test_fetcher_can_disable_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = CNNColombiaFetcher(max_articles=5, use_rss=False)
    monkeypatch.setattr(
        fetcher._html,
        "fetch_and_parse",
        lambda: [CNNArticle("solo html", "https://cnn.com/a", datetime(2026, 6, 24, tzinfo=UTC))],
    )
    # Si _fetch_rss se invocara, fallaría el test (no debe llamarse).
    monkeypatch.setattr(
        fetcher, "_fetch_rss", lambda: (_ for _ in ()).throw(AssertionError("RSS desactivado"))
    )
    assert len(fetcher.fetch()) == 1
