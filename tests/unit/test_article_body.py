"""Unit tests for full-article body extraction."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from cop_fx.data import article_body
from cop_fx.data.news_fetcher import Article

_HTML = """
<html><head><script>var x=1;</script></head><body>
<nav><p>Menú de navegación con muchos enlaces y texto basura del sitio</p></nav>
<article>
  <p>El Banco de la República decidió mantener la tasa de interés en su reunión
  de junio, citando presiones inflacionarias persistentes en alimentos.</p>
  <p>Foto: archivo</p>
  <p>Analistas consultados esperaban un recorte de 25 puntos básicos, por lo que
  la decisión sorprendió al mercado cambiario y fortaleció al peso.</p>
  <figure>
    <figcaption>El gerente del banco durante la rueda de prensa con periodistas</figcaption>
  </figure>
</article>
<footer><p>Términos y condiciones del sitio web con texto legal extenso aquí</p></footer>
</body></html>
"""


@pytest.mark.unit()
def test_extract_body_keeps_article_paragraphs_only() -> None:
    body = article_body.extract_body_from_html(_HTML)
    assert "Banco de la República" in body
    assert "sorprendió al mercado" in body
    assert "Foto: archivo" not in body  # migaja corta descartada
    assert "navegación" not in body  # nav eliminado
    assert "Términos y condiciones" not in body  # footer eliminado
    assert "rueda de prensa" not in body  # figcaption eliminado


@pytest.mark.unit()
def test_extract_body_respects_max_chars() -> None:
    body = article_body.extract_body_from_html(_HTML, max_chars=50)
    assert len(body) <= 50


@pytest.mark.unit()
def test_attach_bodies_mutates_articles_and_skips_existing() -> None:
    def art(title: str, body: str = "") -> Article:
        a = Article(
            title=title,
            summary="",
            url=f"https://x.com/{title}",
            published_at=datetime.now(tz=UTC),
            source="Test",
        )
        a.body = body
        return a

    articles = [art("a"), art("b", body="ya lo tengo"), art("c")]
    with patch.object(article_body, "fetch_article_body", return_value="cuerpo nuevo") as mock:
        fetched = article_body.attach_bodies(articles)

    assert fetched == 2
    assert mock.call_count == 2  # el que ya tenía body no se re-descarga
    assert articles[0].body == "cuerpo nuevo"
    assert articles[1].body == "ya lo tengo"


@pytest.mark.unit()
def test_attach_bodies_tolerates_fetch_failure() -> None:
    a = Article(
        title="x",
        summary="resumen original",
        url="https://x.com/x",
        published_at=datetime.now(tz=UTC),
        source="Test",
    )
    with patch.object(article_body, "fetch_article_body", return_value=""):
        fetched = article_body.attach_bodies([a])

    assert fetched == 0
    assert a.body == ""  # el analyzer caerá al summary / "(solo titular)"
