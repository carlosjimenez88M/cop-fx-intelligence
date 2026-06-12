"""Extracción del CUERPO COMPLETO de un artículo — los veredictos se hacen
sobre la noticia, no sobre el titular.

Hasta hoy los agentes clasificaban con título + summary, y el summary venía
VACÍO en la mayoría de las fuentes (el listado HTML de CNN no lo trae). Eso
significaba veredictos sobre titulares: un error de fondo. Este módulo lo
corrige con un extractor genérico estilo *readability*:

  1. Busca el tag `<article>`; si no existe, el contenedor con más texto en
     párrafos (heurística robusta entre medios distintos).
  2. Une los `<p>` con texto real, descartando scripts/asides/figcaptions.
  3. Capa el resultado (control de tokens) — el LLM no necesita 20.000
     caracteres para juzgar dirección e impacto.

El COSTO se controla por diseño: el gate de materialidad sigue decidiendo
sobre titulares (barato y suficiente para rutear); el cuerpo completo se
descarga SOLO para los artículos materiales que van a los workers.
"""

from __future__ import annotations

from typing import Protocol

import httpx
from bs4 import BeautifulSoup

from cop_fx.logger import get_logger

logger = get_logger(__name__)

MAX_BODY_CHARS = 3500
# Menos que esto NO es un artículo: es boilerplate de paywall/cookies.
# Devolver '' obliga al analyzer a caer al summary RSS (más honesto).
MIN_BODY_CHARS = 400
_HEADERS = {"User-Agent": "Mozilla/5.0 (cop-fx-intelligence)"}
_SKIP_PARENTS = {"script", "style", "aside", "figure", "figcaption", "footer", "nav", "form"}


class _HasBody(Protocol):
    url: str


def extract_body_from_html(html: str, *, max_chars: int = MAX_BODY_CHARS) -> str:
    """Extrae el texto del artículo desde el HTML — determinista, sin red."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(list(_SKIP_PARENTS)):
        tag.decompose()

    container = soup.find("article")
    if container is None:
        # El contenedor (div/section/main) con más texto en <p> directos o anidados
        best, best_len = None, 0
        for cand in soup.find_all(["main", "section", "div"]):
            length = sum(len(p.get_text(strip=True)) for p in cand.find_all("p", recursive=False))
            if length > best_len:
                best, best_len = cand, length
        container = best or soup

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in container.find_all("p")
        if len(p.get_text(strip=True)) > 40  # descarta migajas (créditos, fechas)
    ]
    return "\n".join(paragraphs)[:max_chars]


def fetch_article_body(url: str, *, max_chars: int = MAX_BODY_CHARS, timeout: float = 15.0) -> str:
    """Descarga la página del artículo y extrae su cuerpo. '' si falla."""
    try:
        with httpx.Client(timeout=timeout, headers=_HEADERS, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
        body = extract_body_from_html(resp.text, max_chars=max_chars)
        if len(body) < MIN_BODY_CHARS:
            logger.warning(
                "Cuerpo demasiado corto (%d chars) en %s — probable paywall/boilerplate",
                len(body), url,
            )
            return ""
        return body
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo extraer el cuerpo de %s: %s", url, exc)
        return ""


def attach_bodies(
    articles: list,
    *,
    max_articles: int = 30,
    max_chars: int = MAX_BODY_CHARS,
) -> int:
    """Adjunta `body` (texto completo) a cada artículo que no lo tenga.

    Muta los objetos in-place (`setattr`) para servir tanto a `Article`
    como a `CNNArticle`. Devuelve cuántos cuerpos se descargaron.
    """
    fetched = 0
    for article in articles[:max_articles]:
        if getattr(article, "body", ""):
            continue
        body = fetch_article_body(article.url, max_chars=max_chars)
        article.body = body
        if body:
            fetched += 1
    if fetched:
        logger.info("Cuerpos descargados: %d/%d artículos", fetched, min(len(articles), max_articles))
    return fetched
