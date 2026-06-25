"""Cliente HTTP compartido — un solo lugar para headers, timeouts y reintentos.

Antes, cada fetcher (`cnn_fetcher`, `article_body`, `news_fetcher`) construía
su propio ``httpx.Client`` con un User-Agent distinto y sin política de
reintentos. Eso significaba:

  - Tres User-Agents inconsistentes (uno de ellos delataba al bot → 403).
  - Ningún reintento ante errores transitorios (5xx, timeouts, cortes de red),
    el caso más común al scrapear medios.

Este módulo centraliza esa responsabilidad: un User-Agent de navegador real,
reintentos con backoff exponencial sobre fallos *transitorios* (no sobre un
404, que es permanente) vía ``tenacity``, y dos helpers —``fetch_text`` y
``fetch_bytes``— que devuelven ``None`` en vez de propagar la excepción, de
modo que un medio caído nunca tumba la corrida.
"""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from cop_fx.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30.0

# User-Agent de navegador real: varios medios responden 403 a clientes que se
# identifican como bots/scripts. Aceptamos español de Colombia primero.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
}


def _is_transient(exc: BaseException) -> bool:
    """¿Vale la pena reintentar este error?

    Sí: timeouts, cortes de conexión y 5xx (problemas del servidor o de red).
    No: un 4xx (404, 403) es determinista — reintentarlo solo gasta tiempo.
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


@retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4.0),
    reraise=True,
)
def _get_with_retry(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
) -> httpx.Response:
    """GET con reintentos sobre fallos transitorios. ``raise_for_status`` deja
    que :func:`_is_transient` decida si reintentar (5xx) o abortar (4xx)."""
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp


def fetch_text(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str | None:
    """Descarga ``url`` y devuelve el cuerpo como texto, o ``None`` si falla.

    Nunca propaga la excepción: registra el motivo y devuelve ``None`` para que
    el llamador decida el fallback. Reintenta errores transitorios.
    """
    resp = _safe_get(url, headers=headers or BROWSER_HEADERS, timeout=timeout)
    if resp is None:
        return None
    logger.success("HTTP %d · %d bytes · %s", resp.status_code, len(resp.text), url)  # type: ignore[attr-defined]
    return resp.text


def fetch_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> bytes | None:
    """Igual que :func:`fetch_text` pero devuelve los bytes crudos."""
    resp = _safe_get(url, headers=headers or BROWSER_HEADERS, timeout=timeout)
    return resp.content if resp is not None else None


def _safe_get(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response | None:
    try:
        return _get_with_retry(url, headers=headers, timeout=timeout)
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %d en %s", exc.response.status_code, url)
    except httpx.TimeoutException:
        logger.warning("Timeout en %s", url)
    except httpx.HTTPError as exc:
        logger.warning("Error HTTP en %s: %s", url, exc)
    return None
