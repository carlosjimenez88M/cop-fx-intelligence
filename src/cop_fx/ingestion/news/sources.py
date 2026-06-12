"""Registro de fuentes de noticias.

DOS bloques a propósito:
  - CO     → noticias locales (riesgo político/fiscal, BanRep, TRM)
  - GLOBAL → el lado USD (Fed, dólar, mercados) — sin esto te pierdes
             la mitad de la historia del USD/COP.

Las URLs de RSS cambian con el tiempo. NO confíes en esta lista a ciegas:
corre `python -m cop_fx.ingestion.news validate` para ver cuáles están vivas
hoy. Las que fallen, simplemente se ignoran en runtime (el extractor es
tolerante a feeds caídos).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Country = Literal["CO", "GLOBAL"]


@dataclass(frozen=True)
class NewsSource:
    name: str
    url: str
    country: Country
    kind: Literal["rss"] = "rss"


# --- Colombia (local) -------------------------------------------------------
COLOMBIA_SOURCES: list[NewsSource] = [
    NewsSource("CNN en Español — Economía", "https://cnnespanol.cnn.com/economia/feed/", "CO"),
    NewsSource("Portafolio — Economía", "https://www.portafolio.co/rss/economia.xml", "CO"),
    NewsSource("El Tiempo — Economía", "https://www.eltiempo.com/rss/economia.xml", "CO"),
    NewsSource("La República", "https://www.larepublica.co/rss/", "CO"),
]

# --- Global / lado USD ------------------------------------------------------
GLOBAL_SOURCES: list[NewsSource] = [
    NewsSource("Investing.com — Forex", "https://www.investing.com/rss/news_1.rss", "GLOBAL"),
    NewsSource("Federal Reserve — Press releases", "https://www.federalreserve.gov/feeds/press_all.xml", "GLOBAL"),
]

SOURCES: list[NewsSource] = COLOMBIA_SOURCES + GLOBAL_SOURCES
