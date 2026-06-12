"""News article classification: topic tagging and FX-impact severity scoring.

Etapa 1 (docs/plan_maestro.md): el LLM responde contra el schema
`BatchAnalysis` vía `with_structured_output` — sin parsing manual de JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cop_fx.contracts import BatchAnalysis
from cop_fx.llm import get_chat_model
from cop_fx.logger import get_logger

if TYPE_CHECKING:
    from cop_fx.data.news_fetcher import Article

logger = get_logger(__name__)


@dataclass
class AnalyzedArticle:
    article: Article
    topic: str
    severity: str
    bullish_cop: bool
    reasoning: str
    keywords: list[str]
    entities: list[str]
    fx_relevance: str
    fx_channel: str


@dataclass
class NewsAnalysis:
    """Resultado del lote: veredictos por artículo + narrativa del día."""

    items: list[AnalyzedArticle]
    narrative: str


_PROMPT_TEMPLATE = """You are a skeptical Colombian FX transmission analyst.

Your job is not to summarize news. Your job is to decide whether the FULL
article contains a plausible, near-term transmission channel into USD/COP.
Be conservative: most articles are noise, repeated information, or too local
to move the exchange rate. A strong answer names the mechanism, the surprise
relative to normal market expectations, and the sign for COP.

All natural-language fields must be in Spanish only. Do not use English or
other languages except proper nouns, tickers and institution names.
The target is Colombia's peso against the US dollar. Do not classify generic
global macro as material unless it has one of these explicit bridges:
Colombia/COP, Fed/US macro/USD leg, oil or other terms of trade, Colombia/EM
capital flows, or country-risk repricing.

Each item below contains the article TITLE and its full TEXT (when a body
could not be retrieved you will see "(solo titular disponible)" — be more
conservative with severity in that case).

Classify each article based on its FULL TEXT. The taxonomy has TWO levels —
what the news IS (`topic`) and HOW it transmits to the USD/COP rate
(`fx_channel`):
  - index: the 0-based position of the article in the list below
  - topic: the article's domain (sports and culture stories exist in the
    taxonomy — do NOT force them into economic categories)
  - keywords: 3-5 key terms in Spanish
  - entities: up to 5 named entities (people, institutions, companies, places)
  - fx_relevance: direct | indirect | none. Think second-order channels:
    a drought is indirect via food inflation; an epidemic via growth and
    fiscal risk; a football match is none. Foreign macro OUTSIDE the US
    (UK GDP, ECB, Asia) is at most INDIRECT via the dollar index or global
    risk appetite — low/medium severity, or none if purely domestic to
    that country.
  - fx_channel: the transmission mechanism (interest_rates, inflation,
    terms_of_trade, country_risk, capital_flows, growth, none)
  - severity: expected magnitude of the FX impact (high | medium | low)
  - bullish_cop: true if the news is likely to strengthen COP vs USD
  - reasoning: <= 20 words naming the channel explicitly

Calibration rules:
  - high = fresh, specific, credible shock with direct FX channel and likely
    repricing within 1-7 business days (BanRep surprise, fiscal rule breach,
    oil shock, ratings action, Fed surprise, large country-risk event).
  - medium = relevant but partly expected, second-order, or mixed evidence.
  - low = routine, stale, narrow sector story, or weak channel.
  - none = no actionable USD/COP mechanism. Do not force an economic label.
  - If the article is not about Colombia and is not about the USD leg, oil,
    terms of trade, country risk or capital flows, set fx_relevance=none even
    if it is economically interesting.
  - If only a title/short summary is available, severity cannot be high unless
    the headline itself describes a major policy, fiscal, oil, Fed, or risk shock.
  - Do not double-count routine follow-ups as fresh shocks. Penalize recycled
    stories unless the article adds a new fact that changes the FX view.
  - Penalize long-dated policy announcements (for example taxes in 2027) unless
    the article explains why markets should reprice USD/COP during the current
    1-7 business day horizon.

Also write `market_narrative`: 3 sentences on the day's FX outlook,
based ONLY on the articles with fx_relevance != none. It must state the
dominant channels, the main conflict or uncertainty, and whether the news
signal is strong enough to challenge a quantitative time-series signal.
Write the narrative in Spanish only.

ARTICLES:
{digest}
"""


class NewsAnalyzer:
    """Classifies articles and estimates their FX impact via structured output."""

    def __init__(self) -> None:
        # Clasificar por noticia = alto volumen → tier barato ('fast')
        self._llm = get_chat_model("fast", temperature=0.0).with_structured_output(BatchAnalysis)

    def analyze(self, articles: list[Article]) -> NewsAnalysis:
        """Classify a batch of articles in a single structured LLM call."""
        if not articles:
            return NewsAnalysis(items=[], narrative="No news available.")

        # El veredicto se hace sobre la NOTICIA, no el titular: body completo
        # cuando está disponible (attach_bodies), summary como respaldo.
        def _text(a: Article) -> str:
            full = getattr(a, "body", "") or a.summary
            return full[:1800] if full else "(solo titular disponible)"

        digest = "\n\n".join(
            f"[{i}] TITLE: {a.title}\nTEXT: {_text(a)}" for i, a in enumerate(articles)
        )

        try:
            raw = self._llm.invoke(_PROMPT_TEMPLATE.format(digest=digest))
            batch = raw if isinstance(raw, BatchAnalysis) else BatchAnalysis.model_validate(raw)
        except Exception as exc:
            logger.warning("NewsAnalyzer LLM failed: %s", exc)
            return NewsAnalysis(
                items=self._fallback_classify(articles),
                narrative="News analysis unavailable (fallback heuristic).",
            )

        by_index = {item.index: item for item in batch.items}
        items: list[AnalyzedArticle] = []
        for i, article in enumerate(articles):
            item = by_index.get(i)
            if item is None:
                items.append(self._fallback_classify([article])[0])
                continue
            items.append(
                AnalyzedArticle(
                    article=article,
                    topic=item.topic,
                    severity=item.severity,
                    bullish_cop=item.bullish_cop,
                    reasoning=item.reasoning,
                    keywords=item.keywords,
                    entities=item.entities,
                    fx_relevance=item.fx_relevance,
                    fx_channel=item.fx_channel,
                )
            )
        return NewsAnalysis(items=items, narrative=batch.market_narrative)

    def analyze_batch(self, articles: list[Article]) -> list[AnalyzedArticle]:
        """Backward-compatible wrapper: only the per-article verdicts."""
        return self.analyze(articles).items

    def _fallback_classify(self, articles: list[Article]) -> list[AnalyzedArticle]:
        """Keyword-based fallback when the LLM is unavailable."""
        high_kw = {"petróleo", "oil", "banrep", "tasas", "interés", "inflación", "dólar"}
        result: list[AnalyzedArticle] = []
        for article in articles:
            text = (article.title + " " + article.summary).lower()
            hits = sorted(kw for kw in high_kw if kw in text)
            severity = "high" if hits else "low"
            result.append(
                AnalyzedArticle(
                    article=article,
                    topic="other",
                    severity=severity,
                    bullish_cop=False,
                    reasoning="fallback: keyword heuristic",
                    keywords=hits[:5] or ["sin_clasificar"],
                    entities=[],
                    fx_relevance="indirect" if hits else "none",
                    fx_channel="terms_of_trade" if hits else "none",
                )
            )
        return result
