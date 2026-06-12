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


_PROMPT_TEMPLATE = """You are a Colombian FX market analyst.

Classify each article. The taxonomy has TWO levels — what the news IS
(`topic`) and HOW it transmits to the USD/COP rate (`fx_channel`):
  - index: the 0-based position of the article in the list below
  - topic: the article's domain (sports and culture stories exist in the
    taxonomy — do NOT force them into economic categories)
  - keywords: 3-5 key terms in Spanish
  - entities: up to 5 named entities (people, institutions, companies, places)
  - fx_relevance: direct | indirect | none. Think second-order channels:
    a drought is indirect via food inflation; an epidemic via growth and
    fiscal risk; a football match is none.
  - fx_channel: the transmission mechanism (interest_rates, inflation,
    terms_of_trade, country_risk, capital_flows, growth, none)
  - severity: expected magnitude of the FX impact (high | medium | low)
  - bullish_cop: true if the news is likely to strengthen COP vs USD
  - reasoning: <= 20 words naming the channel explicitly

Also write `market_narrative`: 3 sentences on the day's FX outlook,
based ONLY on the articles with fx_relevance != none.

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

        digest = "\n".join(
            f"[{i}] title={a.title!r} summary={a.summary[:300]!r}"
            for i, a in enumerate(articles)
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
