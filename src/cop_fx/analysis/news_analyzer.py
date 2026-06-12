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

TOPIC_LABELS = [
    "monetary_policy",  # BanRep interest rate decisions, inflation
    "trade",            # imports / exports / trade balance
    "political_risk",   # elections, social unrest, regulation
    "commodities",      # oil, coffee, coal prices
    "macro",            # GDP, employment, fiscal deficit
    "other",
]

SEVERITY_LABELS = ["high", "medium", "low"]


@dataclass
class AnalyzedArticle:
    article: Article
    topic: str
    severity: str
    bullish_cop: bool
    reasoning: str
    keywords: list[str]


@dataclass
class NewsAnalysis:
    """Resultado del lote: veredictos por artículo + narrativa del día."""

    items: list[AnalyzedArticle]
    narrative: str


_PROMPT_TEMPLATE = """You are a Colombian FX market analyst.

Classify each article by its expected impact on the USD/COP exchange rate:
  - topic: the article's economic category
  - keywords: 3-5 key terms in Spanish
  - severity: expected impact on the COP/USD rate (high | medium | low)
  - bullish_cop: true if the news is likely to strengthen COP vs USD
  - reasoning: <= 20 words justifying your choice
  - index: the 0-based position of the article in the list below

Also write `market_narrative`: 3 sentences on the day's FX outlook,
based ONLY on these articles.

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
            severity = "high" if any(kw in text for kw in high_kw) else "low"
            result.append(
                AnalyzedArticle(
                    article=article,
                    topic="other",
                    severity=severity,
                    bullish_cop=False,
                    reasoning="fallback: keyword heuristic",
                    keywords=sorted(kw for kw in high_kw if kw in text)[:5],
                )
            )
        return result
