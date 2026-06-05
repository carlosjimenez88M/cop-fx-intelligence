"""News article classification: topic tagging and FX-impact severity scoring."""

from __future__ import annotations

import json
from cop_fx.logger import get_logger
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from cop_fx.llm import get_chat_model
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


class NewsAnalyzer:
    """Uses Claude to classify articles and estimate their FX impact."""

    def __init__(self) -> None:
        # Clasificar por noticia = alto volumen → tier barato ('fast')
        self._llm = get_chat_model("fast", temperature=0.0)

    def analyze_batch(self, articles: list[Article]) -> list[AnalyzedArticle]:
        """Classify a batch of articles in a single LLM call."""
        if not articles:
            return []

        digest = "\n".join(
            f"[{i}] title={a.title!r} summary={a.summary[:300]!r}"
            for i, a in enumerate(articles)
        )

        prompt = f"""You are a Colombian FX market analyst.

Classify each article. Return a JSON array of objects with keys:
  index       (integer, same as input)
  topic       (one of: {", ".join(TOPIC_LABELS)})
  severity    (one of: {", ".join(SEVERITY_LABELS)} — impact on COP/USD rate)
  bullish_cop (boolean — true if likely to strengthen COP vs USD)
  reasoning   (≤ 20 words explaining your choice)

Return ONLY the JSON array, no extra text.

ARTICLES:
{digest}
"""

        try:
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            parsed: list[dict[str, Any]] = json.loads(raw.strip())
        except Exception as exc:  # noqa: BLE001
            logger.warning("NewsAnalyzer LLM failed: %s", exc)
            return self._fallback_classify(articles)

        result: list[AnalyzedArticle] = []
        classified = {item["index"]: item for item in parsed}

        for i, article in enumerate(articles):
            item = classified.get(i, {})
            result.append(
                AnalyzedArticle(
                    article=article,
                    topic=item.get("topic", "other"),
                    severity=item.get("severity", "low"),
                    bullish_cop=bool(item.get("bullish_cop", False)),
                    reasoning=item.get("reasoning", ""),
                )
            )
        return result

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
                )
            )
        return result
