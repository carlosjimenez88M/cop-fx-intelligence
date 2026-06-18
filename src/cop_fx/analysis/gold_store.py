"""Persistencia de la capa GOLD a SQLite — separada de la lógica del grafo.

`aggregate_signals` produce los veredictos GOLD (artículos clasificados +
ranking de keywords); ESCRIBIRLOS a disco es una responsabilidad distinta de
DECIDIR la señal. Este módulo aísla esa frontera: el dashboard y los notebooks
leen de `data/cnn_articles.db`, que se reescribe entero en cada corrida con la
foto del día.

Cero LLM salvo el juez de keywords opcional (`keyword_llm_judge_enabled`),
que también degrada con gracia a la heurística determinista.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cop_fx.analysis.keyword_analysis import (
    build_term_documents,
    judge_terms,
    pairwise_phi,
    rank_terms,
)
from cop_fx.analysis.topic_taxonomy import article_importance
from cop_fx.config.settings import get_settings
from cop_fx.logger import get_logger
from cop_fx.paths import DATA_DIR

logger = get_logger(__name__)

GOLD_DB_PATH = DATA_DIR / "cnn_articles.db"


def persist_articles(articles: list[dict[str, Any]]) -> None:
    """Reescribe la tabla `articles` con la clasificación GOLD de la corrida."""
    if not articles:
        return
    GOLD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    rows = []
    for article in articles:
        published = article.get("published_at")
        published_at = (
            published.isoformat() if isinstance(published, datetime) else str(published or "")
        )
        rows.append(
            (
                article.get("title", ""),
                article.get("author", ""),
                published_at,
                article.get("summary", ""),
                article.get("url", ""),
                article.get("source", ""),
                article.get("topic", "other"),
                json.dumps(article.get("keywords") or [], ensure_ascii=False),
                now,
                json.dumps(article.get("entities") or [], ensure_ascii=False),
                article.get("fx_relevance", "none"),
                article.get("fx_channel", "none"),
                article.get("severity", "low"),
                int(bool(article.get("bullish_cop", False))),
                article.get("reasoning", ""),
            )
        )

    with sqlite3.connect(GOLD_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                author TEXT,
                published_at TEXT,
                summary TEXT,
                url TEXT,
                source TEXT,
                topic TEXT,
                keywords TEXT,
                fetched_at TEXT,
                entities TEXT,
                fx_relevance TEXT,
                fx_channel TEXT,
                severity TEXT,
                bullish_cop INTEGER,
                reasoning TEXT
            )
            """
        )
        conn.execute("DELETE FROM articles")
        conn.executemany(
            """
            INSERT INTO articles (
                title, author, published_at, summary, url, source, topic,
                keywords, fetched_at, entities, fx_relevance, fx_channel,
                severity, bullish_cop, reasoning
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    logger.info("Persisted %d GOLD articles to %s", len(rows), GOLD_DB_PATH)


def _entity_rollup(
    articles: pd.DataFrame,
    *,
    domain_stopwords: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stopwords = {word.casefold() for word in domain_stopwords}
    for _, row in articles.iterrows():
        for entity in row.get("entities_list") or []:
            if str(entity).casefold() in stopwords:
                continue
            rows.append(
                {
                    "term": str(entity),
                    "score": float(row.get("importance") or 0.0),
                    "article_count": 1,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["term", "score", "article_count"])
    return (
        pd.DataFrame(rows)
        .groupby("term", as_index=False)
        .agg(score=("score", "sum"), article_count=("article_count", "sum"))
        .sort_values(["score", "article_count"], ascending=False)
        .head(top_n)
    )


def persist_keyword_insights(articles: list[dict[str, Any]]) -> None:
    """Reescribe los rankings de keywords/entidades y las co-ocurrencias.

    Flujo: keywords GOLD → quita stopwords de dominio → el juez (LLM o
    heurística) decide accionabilidad → ranking ponderado por importancia.
    `keyword_pairs` queda como artefacto exploratorio multi-día.
    """
    if not articles:
        return

    settings = get_settings()
    now = datetime.now(UTC).isoformat()
    material = pd.DataFrame([item for item in articles if item.get("fx_relevance") != "none"])
    GOLD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(GOLD_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS keyword_terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT,
                term_key TEXT,
                score REAL,
                article_count INTEGER,
                llm_score REAL,
                reason TEXT,
                computed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS keyword_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item1 TEXT,
                item2 TEXT,
                correlation REAL,
                n_both INTEGER,
                computed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS keyword_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT,
                score REAL,
                article_count INTEGER,
                computed_at TEXT
            )
            """
        )
        conn.execute("DELETE FROM keyword_terms")
        conn.execute("DELETE FROM keyword_pairs")
        conn.execute("DELETE FROM keyword_entities")

        if material.empty:
            return

        material["keywords_list"] = material["keywords"].apply(lambda value: list(value or []))
        material["entities_list"] = material["entities"].apply(lambda value: list(value or []))
        material["importance"] = material.apply(article_importance, axis=1)
        term_docs = build_term_documents(
            material,
            domain_stopwords=settings.keyword_domain_stopwords,
            include_entities=False,
        )
        if term_docs.empty:
            entities = _entity_rollup(
                material,
                domain_stopwords=settings.keyword_domain_stopwords,
            )
            conn.executemany(
                """
                INSERT INTO keyword_entities (term, score, article_count, computed_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        str(row["term"]),
                        float(str(row["score"])),
                        int(float(str(row["article_count"]))),
                        now,
                    )
                    for row in entities.to_dict("records")
                ],
            )
            return

        candidate_terms = (
            term_docs.groupby(["term_key", "term"], as_index=False)
            .agg(score=("importance", "sum"), noticias=("doc_id", "nunique"))
            .sort_values(["score", "noticias"], ascending=False)
            .head(max(settings.keyword_top_n * 2, 30))["term"]
            .tolist()
        )
        decisions = judge_terms(
            candidate_terms,
            use_llm=settings.keyword_llm_judge_enabled,
            domain_stopwords=settings.keyword_domain_stopwords,
        )
        terms = rank_terms(
            term_docs,
            decisions,
            min_articles=settings.keyword_min_articles,
            top_n=settings.keyword_top_n,
        )
        pairs = pairwise_phi(
            term_docs,
            terms["termino"].tolist() if not terms.empty else [],
            min_joint=settings.keyword_pairwise_min_joint,
        )
        if not pairs.empty:
            pairs = pairs[pairs["correlation"] >= settings.keyword_pairwise_min_correlation].head(
                settings.keyword_pairwise_top_n
            )
        entities = _entity_rollup(
            material,
            domain_stopwords=settings.keyword_domain_stopwords,
        )

        conn.executemany(
            """
            INSERT INTO keyword_terms (
                term, term_key, score, article_count, llm_score, reason, computed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["termino"]),
                    str(row["term_key"]),
                    float(str(row["score"])),
                    int(float(str(row["noticias"]))),
                    float(str(row["llm_score"])),
                    str(row["reason"]),
                    now,
                )
                for row in terms.to_dict("records")
            ],
        )
        conn.executemany(
            """
            INSERT INTO keyword_pairs (item1, item2, correlation, n_both, computed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["item1"]),
                    str(row["item2"]),
                    float(str(row["correlation"])),
                    int(float(str(row["n_both"]))),
                    now,
                )
                for row in pairs.to_dict("records")
            ],
        )
        conn.executemany(
            """
            INSERT INTO keyword_entities (term, score, article_count, computed_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    str(row["term"]),
                    float(str(row["score"])),
                    int(float(str(row["article_count"]))),
                    now,
                )
                for row in entities.to_dict("records")
            ],
        )
    logger.info("Persisted GOLD keyword insights to %s", GOLD_DB_PATH)
