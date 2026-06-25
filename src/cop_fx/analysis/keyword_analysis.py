"""Keyword ranking and pairwise co-occurrence for GOLD news."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from pydantic import BaseModel, Field

from cop_fx.config.settings import get_settings
from cop_fx.llm import get_chat_model

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\- ]+")

_BASE_STOPWORDS = {
    "a",
    "al",
    "ante",
    "con",
    "de",
    "del",
    "el",
    "en",
    "entre",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "se",
    "sin",
    "sobre",
    "su",
    "sus",
    "un",
    "una",
    "y",
}


class KeywordDecision(BaseModel):
    """LLM decision about one candidate term."""

    term: str
    important: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=160)


class KeywordJudgement(BaseModel):
    """Structured output for a batch of keyword decisions."""

    items: list[KeywordDecision]


_KEYWORD_JUDGE_PROMPT = """You are a Colombian FX research editor.
Decide whether each candidate term is analytically useful for USD/COP news
analysis before a word co-occurrence graph is built.

Keep terms that name a tradable driver, event, mechanism, policy, shock,
institutional risk, macro release, commodity, inflation/tax/rate channel or
specific political/electoral event.

Reject generic context words, country names, broad labels, source names, and
terms that are only frequent because the product is about Colombia or FX.

Return one decision per term. Use Spanish in reasons.

Think step by step and take a feedback loop into your conclusions.:
"""


def normalize_term(term: str) -> str:
    """Normalize term for grouping while preserving Spanish letters via accent stripping."""
    stripped = unicodedata.normalize("NFKD", term.strip().lower())
    ascii_term = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    compact = re.sub(r"\s+", " ", ascii_term)
    return compact.strip(" -_.,;:()[]{}\"'")


def is_domain_stopword(term: str, domain_stopwords: Iterable[str] | None = None) -> bool:
    """Return True for terms that are too generic for this product."""
    normalized = normalize_term(term)
    stopwords = {normalize_term(word) for word in (domain_stopwords or [])}
    return normalized in _BASE_STOPWORDS or normalized in stopwords


def clean_candidate_terms(
    terms: Iterable[str],
    *,
    domain_stopwords: Iterable[str] | None = None,
) -> list[str]:
    """Clean, deduplicate and remove generic terms."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in terms:
        match = _TOKEN_RE.search(str(raw))
        if not match:
            continue
        term = " ".join(match.group(0).split())
        key = normalize_term(term)
        if len(key) < 3 or key.isnumeric() or is_domain_stopword(key, domain_stopwords):
            continue
        if key not in seen:
            seen.add(key)
            cleaned.append(term)
    return cleaned


def heuristic_judge_terms(
    terms: Sequence[str],
    *,
    domain_stopwords: Iterable[str] | None = None,
) -> list[KeywordDecision]:
    """Deterministic fallback when LLM judging is disabled or unavailable."""
    decisions: list[KeywordDecision] = []
    for term in terms:
        normalized = normalize_term(term)
        important = not is_domain_stopword(normalized, domain_stopwords)
        token_count = len(normalized.split())
        score = 0.75 if token_count > 1 else 0.55
        decisions.append(
            KeywordDecision(
                term=term,
                important=important,
                score=score if important else 0.0,
                reason="Termino especifico" if important else "Termino generico del dominio",
            )
        )
    return decisions


def judge_terms_with_llm(terms: Sequence[str]) -> list[KeywordDecision]:
    """Use the configured LLM to decide which terms are analytically useful."""
    if not terms:
        return []
    payload = "\n".join(f"- {term}" for term in terms)
    llm = get_chat_model("fast").with_structured_output(KeywordJudgement)
    result = cast(
        "KeywordJudgement",
        llm.invoke(f"{_KEYWORD_JUDGE_PROMPT}\n\nCandidate terms:\n{payload}"),
    )
    return list(result.items)


def build_term_documents(
    articles: pd.DataFrame,
    *,
    domain_stopwords: Iterable[str] | None = None,
    include_entities: bool = False,
) -> pd.DataFrame:
    """Return tidy document-term rows from GOLD article keywords."""
    rows: list[dict[str, Any]] = []
    if articles.empty:
        return pd.DataFrame(columns=["doc_id", "term", "term_key", "importance"])

    for doc_id, (_, row) in enumerate(articles.reset_index(drop=True).iterrows()):
        terms = list(row.get("keywords_list") or [])
        if include_entities:
            terms.extend(list(row.get("entities_list") or []))
        for term in clean_candidate_terms(terms, domain_stopwords=domain_stopwords):
            rows.append(
                {
                    "doc_id": int(doc_id),
                    "term": term,
                    "term_key": normalize_term(term),
                    "importance": float(row.get("importance") or 0.0),
                }
            )
    return pd.DataFrame(rows)


def rank_terms(
    term_docs: pd.DataFrame,
    decisions: Sequence[KeywordDecision],
    *,
    min_articles: int = 1,
    top_n: int = 25,
) -> pd.DataFrame:
    """Rank judged terms by weighted GOLD importance."""
    if term_docs.empty:
        return pd.DataFrame(columns=["termino", "score", "noticias", "llm_score", "reason"])

    decision_by_key = {normalize_term(item.term): item for item in decisions}

    def _decision_for(key: object) -> KeywordDecision:
        key_str = str(key)
        return decision_by_key.get(
            key_str,
            KeywordDecision(term=key_str, important=True, score=1.0, reason=""),
        )

    judged = term_docs.copy()
    judged["llm_score"] = judged["term_key"].map(lambda key: _decision_for(key).score)
    judged["important"] = judged["term_key"].map(lambda key: _decision_for(key).important)
    judged = judged[judged["important"]]
    if judged.empty:
        return pd.DataFrame(columns=["termino", "score", "noticias", "llm_score", "reason"])

    ranked = (
        judged.groupby(["term_key", "term"], as_index=False)
        .agg(
            score=("importance", "sum"),
            noticias=("doc_id", "nunique"),
            llm_score=("llm_score", "max"),
        )
        .query("noticias >= @min_articles")
    )
    if ranked.empty:
        return pd.DataFrame(columns=["termino", "score", "noticias", "llm_score", "reason"])
    ranked["score"] = (ranked["score"] * ranked["llm_score"]).round(3)
    ranked["reason"] = ranked["term_key"].map(lambda key: _decision_for(key).reason)
    return (
        ranked.rename(columns={"term": "termino"})[
            ["termino", "score", "noticias", "llm_score", "reason", "term_key"]
        ]
        .sort_values(["score", "noticias"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def pairwise_phi(
    term_docs: pd.DataFrame,
    keep_terms: Sequence[str],
    *,
    min_joint: int = 1,
) -> pd.DataFrame:
    """Compute pairwise phi correlation of terms across article documents."""
    if term_docs.empty or len(keep_terms) < 2:
        return pd.DataFrame(columns=["item1", "item2", "correlation", "n_both"])

    keep = {normalize_term(term) for term in keep_terms}
    docs = term_docs[term_docs["term_key"].isin(keep)][["doc_id", "term_key"]].drop_duplicates()
    if docs.empty:
        return pd.DataFrame(columns=["item1", "item2", "correlation", "n_both"])

    matrix = pd.crosstab(docs["doc_id"], docs["term_key"]).astype(bool)
    n_docs = len(matrix)
    rows: list[dict[str, Any]] = []
    terms = list(matrix.columns)
    for i, item1 in enumerate(terms):
        x = matrix[item1]
        n1_dot = int(x.sum())
        n0_dot = n_docs - n1_dot
        for item2 in terms[i + 1 :]:
            y = matrix[item2]
            n_dot1 = int(y.sum())
            n_dot0 = n_docs - n_dot1
            n11 = int((x & y).sum())
            n10 = n1_dot - n11
            n01 = n_dot1 - n11
            numerator = n11 * n_dot0 - n10 * n01
            denominator = math.sqrt(n1_dot * n0_dot * n_dot1 * n_dot0)
            if denominator == 0:
                continue
            corr = numerator / denominator
            if corr > 0 and n11 >= min_joint:
                rows.append(
                    {
                        "item1": item1,
                        "item2": item2,
                        "correlation": round(corr, 3),
                        "n_both": n11,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["item1", "item2", "correlation", "n_both"])
    return pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)


def judge_terms(
    terms: Sequence[str],
    *,
    use_llm: bool | None = None,
    domain_stopwords: Iterable[str] | None = None,
) -> list[KeywordDecision]:
    """Judge terms with LLM when enabled; otherwise use deterministic fallback."""
    settings = get_settings()
    enabled = settings.keyword_llm_judge_enabled if use_llm is None else use_llm
    if enabled:
        try:
            return judge_terms_with_llm(terms)
        except Exception:
            pass
    return heuristic_judge_terms(terms, domain_stopwords=domain_stopwords)
