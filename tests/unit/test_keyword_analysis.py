from __future__ import annotations

import pandas as pd
import pytest

from cop_fx.analysis.keyword_analysis import (
    build_term_documents,
    clean_candidate_terms,
    heuristic_judge_terms,
    pairwise_phi,
    rank_terms,
)


@pytest.mark.unit()
def test_clean_candidate_terms_removes_domain_stopwords() -> None:
    terms = clean_candidate_terms(
        ["Colombia", "segunda vuelta", "inflación", "de"], domain_stopwords=["colombia"]
    )
    assert "Colombia" not in terms
    assert "segunda vuelta" in terms
    assert "inflación" in terms


@pytest.mark.unit()
def test_rank_terms_uses_judgement_score() -> None:
    docs = pd.DataFrame(
        [
            {"doc_id": 0, "term": "inflación", "term_key": "inflacion", "importance": 1.0},
            {"doc_id": 1, "term": "inflación", "term_key": "inflacion", "importance": 0.5},
            {"doc_id": 1, "term": "ruido", "term_key": "ruido", "importance": 1.0},
        ]
    )
    decisions = heuristic_judge_terms(["inflación", "ruido"])
    ranked = rank_terms(docs, decisions, min_articles=1, top_n=10)
    assert ranked.iloc[0]["termino"] == "inflación"


@pytest.mark.unit()
def test_build_term_documents_uses_keywords_not_generic_entities() -> None:
    articles = pd.DataFrame(
        [
            {
                "keywords_list": ["Colombia", "segunda vuelta"],
                "entities_list": ["Colombia"],
                "importance": 0.7,
            }
        ]
    )
    docs = build_term_documents(articles, domain_stopwords=["colombia"], include_entities=False)
    assert docs["term"].tolist() == ["segunda vuelta"]


@pytest.mark.unit()
def test_pairwise_phi_finds_repeated_cooccurrence() -> None:
    docs = pd.DataFrame(
        [
            {"doc_id": 0, "term_key": "segunda vuelta"},
            {"doc_id": 0, "term_key": "riesgo politico"},
            {"doc_id": 1, "term_key": "segunda vuelta"},
            {"doc_id": 1, "term_key": "riesgo politico"},
            {"doc_id": 2, "term_key": "inflacion"},
        ]
    )
    pairs = pairwise_phi(docs, ["segunda vuelta", "riesgo politico", "inflacion"])
    top = pairs.iloc[0]
    assert {top["item1"], top["item2"]} == {"segunda vuelta", "riesgo politico"}
    assert top["correlation"] > 0
