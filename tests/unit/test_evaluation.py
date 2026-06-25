"""Unit tests de la evaluación direccional unificada."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cop_fx.tracking.evaluation import (
    EvalConfig,
    lagged_selection,
    majority_baseline,
    score_strategies,
    to_direction,
)


@pytest.mark.unit()
def test_to_direction_respects_band() -> None:
    cfg = EvalConfig(neutral_band_pct=0.1)
    assert to_direction(0.5, cfg) == "up"
    assert to_direction(-0.5, cfg) == "down"
    assert to_direction(0.05, cfg) == "neutral"  # dentro de la banda


@pytest.mark.unit()
def test_majority_baseline_ignores_abstentions() -> None:
    actuals = ["up", "up", "down", "neutral"]
    direction, rate = majority_baseline(actuals)
    assert direction == "up"
    assert rate == pytest.approx(2 / 3)  # 2 up de 3 decididos


@pytest.mark.unit()
def test_score_strategies_flags_majority_collapse() -> None:
    cfg = EvalConfig(n_boot=200)
    # 70% de los reales son "down"; una estrategia que SIEMPRE dice down colapsa.
    actuals = ["down"] * 7 + ["up"] * 3
    preds = {
        "always_down": ["down"] * 10,
        "perfect": list(actuals),
    }
    scored = score_strategies(preds, actuals, cfg=cfg)
    assert scored.attrs["base_rate"] == pytest.approx(0.7)

    by = scored.set_index("estrategia")
    assert by.loc["always_down", "hit_rate"] == pytest.approx(0.7)
    assert by.loc["always_down", "colapso?"] == "sí"  # accuracy == baseline
    assert by.loc["perfect", "hit_rate"] == pytest.approx(1.0)
    assert by.loc["perfect", "colapso?"] == ""


@pytest.mark.unit()
def test_score_strategies_excludes_neutral_from_hit_rate() -> None:
    cfg = EvalConfig(n_boot=100)
    actuals = ["up", "down", "up", "down"]
    preds = {"abstainer": ["up", "neutral", "up", "neutral"]}  # solo se moja 2 veces, ambas bien
    scored = score_strategies(preds, actuals, cfg=cfg).set_index("estrategia")
    assert scored.loc["abstainer", "n_decididos"] == 2
    assert scored.loc["abstainer", "abstenciones"] == 2
    assert scored.loc["abstainer", "hit_rate"] == pytest.approx(1.0)


@pytest.mark.unit()
def test_lagged_selection_uses_lag_not_contemporaneous() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    # cop[t] depende de x[t-1] (lead-lag real), no de x[t]
    cop = np.r_[0.0, x[:-1]] + rng.normal(0, 0.1, n)
    noise = rng.normal(size=n)
    rets = pd.DataFrame({"cop": cop, "x_lead": x, "ruido": noise})

    sel_lag1 = lagged_selection(rets, target_col="cop", lag=1, band=0.2)
    assert "x_lead" in sel_lag1  # su rezago 1 SÍ correlaciona
    assert "ruido" not in sel_lag1

    # Con look-ahead controlado: upto recorta la muestra (no mira el futuro)
    sel_partial = lagged_selection(rets, target_col="cop", lag=1, band=0.2, upto=100)
    assert isinstance(sel_partial, list)


@pytest.mark.unit()
def test_lagged_selection_band_filters() -> None:
    rng = np.random.default_rng(1)
    rets = pd.DataFrame({"cop": rng.normal(size=200), "ruido": rng.normal(size=200)})
    # banda altísima ⇒ nada pasa
    assert lagged_selection(rets, target_col="cop", lag=1, band=0.99) == []


@pytest.mark.unit()
def test_render_helpers_generate_text_from_live_data() -> None:
    from cop_fx.tracking import render

    cfg = EvalConfig(n_boot=100)
    actuals = ["down"] * 7 + ["up"] * 3
    scored = score_strategies({"always_down": ["down"] * 10}, actuals, cfg=cfg)
    verdict = render.render_eval_verdict(scored, cfg, label="Test")
    assert "70%" in verdict  # base rate generada en vivo
    assert "colapso" in verdict.lower()  # delata el colapso a clase mayoritaria

    corr = pd.Series({"cop": 1.0, "equity": -0.68, "ruido": 0.01})
    txt = render.render_correlation_verdict(corr, band=0.12)
    assert "equity" in txt  # detecta la serie significativa
    assert "ruido" not in txt  # filtra la no significativa

    note = render.render_run_note("2026-06-24", n_days=1090)
    assert "2026-06-24" in note and "a mano" in note
