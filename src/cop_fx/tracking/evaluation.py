"""Evaluación direccional unificada — una sola regla para TODO el estudio.

Por qué existe
--------------
La notebook 02 tenía dos estándares de evaluación conviviendo: Parte A usaba
banda muerta (±0.1%) con abstenciones y la Extensión usaba signo crudo, así que
"momentum" daba 54% en una sección y 68% en otra y NO eran comparables. Este
módulo fija una sola configuración (:class:`EvalConfig`) y un solo marcador, de
modo que cualquier estrategia se mide igual en todas las secciones.

Además aporta lo que faltaba para no auto-engañarse:
  - ``base_rate``: la tasa de la clase mayoritaria de la ventana (el baseline
    trivial que delata cuándo un modelo "gana" solo porque colapsó a una clase).
  - ``ci95``: intervalo de confianza bootstrap sobre los orígenes — con ~60
    orígenes un 57% y un 68% pueden ser indistinguibles, y hay que mostrarlo.
  - ``p_binom``: test binomial del hit-rate contra 50% (¿mejor que una moneda?).
  - :func:`lagged_selection`: selección de features REZAGADA (no contemporánea)
    y por-fold (sin look-ahead), que reemplaza el filtro roto del notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Direction = str  # "up" | "down" | "neutral"


@dataclass(frozen=True)
class EvalConfig:
    """Regla única de evaluación direccional (fijada al inicio del estudio).

    Attributes:
        horizon_days: a cuántos días se evalúa la dirección.
        neutral_band_pct: banda muerta en %; |Δ%| por debajo ⇒ abstención
            (``neutral``). 0.0 = signo crudo (sin abstenciones).
        n_origins: número de orígenes out-of-sample (cola de la serie).
        min_train: mínimo de observaciones antes del primer origen.
        n_boot: remuestras bootstrap para el IC95.
        seed: semilla del bootstrap (reproducibilidad).
    """

    horizon_days: int = 5
    neutral_band_pct: float = 0.0
    n_origins: int = 120
    min_train: int = 252
    n_boot: int = 2000
    seed: int = 42


def to_direction(delta_pct: float, cfg: EvalConfig) -> Direction:
    """Mapea un Δ% (previsto o real) a dirección bajo la banda muerta de ``cfg``."""
    if delta_pct > cfg.neutral_band_pct:
        return "up"
    if delta_pct < -cfg.neutral_band_pct:
        return "down"
    return "neutral"


def majority_baseline(actuals: Sequence[Direction]) -> tuple[Direction, float]:
    """Clase mayoritaria entre los reales DECIDIDOS y su frecuencia (la tasa base)."""
    decided = [a for a in actuals if a != "neutral"]
    if not decided:
        return "neutral", 0.0
    ups = sum(a == "up" for a in decided)
    frac_up = ups / len(decided)
    return ("up", frac_up) if frac_up >= 0.5 else ("down", 1 - frac_up)


def _bootstrap_ci(hits: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    if len(hits) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(hits, size=(n_boot, len(hits)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def _binom_p(n_hits: int, n: int) -> float:
    """p-valor de dos colas del test binomial vs 50% (sin SciPy: exacto)."""
    if n == 0:
        return float("nan")
    from math import comb

    def tail(k: int) -> float:
        return sum(comb(n, i) for i in range(k + 1)) / (2.0**n)

    # dos colas alrededor de la mediana n/2
    k = min(n_hits, n - n_hits)
    return round(min(1.0, 2 * tail(k)), 4)


def score_strategies(
    preds_by_strategy: Mapping[str, Sequence[Direction]],
    actuals: Sequence[Direction],
    *,
    cfg: EvalConfig,
) -> pd.DataFrame:
    """Marca cada estrategia con la MISMA regla y reporta su incertidumbre.

    Las abstenciones (``neutral``) de la predicción se excluyen del hit-rate
    (solo se puntúan los días en que la estrategia se mojó). Devuelve un
    DataFrame con: hit_rate, n_decididos, abstenciones, base_rate, IC95 y
    p binomial vs 50%.
    """
    base_dir, base_rate = majority_baseline(actuals)
    rows: list[dict[str, object]] = []
    for name, preds in preds_by_strategy.items():
        hits = np.array(
            [int(p == a) for p, a in zip(preds, actuals, strict=False) if p != "neutral"]
        )
        n = len(hits)
        hit_rate = float(hits.mean()) if n else float("nan")
        lo, hi = _bootstrap_ci(hits, n_boot=cfg.n_boot, seed=cfg.seed)
        rows.append(
            {
                "estrategia": name,
                "hit_rate": round(hit_rate, 3),
                "ci95": f"[{lo:.2f}, {hi:.2f}]" if n else "—",
                "p_vs_50%": _binom_p(int(hits.sum()), n),
                "n_decididos": n,
                "abstenciones": sum(p == "neutral" for p in preds),
                "vs_base": round(hit_rate - base_rate, 3) if n else float("nan"),
                "colapso?": "sí" if n and abs(hit_rate - base_rate) < 0.01 else "",
            }
        )
    out = pd.DataFrame(rows).sort_values("hit_rate", ascending=False).reset_index(drop=True)
    out.attrs["base_dir"] = base_dir
    out.attrs["base_rate"] = round(base_rate, 3)
    return out


def lagged_selection(
    returns: pd.DataFrame,
    *,
    target_col: str = "cop",
    lag: int = 1,
    band: float,
    upto: int | None = None,
) -> list[str]:
    """Series cuyo retorno REZAGADO correlaciona con el target — sin look-ahead.

    Corrige el bug del notebook (que usaba correlación CONTEMPORÁNEA sobre toda
    la muestra): aquí se aplica ``shift(lag)`` (el pasado de la serie contra el
    presente del target) y, si se pasa ``upto``, solo se usan filas ``< upto``
    (los datos disponibles en ese origen del rolling).

    Args:
        returns: DataFrame de retornos con ``target_col`` entre sus columnas.
        target_col: la serie objetivo (el COP).
        lag: rezago en pasos (días) de las exógenas.
        band: umbral de |correlación| para entrar a la selección.
        upto: índice exclusivo; solo se miran filas anteriores (evita fuga).
    """
    sub = returns.iloc[:upto] if upto is not None else returns
    selected: list[str] = []
    for col in returns.columns:
        if col == target_col:
            continue
        corr = sub[col].shift(lag).corr(sub[target_col])
        if pd.notna(corr) and abs(float(corr)) > band:
            selected.append(col)
    return selected
