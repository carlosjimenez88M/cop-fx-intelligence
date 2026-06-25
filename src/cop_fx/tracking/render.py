"""Narrativa de conclusiones GENERADA desde la corrida en vivo.

Las conclusiones del notebook 02 estaban escritas como prosa con números fijos
de una corrida vieja, y al re-ejecutar con datos frescos contradecían sus
propios outputs. Estas funciones construyen el texto/tabla a partir de las
variables reales de la corrida actual: ninguna cifra se escribe a mano.

Las afirmaciones cualitativas ("equity es la señal más fuerte", "el orden por
AIC sobreajusta") se derivan con un umbral explícito sobre los datos, no se
asumen. Devuelven Markdown para mostrarse con ``IPython.display.Markdown``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from cop_fx.tracking.evaluation import EvalConfig


def render_run_note(run_date: str, *, n_days: int | None = None) -> str:
    """Nota de cierre: cifras en vivo, sin números a mano."""
    span = f" · {n_days} días alineados" if n_days else ""
    return (
        f"> _Cifras generadas en vivo el **{run_date}**{span}; "
        "ningún número está escrito a mano — todo sale de los DataFrames de esta corrida._"
    )


def render_arima_verdict(
    table: pd.DataFrame,
    *,
    bic_best: tuple[int, int, int],
    aic_best: tuple[int, int, int],
    residuals_white_noise: bool,
) -> str:
    """Veredicto del orden ARIMA, data-driven sobre AIC vs BIC de ESTA corrida."""
    aic_row = table.sort_values("aic").iloc[0]
    bic_row = table.sort_values("bic").iloc[0]
    agree = aic_best == bic_best
    if agree:
        head = (
            f"**AIC y BIC coinciden en `{aic_best}`** "
            f"(AIC {aic_row['aic']}, BIC {bic_row['bic']}). "
            "Sin disputa de orden en esta corrida: no hay señal de sobreajuste "
            "entre criterios."
        )
    else:
        head = (
            f"**AIC elige `{aic_best}` (AIC {aic_row['aic']}) pero BIC elige "
            f"`{bic_best}` (BIC {bic_row['bic']})**. El BIC castiga más la "
            "complejidad: cuando difieren, el orden más parco suele generalizar "
            "mejor out-of-sample (el AIC tiende a premiar parámetros que ajustan "
            "ruido)."
        )
    resid = (
        "Los residuales del orden elegido pasan Ljung-Box (ruido blanco)."
        if residuals_white_noise
        else "⚠️ Los residuales NO son ruido blanco: queda estructura sin modelar."
    )
    return f"### Veredicto de orden (vivo)\n\n{head}\n\n{resid}"


def render_correlation_verdict(
    corr_weekly: pd.Series,
    *,
    band: float,
    target: str = "cop",
) -> str:
    """Lee qué series superan la banda y cuál es la más fuerte, con su signo."""
    s = corr_weekly.drop(labels=[target], errors="ignore").dropna()
    significant = s[s.abs() > band].sort_values(key=lambda x: x.abs(), ascending=False)
    if significant.empty:
        return f"Ninguna serie supera la banda ±{band:.3f}: sin estructura semanal clara."
    strongest = significant.index[0]
    val = significant.iloc[0]
    sign_txt = "negativa" if val < 0 else "positiva"
    lines = [
        f"- **{len(significant)} de {len(s)} series** superan la banda ±{band:.3f} "
        "(correlación semanal).",
        f"- La más fuerte es **{strongest}** ({val:+.3f}, {sign_txt}).",
        f"- Significativas: {', '.join(significant.index)}.",
    ]
    return "### Lectura de correlaciones (vivo)\n\n" + "\n".join(lines)


def render_eval_verdict(scored: pd.DataFrame, cfg: EvalConfig, *, label: str) -> str:
    """Resume una tabla de :func:`score_strategies` con su baseline y significancia.

    Marca explícitamente colapsos a la clase mayoritaria y si el mejor modelo
    es estadísticamente distinguible de una moneda.
    """
    base_rate = scored.attrs.get("base_rate", float("nan"))
    base_dir = scored.attrs.get("base_dir", "?")
    best = scored.iloc[0]
    collapsed = scored[scored["colapso?"] == "sí"]["estrategia"].tolist()
    sig = "sí" if best["p_vs_50%"] == best["p_vs_50%"] and best["p_vs_50%"] < 0.05 else "no"

    lines = [
        f"**{label}** · regla única: horizonte {cfg.horizon_days}d, "
        f"banda muerta ±{cfg.neutral_band_pct:.2f}%, {best['n_decididos']} decididos.",
        f"- Baseline trivial (clase mayoritaria = `{base_dir}`): **{base_rate:.0%}**. "
        "Ninguna estrategia es 'buena' si no le gana a esto.",
        f"- Mejor estrategia: **{best['estrategia']}** = {best['hit_rate']:.0%} "
        f"(IC95 {best['ci95']}); ¿distinguible de una moneda? **{sig}** "
        f"(p={best['p_vs_50%']}).",
    ]
    if collapsed:
        lines.append(
            f"- ⚠️ Colapso a la clase mayoritaria (accuracy ≈ baseline, skill ~0): "
            f"{', '.join(collapsed)}."
        )
    if best["vs_base"] == best["vs_base"] and best["vs_base"] <= 0:
        lines.append(
            "- 🚩 Ni la mejor estrategia supera al baseline: con esta ventana, "
            "la dirección a este horizonte es prácticamente impredecible."
        )
    return "### Veredicto del backtest (vivo)\n\n" + "\n".join(lines)
