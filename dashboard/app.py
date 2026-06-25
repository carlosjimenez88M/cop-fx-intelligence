"""Dashboard COP/USD — minimalista, centrado en el objetivo del sistema.

El sistema responde UNA pregunta: ¿hacia dónde va el USD/COP mañana —baja,
sube o conviene abstenerse— y con cuánta confianza? Este dashboard muestra
exactamente eso y nada más:

    1. El call vigente (dirección, confianza, racional) y las tres señales
       que lo sustentan (noticias · serie · mercado).
    2. La noticia del día que más pesa.
    3. El forecast Prophet+ARIMA que ancla el signo de la serie.
    4. El track record: ¿le está acertando el sistema?

La exploración profunda (keywords, taxonomía, huecos de investigación) vivía
aquí antes y saturaba la lectura; ahora vive en la API (`/api/v1/news`) y en
las notebooks. El desk se queda con la decisión.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from cop_fx.data.fx_fetcher import FXFetcher
from cop_fx.paths import DATA_DIR, REPORTS_DIR
from cop_fx.timeseries.models import ARIMAForecaster, ProphetForecaster, ensemble_forecast
from cop_fx.tracking.predictions import PredictionStore

FX_CSV = DATA_DIR / "cop_usd.csv"
PREDICTIONS_DB = DATA_DIR / "predictions.db"

DIRECTION_LABEL = {"down": "USD/COP baja", "up": "USD/COP sube", "neutral": "Abstenerse"}
DIRECTION_COLOR = {"down": "#0f7b52", "up": "#b42318", "neutral": "#52606d"}

st.set_page_config(page_title="COP/USD Intelligence Desk", layout="centered")


# ---------------------------------------------------------------------------
# Carga de datos (cacheada)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_fx() -> pd.DataFrame:
    if FX_CSV.exists():
        return pd.read_csv(FX_CSV, parse_dates=["ds"]).sort_values("ds")
    try:
        return FXFetcher().fetch()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return PredictionStore(PREDICTIONS_DB).all()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    return PredictionStore(PREDICTIONS_DB).metrics()


@st.cache_data(show_spinner=True)
def load_forecast(fx: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Ensemble Prophet+ARIMA; vacío si no hay serie."""
    if fx.empty:
        return pd.DataFrame()
    prophet = ProphetForecaster().fit_predict(fx, horizon_days=horizon_days)
    arima = ARIMAForecaster().fit_predict(fx, horizon_days=horizon_days)
    return ensemble_forecast([prophet, arima])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest(preds: pd.DataFrame) -> pd.Series | None:
    if preds.empty:
        return None
    return preds.sort_values("run_date", ascending=False).iloc[0]


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.2f}%"


def _brief(text: Any, limit: int = 400) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rsplit(" ", 1)[0] + "…"


def _run_pipeline() -> None:
    with st.spinner("Corriendo el pipeline completo…"):
        result = subprocess.run(
            ["uv", "run", "python", "-m", "cop_fx.cli", "run"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode == 0:
        st.cache_data.clear()
        st.success("Pipeline completado. Refresca para ver el nuevo call.")
    else:
        st.error("El pipeline falló.")
        st.code((result.stderr or result.stdout)[-3000:])


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------


def render_call(pred: pd.Series | None) -> None:
    if pred is None:
        st.info("Aún no hay un call. Corre el pipeline para generar el primero.")
        return

    direction = str(pred.get("direction", "neutral"))
    confidence = float(pred.get("confidence", 0) or 0)
    color = DIRECTION_COLOR.get(direction, "#52606d")

    label = DIRECTION_LABEL.get(direction, direction)
    st.markdown(
        f"<h1 style='margin-bottom:0;color:{color}'>{label}</h1>"
        f"<p style='color:#5f6b7a;margin-top:.25rem'>"
        f"Horizonte {pred.get('horizon_days', '—')} días · confianza {confidence:.0%} · "
        f"reconciliación {pred.get('reconciliation', '—')} · "
        f"señal dominante {pred.get('dominant_signal', 'none')}</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cols[0].metric("📰 Noticias", pred.get("news_direction", "—"), pred.get("news_score"))
    cols[1].metric("📈 Serie", pred.get("ts_direction", "—"), _pct(pred.get("ts_delta_pct")))
    cols[2].metric("🌎 Mercado", pred.get("market_direction", "—"))

    rationale = str(pred.get("rationale") or "Sin racional guardado.")
    st.write(_brief(rationale, 460))
    with st.expander("Racional completo y contra-argumento"):
        st.markdown("**Racional**")
        st.write(rationale)
        st.markdown("**Abogado del diablo**")
        st.write(str(pred.get("devils_advocate") or "—"))


def render_top_story(pred: pd.Series | None) -> None:
    if pred is None:
        return
    title = pred.get("top_story_display_title") or pred.get("top_story_title")
    if not title:
        return
    st.subheader("Noticia del día")
    st.markdown(f"**{title}**  ·  _{pred.get('top_story_source') or 'fuente n/d'}_")
    st.write(_brief(pred.get("top_story_why"), 360))


def render_forecast(pred: pd.Series | None, fx: pd.DataFrame) -> None:
    if fx.empty:
        return
    st.subheader("Forecast del USD/COP")
    horizon = int(pred.get("horizon_days") or 7) if pred is not None else 7
    forecast = load_forecast(fx, horizon)
    if forecast.empty:
        st.caption("No se pudo calcular el forecast.")
        return
    history = fx.tail(90)[["ds", "y"]].rename(columns={"y": "TRM real"})
    line = forecast[["ds", "yhat"]].rename(columns={"yhat": "Ensemble"})
    chart = history.merge(line, on="ds", how="outer").sort_values("ds").set_index("ds")
    st.line_chart(chart, height=280)
    st.caption(
        "El signo de la serie sale de comparar el `yhat` final del ensemble con la última "
        "TRM observada; dentro de la banda muerta, la serie se abstiene."
    )


def render_track_record(preds: pd.DataFrame, metrics: dict[str, Any]) -> None:
    st.subheader("Track record")
    cols = st.columns(4)
    cols[0].metric("Predicciones", metrics.get("n_total", 0))
    cols[1].metric("Evaluadas", metrics.get("n_evaluated", 0))
    cols[2].metric("Decididas", metrics.get("n_decided", 0))
    hit = metrics.get("hit_rate")
    cols[3].metric("Hit-rate", "—" if hit is None else f"{hit:.0%}")

    if preds.empty:
        st.caption("Sin historial aún.")
        return
    show = ["run_date", "direction", "confidence", "actual_direction", "final_hit"]
    existing = [c for c in show if c in preds.columns]
    st.dataframe(
        preds.sort_values("run_date", ascending=False)[existing].head(12),
        width="stretch",
        hide_index=True,
    )


def render_report_expander() -> None:
    reports = sorted(REPORTS_DIR.glob("report_*.md"), reverse=True)
    if not reports:
        return
    with st.expander("Reporte diario en Markdown"):
        selected = st.selectbox("Reporte", reports, format_func=lambda p: p.name)
        st.markdown(selected.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    with st.sidebar:
        st.title("COP/USD Desk")
        st.caption("La decisión diaria del USD/COP, con su evidencia.")
        if st.button("Refrescar datos", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        if st.button("Correr pipeline", width="stretch"):
            _run_pipeline()
        st.caption(f"UI: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")

    fx = load_fx()
    preds = load_predictions()
    metrics = load_metrics()
    pred = _latest(preds)

    render_call(pred)
    render_top_story(pred)
    st.divider()
    render_forecast(pred, fx)
    st.divider()
    render_track_record(preds, metrics)
    render_report_expander()


if __name__ == "__main__":
    main()
