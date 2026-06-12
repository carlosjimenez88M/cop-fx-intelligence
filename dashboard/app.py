"""Streamlit dashboard for the COP/USD intelligence product."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from cop_fx.analysis.topic_taxonomy import (
    article_importance,
    channel_label,
    directional_bias,
    normalize_topic,
    research_gap,
)
from cop_fx.config.settings import get_settings
from cop_fx.data.fx_fetcher import FXFetcher
from cop_fx.data.market_fetcher import fetch_market_panel
from cop_fx.paths import DATA_DIR, REPORTS_DIR
from cop_fx.timeseries.models import ARIMAForecaster, ProphetForecaster, ensemble_forecast
from cop_fx.tracking.predictions import PredictionStore

FX_CSV = DATA_DIR / "cop_usd.csv"
CNN_DB = DATA_DIR / "cnn_articles.db"
PREDICTIONS_DB = DATA_DIR / "predictions.db"
SETTINGS = get_settings()
NEWS_LIMIT = SETTINGS.news_max_articles

DIRECTION_LABELS = {
    "down": "USD/COP baja",
    "up": "USD/COP sube",
    "neutral": "Abstenerse",
}


st.set_page_config(page_title="COP/USD Intelligence Desk", layout="wide")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            border: 1px solid #d7dde5;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #ffffff;
            color: #101828;
        }
        div[data-testid="stMetric"] * { color: #101828 !important; }
        div[data-testid="stMetric"] label {
            color: #52606d;
            font-size: 0.82rem;
        }
        .decision-strip {
            border-left: 5px solid #2f6fed;
            padding: 0.9rem 1rem;
            background: #f7f9fc;
            border-radius: 8px;
            margin: 0.3rem 0 1rem 0;
            color: #101828;
        }
        .decision-strip h2 { color: #101828; }
        .top-story {
            border: 1px solid #d7dde5;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
            color: #101828;
        }
        .top-story h4, .top-story p { color: #101828; }
        .small-note {
            color: #5f6b7a;
            font-size: 0.9rem;
        }
        .section-label {
            color: #3f4b5a;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_json(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        loaded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    return []


@st.cache_data(show_spinner=False)
def load_fx() -> pd.DataFrame:
    if FX_CSV.exists():
        df = pd.read_csv(FX_CSV, parse_dates=["ds"])
        return df.sort_values("ds")
    try:
        return FXFetcher().fetch()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_articles() -> pd.DataFrame:
    if not CNN_DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(CNN_DB) as conn:
        df = pd.read_sql(
            "SELECT * FROM articles ORDER BY published_at DESC LIMIT ?",
            conn,
            params=(NEWS_LIMIT,),
        )
    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["topic_family"] = df["topic"].apply(normalize_topic)
    df["channel_label"] = df["fx_channel"].apply(channel_label)
    df["keywords_list"] = df["keywords"].apply(_safe_json)
    df["entities_list"] = df["entities"].apply(_safe_json)
    df["importance"] = df.apply(article_importance, axis=1)
    df["directional_bias"] = df.apply(directional_bias, axis=1)
    df["research_gap"] = df.apply(research_gap, axis=1)
    return df.sort_values(["importance", "published_at"], ascending=[False, False])


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return PredictionStore(PREDICTIONS_DB).all()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


@st.cache_data(show_spinner=False)
def load_news_insights() -> dict[str, pd.DataFrame | str]:
    """Load pipeline-computed keyword and co-occurrence insights."""
    empty = {
        "terms": pd.DataFrame(),
        "pairs": pd.DataFrame(),
        "entities": pd.DataFrame(),
        "computed_at": "",
    }
    if not CNN_DB.exists():
        return empty
    with sqlite3.connect(CNN_DB) as conn:
        required = ("keyword_terms", "keyword_entities")
        if not all(_table_exists(conn, table) for table in required):
            return empty
        terms = pd.read_sql(
            """
            SELECT
                term AS termino,
                term_key,
                score,
                article_count AS noticias,
                llm_score,
                reason,
                computed_at
            FROM keyword_terms
            ORDER BY score DESC, noticias DESC
            """,
            conn,
        )
        entities = pd.read_sql(
            """
            SELECT term AS termino, score, article_count AS noticias, computed_at
            FROM keyword_entities
            ORDER BY score DESC, noticias DESC
            """,
            conn,
        )
    computed_values = [
        str(value)
        for frame in (terms, entities)
        if not frame.empty
        for value in frame["computed_at"].dropna().head(1).tolist()
    ]
    return {
        "terms": terms.drop(columns=["computed_at"], errors="ignore"),
        "entities": entities.drop(columns=["computed_at"], errors="ignore"),
        "computed_at": computed_values[0] if computed_values else "",
    }


@st.cache_data(show_spinner=False)
def load_forecast_view(fx: pd.DataFrame, horizon_days: int) -> dict[str, Any]:
    if fx.empty:
        return {}
    prophet_result = ProphetForecaster().fit_predict(fx, horizon_days=horizon_days)
    arima_result = ARIMAForecaster().fit_predict(fx, horizon_days=horizon_days)
    ensemble = ensemble_forecast([prophet_result, arima_result])
    latest = float(fx["y"].iloc[-1])

    rows = []
    for name, result in (("Prophet", prophet_result), ("ARIMA", arima_result)):
        yhat = float(result.forecast["yhat"].iloc[-1])
        rows.append(
            {
                "modelo": name,
                "yhat_final": yhat,
                "delta_pct": round((yhat - latest) / latest * 100, 3),
            }
        )
    yhat_ensemble = float(ensemble["yhat"].iloc[-1])
    rows.append(
        {
            "modelo": "Ensemble",
            "yhat_final": yhat_ensemble,
            "delta_pct": round((yhat_ensemble - latest) / latest * 100, 3),
        }
    )
    return {
        "latest": latest,
        "ensemble": ensemble,
        "models": pd.DataFrame(rows),
    }


@st.cache_data(show_spinner=False)
def load_market_trends(fx: pd.DataFrame) -> pd.DataFrame:
    if fx.empty:
        return pd.DataFrame()
    try:
        panel = fetch_market_panel(fx.tail(260), lookback_days=365)
    except Exception:
        return pd.DataFrame()
    if panel.empty:
        return panel
    numeric_cols = [col for col in panel.columns if col != "ds"]
    base = panel.copy()
    for col in numeric_cols:
        first = base[col].dropna().iloc[0]
        base[col] = 100 * base[col] / first
    return base


def _latest_prediction(preds: pd.DataFrame) -> pd.Series | None:
    if preds.empty:
        return None
    return preds.sort_values("run_date", ascending=False).iloc[0]


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}%"


def _brief(text: Any, limit: int = 560) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rsplit(" ", 1)[0] + "..."


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#0f7b52"
    if score >= 0.45:
        return "#9a6200"
    return "#52606d"


def _topic_rollup(articles: pd.DataFrame) -> pd.DataFrame:
    if articles.empty:
        return pd.DataFrame()
    material = articles[articles["fx_relevance"] != "none"].copy()
    if material.empty:
        return pd.DataFrame()
    material["signed_score"] = material.apply(
        lambda row: row["importance"] if row["bullish_cop"] else -row["importance"], axis=1
    )
    rollup = (
        material.groupby(["topic_family", "topic", "channel_label"], dropna=False)
        .agg(
            noticias=("id", "count"),
            importancia_media=("importance", "mean"),
            score_neto=("signed_score", "sum"),
        )
        .reset_index()
        .sort_values(["score_neto", "importancia_media"], key=lambda s: s.abs(), ascending=False)
    )
    rollup["importancia_media"] = rollup["importancia_media"].round(2)
    rollup["score_neto"] = rollup["score_neto"].round(2)
    return rollup


def _render_forecast_context(pred: pd.Series | None, fx: pd.DataFrame) -> None:
    if pred is None or fx.empty:
        return
    horizon = int(pred.get("horizon_days") or 7)
    try:
        forecast = load_forecast_view(fx, horizon)
    except Exception as exc:
        st.warning(f"No se pudo recalcular el forecast para graficar: {exc}")
        return
    if not forecast:
        return

    models = forecast["models"]
    ensemble = forecast["ensemble"].copy()
    ensemble["serie"] = "forecast"
    history = fx.tail(80)[["ds", "y"]].rename(columns={"y": "TRM real"})
    forecast_line = ensemble[["ds", "yhat"]].rename(columns={"yhat": "Ensemble forecast"})
    chart = history.merge(forecast_line, on="ds", how="outer").sort_values("ds").set_index("ds")

    st.subheader("Forecast y lectura de signo")
    st.caption(
        "Down no significa una opinion bajista abstracta: significa que el yhat final "
        "del ensemble queda por debajo de la ultima TRM observada. Si el delta es "
        "menor que la banda muerta, la serie se abstiene."
    )
    cols = st.columns(3)
    for idx, row in models.iterrows():
        cols[idx].metric(
            row["modelo"],
            f"{row['yhat_final']:,.2f}",
            _format_pct(row["delta_pct"]),
        )
    st.line_chart(chart, height=280)

    trends = load_market_trends(fx)
    if not trends.empty and len(trends.columns) > 2:
        st.subheader("Tendencias macro normalizadas")
        st.caption(
            "Base 100 para comparar direcciones: COP/TRM, Brent, DXY y equity Colombia. "
            "En el estudio historico, equity fue la unica senal adelantada robusta."
        )
        st.line_chart(trends.set_index("ds").tail(180), height=280)


def _render_decision(pred: pd.Series | None, fx: pd.DataFrame, articles: pd.DataFrame) -> None:
    if pred is None:
        st.info("Aun no hay predicciones guardadas. Corre el pipeline para generar el primer call.")
        return

    direction = str(pred.get("direction", "neutral"))
    confidence = float(pred.get("confidence", 0) or 0)
    color = "#0f7b52" if direction == "down" else "#b42318" if direction == "up" else "#52606d"
    st.markdown(
        f"""
        <div class="decision-strip" style="border-left-color:{color}">
            <div class="section-label">Call vigente</div>
            <h2 style="margin:0.15rem 0 0.1rem 0;">{DIRECTION_LABELS.get(direction, direction)}</h2>
            <div class="small-note">
                Horizonte {pred.get("horizon_days", "n/a")} dias · confianza {confidence:.2f} ·
                reconciliacion {pred.get("reconciliation", "n/a")} · dominante
                {pred.get("dominant_signal", "none")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Noticia", pred.get("news_direction", "n/a"), pred.get("news_score"))
    metric_cols[1].metric(
        "Serie temporal",
        pred.get("ts_direction", "n/a"),
        _format_pct(pred.get("ts_delta_pct")),
    )
    metric_cols[2].metric("Mercado", pred.get("market_direction", "n/a"))
    metric_cols[3].metric("TRM base", f"{float(pred.get('latest_rate') or 0):,.2f}")

    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        rationale = pred.get("rationale") or "Sin racional guardado."
        devils = pred.get("devils_advocate") or "Sin contra-argumento guardado."
        st.subheader("Lectura ejecutiva")
        st.write(_brief(rationale, 620))
        with st.expander("Ver racional completo y contra-argumento"):
            st.markdown("**Racional**")
            st.write(rationale)
            st.markdown("**Contra-argumento**")
            st.write(devils)

    with right:
        st.subheader("Noticia mas importante")
        original_title = pred.get("top_story_title") or ""
        title = pred.get("top_story_display_title") or original_title or "Sin noticia destacada"
        source = pred.get("top_story_source") or "Fuente no registrada"
        why = pred.get("top_story_why") or "Sin justificacion guardada."
        score: float | None = None
        if not articles.empty and (original_title or title):
            title_norm = str(original_title or title).casefold()
            exact = articles["title"].astype(str).str.casefold() == title_norm
            contains = articles["title"].astype(str).str.casefold().str.contains(
                title_norm[:80], regex=False, na=False
            )
            match = articles[exact | contains]
            if not match.empty:
                score = float(match["importance"].iloc[0])
        score_text = f"{score:.2f}" if score is not None else "no disponible en GOLD"
        score_color = _score_color(score or 0.0) if score is not None else "#52606d"
        original_note = (
            f'<p class="small-note">Original: {original_title}</p>'
            if original_title and original_title != title
            else ""
        )
        st.markdown(
            f"""
            <div class="top-story">
                <div class="section-label">{source}</div>
                <h4 style="margin:0.2rem 0 0.45rem 0;">{title}</h4>
                {original_note}
                <div style="color:{score_color}; font-weight:700;">
                    Importancia estimada: {score_text}
                </div>
                <p style="margin-bottom:0;">{_brief(why, 460)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if len(str(why)) > 460:
            with st.expander("Por que importa - detalle"):
                st.write(why)

    if not fx.empty:
        with st.expander("Forecast Prophet+ARIMA y tendencias macro"):
            st.caption(
                "Este calculo ajusta Prophet y ARIMA en vivo; se ejecuta bajo demanda "
                "para que el desk cargue rapido."
            )
            if st.button("Calcular forecast y tendencias", width="stretch"):
                _render_forecast_context(pred, fx)


def _render_news_lab(articles: pd.DataFrame) -> None:
    if articles.empty:
        st.info("No hay noticias clasificadas todavia.")
        return

    material = articles[articles["fx_relevance"] != "none"]
    cols = st.columns(5)
    cols[0].metric("Entrada maxima", f"{NEWS_LIMIT:,}")
    cols[1].metric("GOLD analizadas", f"{len(articles):,}")
    cols[2].metric("Materiales FX", f"{len(material):,}")
    cols[3].metric("Topics activos", material["topic"].nunique() if not material.empty else 0)
    cols[4].metric(
        "Importancia media",
        f"{material['importance'].mean():.2f}" if not material.empty else "0.00",
    )
    st.caption(
        "El pipeline trae hasta 100 noticias; GOLD contiene las que pasaron materialidad, "
        "texto analizable y clasificacion estructurada."
    )
    if material["topic"].nunique() <= 2 and len(material) >= 10:
        st.info(
            "La senal noticiosa esta muy concentrada en pocos topics. "
            "Esto sugiere revisar fuentes, deduplicacion y subtopics antes de "
            "darle demasiado peso al score agregado."
        )

    news_insights = load_news_insights()
    keywords = news_insights["terms"]
    entities = news_insights["entities"]
    computed_at = str(news_insights["computed_at"])
    if computed_at:
        st.caption(f"Insights GOLD calculados por el pipeline: {computed_at}")
    else:
        st.warning(
            "Aun no hay insights GOLD persistidos. Corre el pipeline para calcular "
            "keywords, entidades y co-ocurrencias antes de abrir el dashboard."
        )

    keyword_left, keyword_right = st.columns([0.58, 0.42], gap="large")
    with keyword_left:
        st.subheader("Palabras importantes juzgadas")
        st.caption(
            "Estos terminos vienen del ultimo pipeline: primero se filtran terminos "
            "genericos del dominio, luego el juez LLM decide si cada termino es "
            "accionable para USD/COP, y solo despues se calculan scores."
        )
        if keywords.empty:
            st.caption("No hay keywords suficientes en la capa GOLD.")
        else:
            st.bar_chart(keywords.set_index("termino")["score"], height=300)
            st.dataframe(
                keywords[["termino", "score", "noticias", "llm_score", "reason"]],
                width="stretch",
                hide_index=True,
                height=260,
                column_config={
                    "termino": "Termino",
                    "score": st.column_config.NumberColumn("Score", format="%.2f"),
                    "noticias": "Noticias",
                    "llm_score": st.column_config.NumberColumn("LLM", format="%.2f"),
                    "reason": "Razon LLM",
                },
            )
    with keyword_right:
        st.subheader("Entidades recurrentes")
        if entities.empty:
            st.caption("No hay entidades suficientes para agregar.")
        else:
            st.bar_chart(entities.set_index("termino")["score"], height=300)

    st.subheader("Ranking de importancia")
    ranking_cols = [
        "title",
        "topic_family",
        "channel_label",
        "directional_bias",
        "importance",
    ]
    st.dataframe(
        articles[ranking_cols].head(15),
        width="stretch",
        hide_index=True,
        column_config={
            "title": st.column_config.TextColumn("Noticia", width="medium"),
            "topic_family": st.column_config.TextColumn("Familia", width="small"),
            "channel_label": st.column_config.TextColumn("Canal FX", width="small"),
            "directional_bias": st.column_config.TextColumn("Sesgo", width="small"),
            "importance": st.column_config.ProgressColumn(
                "Importancia",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        },
        height=420,
    )
    with st.expander("Ver detalle completo de clasificacion"):
        detail_cols = [
            "title",
            "source",
            "topic",
            "fx_channel",
            "fx_relevance",
            "severity",
            "importance",
            "reasoning",
        ]
        st.dataframe(
            articles[detail_cols].head(50),
            width="stretch",
            hide_index=True,
            height=420,
        )

    st.subheader("Topics normalizados")
    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        rollup = _topic_rollup(articles)
        if rollup.empty:
            st.caption("No hay noticias materiales para agregar.")
        else:
            st.dataframe(
                rollup,
                width="stretch",
                hide_index=True,
                column_config={
                    "topic_family": "Familia",
                    "topic": "Topic original",
                    "channel_label": "Canal FX",
                    "noticias": "N",
                    "importancia_media": st.column_config.NumberColumn(
                        "Importancia media", format="%.2f"
                    ),
                    "score_neto": st.column_config.NumberColumn(
                        "Score neto",
                        help="Positivo favorece COP; negativo favorece USD/COP al alza.",
                        format="%.2f",
                    ),
                },
                height=360,
            )
    with right:
        families = material["topic_family"].value_counts().rename_axis("familia").reset_index()
        families.columns = ["familia", "noticias"]
        st.bar_chart(families.set_index("familia"), height=260)

    st.subheader("Huecos de investigacion")
    gaps = articles[articles["research_gap"] != ""][
        ["title", "topic", "fx_channel", "fx_relevance", "research_gap"]
    ].head(20)
    if gaps.empty:
        st.success("No aparecen inconsistencias fuertes en la clasificacion actual.")
    else:
        st.dataframe(gaps, width="stretch", hide_index=True)

    with st.expander("Criterio propuesto para ampliar topics"):
        st.markdown(
            """
            La taxonomia no deberia crecer solo por nombres nuevos. Para mejorar resultados,
            conviene separar cinco capas: dominio (`topic`), evento especifico (`event_type`),
            canal FX, sorpresa frente al consenso y horizonte esperado. Asi una noticia de
            seguridad puede distinguir ataque a infraestructura petrolera, paro regional o
            deterioro institucional, que no tienen el mismo canal ni persistencia.

            Nuevos campos candidatos: `event_type`, `surprise_level`, `time_horizon`,
            `source_quality`, `geo_scope` y `duplicate_cluster_id`. Esto reduce ruido,
            ayuda a deduplicar noticias repetidas y permite que el score de importancia
            premie novedad real, no volumen de titulares.
            """
        )


def _render_learning(preds: pd.DataFrame) -> None:
    store = PredictionStore(PREDICTIONS_DB)
    metrics = store.metrics()
    cols = st.columns(4)
    cols[0].metric("Predicciones", metrics.get("n_total", 0))
    cols[1].metric("Evaluadas", metrics.get("n_evaluated", 0))
    cols[2].metric("Decididas", metrics.get("n_decided", 0))
    hit_rate = metrics.get("hit_rate")
    cols[3].metric("Hit-rate final", "n/a" if hit_rate is None else f"{hit_rate:.0%}")

    if preds.empty:
        st.info("Todavia no hay historial para evaluar.")
        return

    st.subheader("Historial de calls")
    cols_to_show = [
        "run_date",
        "direction",
        "confidence",
        "reconciliation",
        "dominant_signal",
        "news_direction",
        "ts_direction",
        "market_direction",
        "actual_direction",
        "final_hit",
    ]
    existing = [col for col in cols_to_show if col in preds.columns]
    st.dataframe(preds[existing], width="stretch", hide_index=True, height=320)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Accuracy por componente")
        component_rows = []
        for label, key in (
            ("Final", "final_hit_rate"),
            ("Noticias", "news_hit_rate"),
            ("Serie temporal", "ts_hit_rate"),
            ("Mercado", "market_hit_rate"),
        ):
            value = metrics.get(key)
            component_rows.append({"componente": label, "hit_rate": value})
        comp = pd.DataFrame(component_rows).dropna()
        if comp.empty:
            st.caption("Faltan predicciones evaluadas para comparar componentes.")
        else:
            st.bar_chart(comp.set_index("componente"), height=260)

    with right:
        st.subheader("Calibracion")
        bucket = metrics.get("hit_rate_by_confidence")
        if isinstance(bucket, pd.DataFrame) and not bucket.empty:
            st.dataframe(bucket, width="stretch")
        else:
            st.caption("Se necesita mas historial evaluado para calibrar confianza.")


def _render_reports() -> None:
    st.subheader("Reportes generados")
    reports = sorted(REPORTS_DIR.glob("report_*.md"), reverse=True)
    if not reports:
        st.info("No hay reportes en la carpeta reports.")
        return
    selected = st.selectbox("Reporte", reports, format_func=lambda path: path.name)
    st.markdown(selected.read_text(encoding="utf-8"))


def _run_pipeline() -> None:
    with st.spinner("Corriendo pipeline completo..."):
        result = subprocess.run(
            ["uv", "run", "python", "-m", "cop_fx.cli", "run"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode == 0:
        st.cache_data.clear()
        st.success("Pipeline completado. Refresca los datos desde el boton superior.")
        if result.stdout:
            st.code(result.stdout[-2500:])
    else:
        st.error("El pipeline fallo.")
        st.code((result.stderr or result.stdout)[-4000:])


def main() -> None:
    _inject_css()

    with st.sidebar:
        st.title("COP/USD Desk")
        st.caption("Decision diaria, evidencia noticiosa y aprendizaje del modelo.")
        if st.button("Refrescar datos", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        if st.button("Correr pipeline", width="stretch"):
            _run_pipeline()
        st.divider()
        st.caption(f"Ultima carga UI: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")

    fx = load_fx()
    articles = load_articles()
    preds = load_predictions()
    pred = _latest_prediction(preds)

    st.title("COP/USD Intelligence Desk")
    st.caption("Un dashboard operativo: call, evidencia, importancia de noticias y feedback loop.")

    tab_decision, tab_news, tab_learning, tab_reports = st.tabs(
        ["Decision", "Noticias y topics", "Aprendizaje", "Reportes"]
    )

    with tab_decision:
        _render_decision(pred, fx, articles)
    with tab_news:
        _render_news_lab(articles)
    with tab_learning:
        _render_learning(preds)
    with tab_reports:
        _render_reports()


if __name__ == "__main__":
    main()
