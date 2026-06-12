"""Dashboard del sistema COP/USD Intelligence.

Lanzar desde la raíz del repo::

    uv run streamlit run dashboard/app.py

Muestra el producto completo: el veredicto direccional del día, las noticias
enriquecidas (GOLD), el grafo de dependencias tópicos↔entidades, la
arquitectura LangGraph y el backtesting de la Etapa 5.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Sin os.chdir ni hacks de cwd: cop_fx.paths resuelve todo desde el paquete,
# y Settings lee el .env de la raíz con ruta absoluta.
from cop_fx.paths import DATA_DIR
from cop_fx.tracking import PredictionStore, directional_backtest

st.set_page_config(page_title="COP/USD Intelligence", page_icon="💵", layout="wide")

ARTICLES_DB = DATA_DIR / "cnn_articles.db"
PREDICTIONS_DB = DATA_DIR / "predictions.db"

DIRECTION_LABEL = {
    "down": "⬇️ USD/COP BAJA (COP se fortalece)",
    "up": "⬆️ USD/COP SUBE (COP se debilita)",
    "neutral": "⏸️ NEUTRAL — el sistema se abstiene",
}


# ---------------------------------------------------------------------------
# Carga de datos (cacheada)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Descargando TRM oficial...")
def load_fx() -> pd.DataFrame:
    from cop_fx.data.fx_fetcher import FXFetcher

    return FXFetcher().fetch()


def load_articles() -> pd.DataFrame:
    if not ARTICLES_DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(ARTICLES_DB) as conn:
        return pd.read_sql("SELECT * FROM articles ORDER BY published_at DESC", conn)


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_DB.exists():
        return pd.DataFrame()
    return PredictionStore(PREDICTIONS_DB).all()


@st.cache_resource(show_spinner="Compilando el grafo LangGraph...")
def graph_mermaid() -> str:
    from cop_fx.agents.graph import build_graph

    return build_graph().compile().get_graph().draw_mermaid()


@st.cache_data(show_spinner="Backtesting ARIMA vs baselines (sin LLM)...")
def run_backtest(horizon: int, n_origins: int) -> tuple[pd.DataFrame, dict]:
    return directional_backtest(load_fx(), horizon_days=horizon, n_origins=n_origins)


def render_mermaid(code: str, height: int = 650) -> None:
    components.html(
        f"""
        <pre class="mermaid" style="background:white">{code}</pre>
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
          mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
        </script>
        """,
        height=height,
        scrolling=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💵 COP/USD Intelligence")
    st.markdown(
        "**El foco:** clasificar la **dirección** del dólar (sube/baja/neutral), "
        "no la magnitud, cruzando dos señales independientes:\n\n"
        "1. 📰 **Noticias** — agentes LangGraph clasifican cada artículo "
        "(tópico + canal de transmisión FX)\n"
        "2. 📈 **Serie de tiempo** — signo del ensemble Prophet+ARIMA\n\n"
        "Un **adjudicador** las reconcilia con abogado del diablo y "
        "confianza acotada por contrato."
    )
    st.divider()
    if st.button("▶️ Correr pipeline ahora", use_container_width=True):
        from cop_fx.agents.graph import run_pipeline

        with st.spinner("Corriendo el grafo completo (FX + noticias + LLM)..."):
            final_state = run_pipeline()
        errs = final_state.get("errors", [])
        if errs:
            st.warning(f"Completado con avisos: {errs}")
        else:
            st.success("Pipeline completado")
        st.cache_data.clear()
        st.rerun()
    st.caption(
        "La corrida descarga la TRM oficial, scrapea CNN + feeds, y gasta "
        "~8-10 llamadas a gpt-5.4-mini (≈ fracciones de centavo)."
    )

tab_call, tab_news, tab_arch, tab_backtest = st.tabs(
    ["🎯 Veredicto del día", "📰 Noticias (GOLD)", "🕸️ Arquitectura", "📈 Backtesting (Etapa 5)"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Veredicto del día
# ---------------------------------------------------------------------------

with tab_call:
    preds = load_predictions()
    if preds.empty:
        st.info("Aún no hay predicciones. Corre el pipeline desde la barra lateral.")
    else:
        latest = preds.iloc[0]
        st.subheader(f"Directional Call — {latest['run_date']} (horizonte {latest['horizon_days']} días)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dirección", DIRECTION_LABEL[latest["direction"]].split(" ")[0] + " " + latest["direction"].upper())
        c2.metric("Confianza", f"{latest['confidence']:.2f}")
        c3.metric("Reconciliación", latest["reconciliation"])
        c4.metric("TRM al predecir", f"{latest['latest_rate']:,.2f}" if latest["latest_rate"] else "—")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**📰 Señal de noticias**")
            st.write(f"Dirección: `{latest['news_direction']}` · score: `{latest['news_score']}`")
        with s2:
            st.markdown("**📈 Señal de la serie**")
            st.write(f"Dirección: `{latest['ts_direction']}` · Δ forecast: `{latest['ts_delta_pct']:+.2f}%`")

        if latest.get("top_story_title"):
            st.markdown("**📌 La noticia más importante del día** *(elegida por el agente editor)*")
            st.info(
                f"**{latest['top_story_title']}** — {latest['top_story_source']}\n\n"
                f"{latest['top_story_why']}"
            )

        st.markdown("**🧠 Racional del adjudicador**")
        st.write(latest["rationale"])
        st.markdown("**😈 Abogado del diablo** *(obligatorio por contrato — combate el sesgo de confirmación)*")
        st.warning(latest["devils_advocate"])

    st.divider()
    st.subheader("TRM oficial — USD/COP")
    try:
        fx = load_fx()
        st.line_chart(fx.set_index("ds")["y"], height=300)
        st.caption(
            f"Última: **{fx['y'].iloc[-1]:,.2f}** ({fx['ds'].iloc[-1].date()}) · "
            f"fuente: TRM oficial vía datos.gov.co"
        )
    except Exception as exc:
        st.error(f"No se pudo descargar la TRM: {exc}")


# ---------------------------------------------------------------------------
# Tab 2 — Noticias GOLD + grafo de dependencias
# ---------------------------------------------------------------------------

with tab_news:
    articles = load_articles()
    if articles.empty:
        st.info("No hay artículos en data/cnn_articles.db — ejecuta notebooks/cnn.ipynb o el pipeline.")
    else:
        st.subheader(f"Capa GOLD — {len(articles)} artículos enriquecidos")
        only_fx = st.toggle("Solo artículos con relevancia FX", value=False)
        shown = articles if not only_fx else articles[articles["fx_relevance"] != "none"]
        st.dataframe(
            shown[["title", "topic", "fx_relevance", "fx_channel", "severity", "bullish_cop", "reasoning"]],
            use_container_width=True,
            height=320,
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Tópico × relevancia FX** — el argumento del router (Etapa 2)")
            if "fx_relevance" in articles.columns:
                st.dataframe(pd.crosstab(articles["topic"], articles["fx_relevance"]))
        with c2:
            st.markdown("**Grafo de dependencias: tópicos (◼) ↔ entidades (●)** — las comunidades son los clusters del `Send` fan-out (Etapa 3)")
            try:
                import matplotlib.pyplot as plt
                import networkx as nx
                from networkx.algorithms.community import greedy_modularity_communities

                G = nx.Graph()
                for _, row in articles.iterrows():
                    if not row.get("entities"):
                        continue
                    t = f"◼ {row['topic']}"
                    G.add_node(t, kind="topic")
                    for e in json.loads(row["entities"]):
                        en = f"● {e}"
                        G.add_node(en, kind="entity")
                        w = (G.get_edge_data(t, en) or {}).get("weight", 0)
                        G.add_edge(t, en, weight=w + 1)

                if G.number_of_edges():
                    comms = list(greedy_modularity_communities(G, weight="weight"))
                    palette = plt.cm.tab10.colors
                    colors = [
                        palette[next(i for i, c in enumerate(comms) if n in c) % 10]
                        for n in G.nodes
                    ]
                    sizes = [
                        700 + 200 * G.degree(n) if G.nodes[n]["kind"] == "topic" else 100 + 70 * G.degree(n)
                        for n in G.nodes
                    ]
                    fig, ax = plt.subplots(figsize=(11, 7))
                    pos = nx.spring_layout(G, k=0.6, seed=42, weight="weight")
                    nx.draw_networkx_edges(G, pos, alpha=0.25, ax=ax)
                    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.85, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Sin entidades registradas todavía.")
            except Exception as exc:
                st.error(f"No se pudo construir el grafo: {exc}")


# ---------------------------------------------------------------------------
# Tab 3 — Arquitectura LangGraph
# ---------------------------------------------------------------------------

with tab_arch:
    st.subheader("El grafo de inteligencia (LangGraph)")
    try:
        render_mermaid(graph_mermaid())
    except Exception as exc:
        st.error(f"No se pudo renderizar el grafo: {exc}")

    st.markdown(
        """
| Patrón agéntico | Nodo(s) | Qué resuelve |
|---|---|---|
| **Routing** (gate de costo) | `check_materiality` → router condicional | Día sin noticia material ⇒ ruta barata (`skip_news`), cero tokens de análisis |
| **Orchestrator-workers** (`Send`) | `orchestrate` → `topic_worker` × N | La cantidad de workers la decide el dato (clusters del día), no el grafo |
| **Prompt chaining** | dentro de cada `topic_worker` | Extraer → clasificar → canal de transmisión → impacto |
| **Paralelización** | `fetch_fx` ‖ `fetch_news`; workers en el mismo superstep | Velocidad; señales independientes |
| **Evaluator / racionalidad** | `adjudicate` (defer=True) | Reconcilia señales, abogado del diablo obligatorio, confianza acotada **por contrato Pydantic** |
| **Tracking** (Etapa 5) | `record_prediction` | Guarda el veredicto y evalúa los pendientes con la TRM fresca |

**Decisiones de diseño clave:**
- El LLM **nunca inventa números**: la señal de la serie es el signo del ensemble (determinista) y el adjudicador devuelve solo el veredicto — las señales las inyecta el sistema.
- `fx_relevance="none"` pesa **cero en la señal por contrato** (un partido de fútbol no puede mover el score ni por error de prompt).
- `adjudicate` tiene **un solo trigger** entrante + `defer=True`: un trigger extra hace que un nodo deferred se ejecute una vez por trigger (quirk de LangGraph verificado empíricamente con `Send`).
"""
    )


# ---------------------------------------------------------------------------
# Tab 4 — Backtesting (Etapa 5)
# ---------------------------------------------------------------------------

with tab_backtest:
    st.subheader("Predicciones del sistema (se evalúan solas al vencer su horizonte)")
    preds = load_predictions()
    if preds.empty:
        st.info("Sin predicciones aún — cada corrida diaria agrega una fila.")
    else:
        st.dataframe(
            preds[
                ["run_date", "direction", "confidence", "reconciliation", "news_direction",
                 "ts_direction", "latest_rate", "actual_rate", "actual_direction", "hit"]
            ],
            use_container_width=True,
        )
        m = PredictionStore(PREDICTIONS_DB).metrics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicciones", m["n_total"])
        c2.metric("Evaluadas", m["n_evaluated"])
        c3.metric("Hit-rate (no neutrales)", f"{m['hit_rate']:.0%}" if m["hit_rate"] is not None else "—")
        c4.metric("Abstenciones", m["n_abstained"])
        if "confusion" in m:
            st.markdown("**Matriz de confusión** (predicho vs real)")
            st.dataframe(m["confusion"])
            st.markdown("**Hit-rate por confianza** — valida la calibración: ¿aciertas más cuando dices alta confianza?")
            st.dataframe(m["hit_rate_by_confidence"])

    st.divider()
    st.subheader("Backtest de la señal de serie — ARIMA vs baselines (sin LLM, sin costo)")
    st.caption(
        "La prueba de honestidad: si `momentum` (repetir el último movimiento) acierta igual "
        "que ARIMA, la serie no aporta y todo el peso recae en las noticias."
    )
    c1, c2 = st.columns(2)
    horizon = c1.slider("Horizonte (días)", 1, 10, 5)
    n_origins = c2.slider("Días de backtest", 10, 60, 30)
    if st.button("Correr backtest"):
        try:
            detail, summary = run_backtest(horizon, n_origins)
            cols = st.columns(3)
            for col, strategy in zip(cols, ("arima", "momentum", "always_up")):
                s = summary[strategy]
                col.metric(
                    strategy,
                    f"{s['hit_rate']:.0%}" if s["hit_rate"] is not None else "—",
                    f"{s['n_decided']} decisiones · {s['n_abstained']} abstenciones",
                )
            with st.expander("Detalle por día"):
                st.dataframe(detail, use_container_width=True)
        except Exception as exc:
            st.error(f"Backtest falló: {exc}")
