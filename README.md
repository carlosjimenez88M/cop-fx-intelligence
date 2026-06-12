# cop-fx-intelligence

Sistema **multiagéntico** (LangGraph) que clasifica la **dirección diaria del USD/COP** — `down` / `up` / `neutral`, no la magnitud — cruzando noticias analizadas por agentes con la señal de la serie de tiempo, bajo una capa de racionalidad que obliga al contra-argumento y **acota la confianza por contrato Pydantic**.

> La abstención (`neutral`) es una capacidad, no una falla: cuando las señales no concluyen, el sistema lo dice.

---

## ¿Qué hace cada día?

1. **Descarga la TRM oficial** (datos.gov.co, Socrata — fallback Yahoo Finance) y el contexto de mercado de ayer: bolsa colombiana (GXG), DXY y Brent.
2. **Scrapea noticias** (La República, Portafolio, CNBC Economy + CNN Colombia) — hasta 100 artículos, ordenados por **fecha con fuentes intercaladas** (mismo día = mismo pie).
3. **Un gate barato** decide si el día tiene noticia material (solo titulares) y etiqueta cada titular con un tópico grueso.
4. **Fan-out dinámico (`Send`)**: un agente analista por cluster de tópico descarga el **texto completo** de sus artículos y los clasifica — tópico (15 dominios), canal de transmisión FX, severidad, dirección, entidades.
5. **Un agente editor** elige LA noticia más importante del día y justifica por qué.
6. **El adjudicador** reconcilia la señal de noticias contra el signo del forecast (Prophet + ARIMA(2,1,2)) con el contexto de mercado como evidencia → emite el `DirectionalCall` con abogado del diablo obligatorio.
7. **Genera el reporte** Markdown, **registra la predicción** (que se auto-califica contra la TRM real al vencer su horizonte) y opcionalmente publica el tweet.

---

## El grafo

```
START ─┬─► fetch_fx ──► run_forecast (señal de la serie, sin LLM) ────────┐
       ├─► fetch_market (equity/DXY/Brent de ayer — contexto)             │
       └─► fetch_news ─► check_materiality ─(router)─┐                    │
                            │ material               ▼                    │
                            ▼                    skip_news                │
                       orchestrate                   │                    │
                            │ Send × N clusters      │                    │
                            ▼                        │                    │
                   topic_worker (dinámicos)          │                    │
                            ▼                        │                    │
                    aggregate_signals                │                    │
                            ▼                        ▼                    ▼
                     pick_top_story ──────────► adjudicate (defer=True)
                                                     │
                                      generate_report → record_prediction → publish → END
```

### Patrones agénticos (la respuesta a "¿cuál se usa?": todos, en capas)

| Patrón | Nodo(s) | Qué resuelve |
|---|---|---|
| **Parallelization** | `fetch_fx` ‖ `fetch_market` ‖ `fetch_news`; workers en el mismo superstep | Señales independientes, a la vez |
| **Routing** | `check_materiality` → conditional edges | Día sin noticia material = cero tokens de análisis |
| **Orchestrator-workers (`Send`)** | `orchestrate` → `topic_worker` × N | La cantidad de analistas la decide el dato del día, no el código |
| **Prompt chaining** | dentro de cada worker | Extraer → clasificar → canal FX → impacto |
| **Evaluator (racionalidad)** | `adjudicate` | Reconciliación explícita + abogado del diablo + confianza acotada por validadores |
| **Agente editor** | `pick_top_story` | Elige la noticia del día entre candidatos ya analizados (elige índice; no inventa hechos) |

### El principio rector: el LLM no puede inventarse un número

- El juez devuelve solo `AdjudicatorVerdict`; las señales las inyecta el sistema al componer `DirectionalCall`.
- Validadores Pydantic: divergencia ⇒ techo de confianza 0.5; < 0.35 ⇒ abstención forzada.
- `score = Σ ±(peso_severidad × peso_relevancia)` — aritmética, no opinión. `fx_relevance="none"` pesa **cero por contrato** (deportes no contamina la señal ni por error del modelo).
- La señal de la serie es el **signo** del ensemble con banda muerta — determinista de punta a punta.
- Los veredictos se hacen sobre el **texto completo del artículo** (`article_body`, cap 3.500 chars, piso de calidad 400 — menos que eso es paywall y se cae al summary RSS).

---

## Configuración

**Dos archivos, responsabilidades separadas** (precedencia: env vars > `.env` > `config.yaml` > defaults):

- [`config.yaml`](config.yaml) — **toda** la configuración operativa: modelo (`gpt-5.4-mini` en tiers fast/judge), nº de artículos, feeds, bandas muertas, tope de workers, horizonte...
- [`.env`](.env.example) — **solo secretos**: `OPENAI_API_KEY` (obligatoria), Alpha Vantage / NewsAPI / Twitter (opcionales).

Las rutas son absolutas vía `cop_fx.paths` (derivadas del paquete, nunca del cwd) — todo funciona desde cualquier directorio.

---

## Instalación y uso

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # prerequisito: uv

git clone <repo> && cd cop-fx-intelligence
cp .env.example .env       # pon tu OPENAI_API_KEY
uv sync

uv run cop-fx run                          # pipeline completo (sin tweet)
uv run streamlit run dashboard/app.py     # dashboard: veredicto, noticias, grafos, backtesting
uv run pytest -m unit                      # tests rápidos sin I/O real
```

Costo por corrida: ~9-11 llamadas a `gpt-5.4-mini` ≈ **fracciones de centavo**.

---

## Estructura

```
config.yaml                  ← TODA la configuración operativa
src/cop_fx/
├── paths.py                 ← Rutas absolutas del proyecto (nunca cwd)
├── contracts.py             ← La racionalidad: schemas Pydantic con validadores que acotan
├── llm.py                   ← Fábrica provider-agnostic, tiers fast/judge
├── config/settings.py       ← config.yaml + .env (pydantic-settings)
├── data/
│   ├── fx_fetcher.py        ← TRM oficial (datos.gov.co) → Yahoo
│   ├── market_fetcher.py    ← Brent, DXY, bolsa CO (GXG)
│   ├── news_fetcher.py      ← Feeds verificados + orden fecha/fuentes intercaladas
│   ├── article_body.py      ← Texto COMPLETO del artículo (estilo readability)
│   └── cnn_fetcher.py       ← CNN Español Colombia (RSS + HTML)
├── analysis/news_analyzer.py← Clasificación estructurada sobre texto completo
├── agents/
│   ├── state.py             ← PipelineState (reducers para el fan-out)
│   ├── nodes.py             ← Los 12 nodos del grafo
│   └── graph.py             ← build_graph() + run_pipeline()
├── timeseries/
│   ├── models.py            ← Prophet + ARIMA(2,1,2) — orden respaldado por BIC y backtest
│   ├── diagnostics.py       ← ADF/KPSS, ACF/PACF, grid AIC/BIC, Ljung-Box
│   └── evaluator.py         ← Walk-forward CV
├── tracking/
│   ├── predictions.py       ← Tabla predictions: cada veredicto se auto-califica
│   └── backtest.py          ← Backtest direccional vs baselines (momentum, always_up)
└── publishers/twitter.py    ← Tweepy v4

notebooks/                   ← El laboratorio (cada una ejecutada, con HTML en notebooks/html/)
├── 01_noticias.ipynb        ← Bronze→Silver→Gold, taxonomía, grafo de dependencias
├── 02_series_de_tiempo.ipynb← Diagnóstico Box-Jenkins + estudio de señales macro
└── 03_producto_end_to_end.ipynb ← El sistema completo corriendo, de scraping a veredicto

dashboard/app.py             ← Streamlit: veredicto del día, noticias GOLD, arquitectura, backtesting
```

---

## Resultados del estudio (los datos mandan)

- **La serie sola no predice**: ARIMA ≈ momentum ≈ 52-55% direccional a 5 días — por eso existen los agentes de noticias y la abstención.
- **El único predictor adelantado validado**: la bolsa colombiana de ayer (`equity[t-1] → cop[t]` ≈ -0.4, robusto entre muestras) → integrado como `MarketSignal`.
- **Probados y descartados con datos**: café, oro, VIX, tasas US, USD/MXN, USD/BRL, acciones individuales (lead-lag dentro de la banda de ruido; el modelo "con todo" rinde PEOR out-of-sample — overfitting).
- **ARIMA(2,1,2) dejó de ser un acto de fe**: gana por BIC, residuales ruido-blanco, y supera al ganador por AIC (2,1,3) fuera de muestra (52% vs 35%).
- El árbitro final es la tabla `predictions`: hit-rate, matriz de confusión y accuracy-por-confianza acumulándose corrida a corrida.

---

## Estado del roadmap (`docs/plan_maestro.md`)

| Etapa | Estado |
|---|---|
| 0-1. Contratos Pydantic + salidas estructuradas | ✅ |
| 2. Router de materialidad | ✅ |
| 3. Orchestrator-workers con `Send` | ✅ |
| 4. Adjudicador (capa de racionalidad) | ✅ |
| 5. Tabla `predictions` + backtest direccional | ✅ |
| — Agente editor (noticia del día), `MarketSignal`, dashboard | ✅ |
| 6. Checkpointer + HITL (`interrupt`) + memoria entre corridas | ⏳ |
| 7. GCP (Cloud Run Jobs + Scheduler + BigQuery) | ⏳ |

---

## Tests

86 tests (unit + integración con red y LLM mockeados):

```bash
uv run pytest -m unit          # contratos, nodos, tracking, diagnósticos, extractores
uv run pytest -m integration   # el grafo completo, mockeado de punta a punta
```

---

## Licencia

This project is licensed under the GNU General Public License v3.0. You are free to use, modify, and distribute this work, provided all derivatives are licensed under GPL 3.0 and proper attribution is given.
