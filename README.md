# cop-fx-intelligence

> ## 🎓 Estás en la rama del proyecto de curso (`feature/task_course`)
>
> **Estudiantes: empiecen aquí → [`docs/proyecto_individual.md`](docs/proyecto_individual.md)**
>
> Ese documento es la consigna del **proyecto individual**: qué hacer, los retos de
> *prompting* y de *arquitectura LangGraph*, cómo medir el impacto y la rúbrica.
> El resto de este README describe el sistema que van a intervenir — léanlo junto
> con [`docs/arquitectura.md`](docs/arquitectura.md) **antes** de tocar código.

---

Sistema **multiagéntico** (LangGraph) que clasifica la **dirección diaria del USD/COP** — `down` / `up` / `neutral`, no la magnitud — cruzando noticias analizadas por agentes con la señal de la serie de tiempo, bajo una capa de racionalidad que obliga al contra-argumento y **acota la confianza por contrato Pydantic**.

> La abstención (`neutral`) es una capacidad, no una falla: cuando las señales no concluyen, el sistema lo dice.

---

## ¿Qué hace cada día?

1. **Descarga la TRM oficial** (datos.gov.co, Socrata — fallback Yahoo Finance) y el contexto de mercado de ayer: bolsa colombiana (GXG), DXY y Brent.
2. **Scrapea noticias** (La República, Portafolio, CNBC Economy + CNN Colombia) — hasta 100 artículos, ordenados por **fecha con fuentes intercaladas** (mismo día = mismo pie).
3. **Un gate barato** decide si el día tiene noticia material (solo titulares) y etiqueta cada titular con un tópico grueso.
4. **Fan-out dinámico (`Send`)**: un agente analista por cluster de tópico descarga el **texto completo** de sus artículos y los clasifica — tópico (15 dominios), canal de transmisión FX, severidad, dirección, entidades.
5. **Un agente editor** elige LA noticia más importante del día y justifica por qué.
6. **El adjudicador** reconcilia la señal de noticias contra el signo del forecast (Prophet + ARIMA(2,1,2)) con el contexto de mercado como evidencia → emite el `DirectionalCall` con abogado del diablo obligatorio, `dominant_signal` y checks de coherencia.
7. **Genera el reporte** Markdown y **registra la predicción**, que se auto-califica contra la TRM real al vencer su horizonte.

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
                                      generate_report → record_prediction → human_review (HITL) → END
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
- `dominant_signal` obliga al adjudicador a declarar qué evidencia manda (`news`, `timeseries`, `market`, `none`). Si la dirección no coincide con esa señal, el código la corrige y deja el motivo en `consistency_notes`.
- `score = Σ ±(peso_severidad × peso_relevancia)` — aritmética, no opinión. `fx_relevance="none"` pesa **cero por contrato** (deportes no contamina la señal ni por error del modelo).
- La señal de la serie es el **signo** del ensemble con banda muerta — determinista de punta a punta.
- Los veredictos se hacen sobre el **texto completo del artículo** (`article_body`, cap 3.500 chars, piso de calidad 400 — menos que eso es paywall y se cae al summary RSS).

### Diseño de prompts: roles independientes, no "resúmenes"

Los prompts están separados por responsabilidad analítica:

- `check_materiality`: oficial de materialidad. Rechaza titulares sin canal FX antes de gastar tokens en cuerpos completos.
- `NewsAnalyzer`: analista escéptico de transmisión FX. Clasifica solo si hay mecanismo, sorpresa y horizonte plausible de 1-7 días.
- `pick_top_story`: editor + risk manager. Elige la noticia que puede repricear USD/COP en el horizonte, no la más dramática.
- `adjudicate`: presidente independiente de comité de inversión. Ataca cada señal, decide cuál domina o abstiene, y debe dejar checks de coherencia.

Reglas importantes: español estricto en texto natural, penalización de noticias recicladas, castigo a anuncios de impacto lejano, y preferencia por `neutral` cuando la evidencia no vence a su contraargumento.

---

## Configuración

**Dos archivos, responsabilidades separadas** (precedencia: env vars > `.env` > `config.yaml` > defaults):

- [`config.yaml`](config.yaml) — **toda** la configuración operativa: modelos (`gpt-5-mini` para el tier fast / `gpt-5.4-mini` para el tier judge), nº de artículos, feeds, bandas muertas, tope de workers, horizonte...
- [`.env`](.env.example) — **solo secretos**: `OPENAI_API_KEY` (obligatoria), Alpha Vantage / NewsAPI (opcionales).

Las rutas son absolutas vía `cop_fx.paths` (derivadas del paquete, nunca del cwd) — todo funciona desde cualquier directorio.

---

## Instalación y uso

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # prerequisito: uv

git clone <repo> && cd cop-fx-intelligence
cp .env.example .env       # pon tu OPENAI_API_KEY
uv sync

uv run cop-fx run                          # pipeline completo
uv run streamlit run dashboard/app.py     # desk minimalista: call del día, forecast y track record
uv run cop-fx-api                          # API FastAPI → http://localhost:8000/docs
uv run pytest -m unit                      # tests rápidos sin I/O real
```

Cada corrida son ~9-11 llamadas al LLM: el grueso al tier fast (`gpt-5-mini`) y **una sola** al tier judge (`gpt-5.4-mini`), el adjudicador.

### API HTTP (FastAPI)

La misma inteligencia se expone como API modular (routers + servicios + protocolos,
async de punta a punta). Endpoints bajo `/api/v1`:

| Método | Ruta | Qué devuelve |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/api/v1/predictions/latest` | El call vigente |
| `GET` | `/api/v1/predictions` · `/metrics` | Historial y métricas del track record |
| `GET` | `/api/v1/forecast` | Ensemble Prophet + ARIMA y su signo |
| `GET` | `/api/v1/news` | Noticias clasificadas (capa GOLD) por importancia |
| `GET` | `/api/v1/reports` · `/{fecha}` | Reportes diarios en Markdown |
| `POST` | `/api/v1/pipeline/runs` | Dispara una corrida (asíncrona, 202) y la sigue por `job_id` |

### Docker

Imagen multi-stage basada en `uv` (no-root, healthcheck), con targets `production`
(API), `dashboard` (Streamlit) y `test`:

```bash
docker compose up api          # API en http://localhost:8000/docs
docker compose up dashboard    # Streamlit en http://localhost:8501
docker compose run --rm test   # suite unitaria dentro del contenedor
```

### Configuración operativa

`config.yaml` es el centro de control no secreto: modelo LLM, número máximo
de noticias (`news_max_articles`), horizonte, workers, bandas de abstención y
la configuración de los insights GOLD del pipeline. La sección `keyword_*`
define stopwords de dominio, si se usa juez LLM, cuántos términos se guardan
y los umbrales de co-ocurrencia exploratoria.

El flujo de palabras importantes es:

1. La capa GOLD trae keywords por artículo desde el clasificador estructurado.
2. Se eliminan términos genéricos de dominio como `Colombia`, `mercado` o `USD`.
3. El LLM juzga si cada término candidato es accionable para USD/COP.
4. Solo con esos términos se calcula el ranking que ve el dashboard.
5. El pipeline persiste `keyword_terms` y `keyword_entities` para el dashboard.
   `keyword_pairs` queda como artefacto experimental para análisis multi-día:
   solo es útil cuando un par se repite en varias noticias, no cuando aparece
   una sola vez.

El filtro de materialidad es deliberadamente estrecho: una noticia global
entra si afecta Colombia/COP directamente, el tramo USD vía Fed/macro de EE. UU.,
petróleo/términos de intercambio, riesgo país o flujos hacia Colombia/EM.
Macro global interesante pero sin ese puente se degrada a `fx_relevance=none`.

## Memoria entre corridas y revisión humana (Etapa 6)

El grafo usa las **dos memorias** de LangGraph, cada una con un rol distinto:

- **Memoria de largo plazo (entre corridas).** El nodo `load_memory` lee el
  track-record durable de `predictions.db` (qué llamó el sistema, con cuánta
  confianza y si acertó al vencer el horizonte) y lo inyecta como contexto al
  adjudicador. El juez se calibra contra su propio historial — "tus últimas
  llamadas de alta confianza fallaron; exige más evidencia" — en vez de empezar
  cada día desde cero. Va siempre activa; corre como rama paralela sin arista
  hacia el adjudicador (no altera el conteo de triggers del nodo deferred).

- **Memoria de corto plazo + HITL (`interrupt`).** Con `--review`, el grafo se
  compila con un **checkpointer** durable (SQLite) bajo un `thread_id` y se
  pausa en `human_review` antes de finalizar, devolviendo el veredicto del día
  para que un humano lo acepte o lo rechace. La corrida se reanuda en otro
  proceso con la decisión humana:

```bash
uv run cop-fx run --review                        # corre y PAUSA pidiendo revisión
uv run cop-fx resume --thread <fecha> --approve   # acepta el veredicto
uv run cop-fx resume --thread <fecha> --reject    # rechaza el veredicto
```

El resume no recomputa el LLM: continúa desde el checkpoint (estado restaurado
de SQLite) y solo ejecuta `human_review → END`.

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
│   ├── cnn_fetcher.py       ← CNN Español Colombia (HTML primario + RSS complemento)
│   └── http.py              ← Cliente HTTP compartido: headers + reintentos (tenacity)
├── analysis/
│   ├── news_analyzer.py     ← Clasificación estructurada sobre texto completo
│   ├── topic_taxonomy.py    ← Familias de tópicos, canal e importancia
│   ├── keyword_analysis.py  ← Ranking de keywords + co-ocurrencia φ
│   └── gold_store.py        ← Persistencia GOLD a SQLite (separada del grafo)
├── agents/
│   ├── state.py             ← PipelineState (reducers para el fan-out)
│   ├── nodes.py             ← Los nodos del grafo (incl. load_memory, human_review)
│   ├── memory.py            ← Memoria de largo plazo: track-record → adjudicador
│   ├── persistence.py       ← Checkpointer (SqliteSaver) + store de memoria
│   └── graph.py             ← build_graph() / compile_graph() / run_pipeline() / resume_pipeline()
├── timeseries/
│   ├── models.py            ← Prophet + ARIMA(2,1,2) — orden respaldado por BIC y backtest
│   ├── diagnostics.py       ← ADF/KPSS, ACF/PACF, grid AIC/BIC, Ljung-Box
│   └── evaluator.py         ← Walk-forward CV
├── tracking/
│   ├── predictions.py       ← Tabla predictions: cada veredicto se auto-califica
│   └── backtest.py          ← Backtest direccional vs baselines (momentum, always_up)
└── api/                     ← FastAPI: routers + servicios + protocolos (async, SOLID)
    ├── app.py               ← create_app(): fábrica con lifespan, CORS y middleware
    ├── routers/             ← health, predictions, forecast, news, reports, pipeline
    ├── services/            ← lógica desacoplada del HTTP (envuelve stores y pipeline)
    ├── schemas.py           ← DTOs de entrada/salida (separados del dominio)
    └── errors.py            ← errores de dominio → handlers JSON uniformes

notebooks/                   ← El laboratorio (cada una ejecutada, con HTML en notebooks/html/)
├── 01_noticias.ipynb        ← Bronze→Silver→Gold, taxonomía, grafo de dependencias
├── 02_series_de_tiempo.ipynb← Diagnóstico Box-Jenkins + estudio de señales macro
└── 03_producto_end_to_end.ipynb ← El sistema completo corriendo, de scraping a veredicto

dashboard/app.py             ← Streamlit minimalista: call del día, forecast y track record
Dockerfile · docker-compose.yml ← Imagen multi-stage (uv) para API, dashboard y tests
```

---

## Resultados del estudio (los datos mandan)

- **La serie sola no predice**: ARIMA ≈ momentum ≈ 52-55% direccional a 5 días — por eso existen los agentes de noticias y la abstención.
- **El único predictor adelantado validado**: la bolsa colombiana de ayer (`equity[t-1] → cop[t]` ≈ -0.4, robusto entre muestras) → integrado como `MarketSignal`.
- **Probados y descartados con datos**: café, oro, VIX, tasas US, USD/MXN, USD/BRL, acciones individuales (lead-lag dentro de la banda de ruido; el modelo "con todo" rinde PEOR out-of-sample — overfitting).
- **ARIMA(2,1,2) dejó de ser un acto de fe**: gana por BIC, residuales ruido-blanco, y supera al ganador por AIC (2,1,3) fuera de muestra (52% vs 35%).
- El árbitro final es la tabla `predictions`: hit-rate, matriz de confusión, accuracy-por-confianza y ahora hit-rate por componente (`final`, `news`, `ts`, `market`) acumulándose corrida a corrida.

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
| 6. Checkpointer + HITL (`interrupt`) + memoria entre corridas | ✅ |
| 7. GCP (Cloud Run Jobs + Scheduler + BigQuery) | ⏳ |

---

## Tests

88 tests (unit + integración con red y LLM mockeados):

```bash
uv run pytest -m unit          # contratos, nodos, tracking, diagnósticos, extractores
uv run pytest -m integration   # el grafo completo, mockeado de punta a punta
```

---

## Licencia

This project is licensed under the GNU General Public License v3.0. You are free to use, modify, and distribute this work, provided all derivatives are licensed under GPL 3.0 and proper attribution is given.
