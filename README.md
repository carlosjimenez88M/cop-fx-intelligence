# cop-fx-intelligence

Pipeline de inteligencia diaria para la tasa de cambio **COP/USD** (Peso colombiano / Dólar estadounidense).

Combina datos históricos de FX, scraping de noticias económicas colombianas, clasificación con IA (Claude), modelos de pronóstico de series de tiempo y publicación automática en Twitter/X.

---

## ¿Qué hace?

Cada día hábil a las **6:00 AM hora Colombia (UTC-5)** el pipeline:

1. Descarga el histórico diario COP/USD desde Alpha Vantage (fallback: BanRep TRM)
2. Extrae noticias económicas de feeds RSS + NewsAPI en español
3. Clasifica cada artículo por **tema** y **severidad de impacto en el tipo de cambio** usando Claude
4. Ejecuta pronósticos paralelos con **Prophet** y **ARIMA**, luego ensambla los resultados
5. Genera un reporte Markdown con análisis y tabla de pronóstico a 7 días
6. Publica un tweet con el resumen ejecutivo

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions (cron / deploy)                   │
│   daily_report.yml (6 AM COT)    deploy.yml (manual / post-CI)         │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │  uv run cop-fx run [--publish]
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LangGraph StateGraph                            │
│                                                                         │
│   START                                                                 │
│     ├── fetch_fx ──────────────────────────► run_forecast               │
│     │   FXFetcher                              ProphetForecaster         │
│     │   Alpha Vantage / BanRep                 ARIMAForecaster           │
│     │                                          ensemble_forecast         │
│     └── fetch_news ──► analyze_news ──────────────────────┐            │
│         NewsFetcher     NewsAnalyzer                        │            │
│         RSS + NewsAPI   Claude (topic/severity)            ▼            │
│                                                    generate_report       │
│                                                    Markdown + tweet      │
│                                                            │            │
│                                                            ▼            │
│                                                         publish          │
│                                                         Tweepy v4        │
│                                                            │            │
│                                                           END           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Módulos

```
src/cop_fx/
│
├── logger.py              ← Logger con colores ANSI (sin imports relativos)
│                            DEBUG=dim, INFO=cyan, SUCCESS=green,
│                            WARNING=yellow, ERROR=red, CRITICAL=bold red
│
├── config/
│   └── settings.py        ← Todas las variables de entorno (pydantic-settings)
│
├── data/
│   ├── fx_fetcher.py      ← Descarga COP/USD histórico (Alpha Vantage + BanRep)
│   └── news_fetcher.py    ← RSS feeds + NewsAPI → lista de Article
│
├── timeseries/
│   ├── models.py          ← ProphetForecaster, ARIMAForecaster, ensemble_forecast
│   └── evaluator.py       ← MAE / RMSE / MAPE + walk-forward cross-validation
│
├── analysis/
│   └── news_analyzer.py   ← Clasifica artículos vía LLM, fallback por keywords
│
├── agents/
│   ├── state.py           ← PipelineState TypedDict (contrato del grafo)
│   ├── nodes.py           ← Funciones de nodo: fetch_fx, fetch_news,
│   │                          analyze_news, run_forecast, generate_report, publish
│   └── graph.py           ← build_graph() + run_pipeline()
│
├── publishers/
│   └── twitter.py         ← TwitterPublisher (Tweepy v4, OAuth 1.0a)
│
└── cli.py                 ← Entrypoint: `cop-fx run [--publish] [--date YYYY-MM-DD]`
```

### Flujo de datos

```
FX histórico (DataFrame ds/y)
        │
        ├──► ProphetForecaster ──► ForecastResult
        └──► ARIMAForecaster   ──► ForecastResult
                                        │
                                   ensemble_forecast
                                        │
                               ┌────────┴─────────┐
                               │   PipelineState  │
                               │  + news_summary  │
                               │  + report_md     │
                               └────────┬─────────┘
                                        │
                                  generate_report → reports/report_YYYY-MM-DD.md
                                        │
                                   publish → tweet_id
```

### GitHub Actions

| Workflow | Trigger | Propósito |
|---|---|---|
| `ci.yml` | Push / PR a `master` | Lint + tests en Python 3.11 y 3.12 |
| `daily_report.yml` | Cron L-V 11:00 UTC | Ejecución automática diaria |
| `deploy.yml` | Manual / post-CI | Deploy bajo demanda con parámetros |

El `deploy.yml` tiene:
- **Gate**: no corre si el CI anterior falló
- **Concurrency lock**: evita dos deploys simultáneos en el mismo environment
- **Step summary**: adjunta las primeras 60 líneas del reporte al resumen del job
- **Artifact**: guarda `reports/` y la salida del pipeline por 90 días

---

## Instalación rápida

```bash
# Prerequisito: uv  (gestor de paquetes)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/carlosdaniel/cop-fx-intelligence.git
cd cop-fx-intelligence

cp .env.example .env      # edita .env y pon al menos ANTHROPIC_API_KEY
uv sync                   # instala todo en .venv (Python 3.11)
uv run cop-fx run         # ejecuta el pipeline sin publicar tweet
```

### Con make

```bash
make install      # uv sync --all-extras
make test-unit    # 20 tests unitarios (sin APIs reales)
make lint         # ruff check
make run          # dry-run pipeline
make run-publish  # pipeline + tweet
```

---

## Variables de entorno

Ver [`.env.example`](.env.example) para la lista completa.

| Variable | Obligatoria | Descripción |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Clave Claude API |
| `ALPHA_VANTAGE_API_KEY` | ⚪ | FX data (gratis en alphavantage.co, fallback BanRep) |
| `NEWSAPI_KEY` | ⚪ | NewsAPI.org (gratis, los RSS funcionan sin ella) |
| `TWITTER_API_KEY` + 4 más | ⚪ | Solo necesarios con `TWITTER_ENABLED=true` |
| `LLM_MODEL` | ⚪ | Default: `claude-sonnet-4-6` |
| `FORECAST_HORIZON_DAYS` | ⚪ | Default: `7` |

---

## Logger de colores

El módulo `cop_fx.logger` provee un logger ANSI con niveles:

```python
from cop_fx.logger import get_logger

log = get_logger(__name__)
log.debug("cargando config…")       # dim blanco
log.info("tasa actual: %s", 4200)   # cyan
log.success("pipeline completo")    # verde ✔  (nivel custom 25)
log.warning("usando fallback BanRep")  # amarillo
log.error("tweet falló: %s", err)   # rojo
log.critical("error irrecuperable") # rojo bold
```

No requiere imports relativos — funciona en cualquier script o notebook con sólo `from cop_fx.logger import get_logger`.

---

## Tests

```
tests/
├── unit/
│   ├── test_timeseries.py      # Prophet, ARIMA, ensemble, evaluator (8 tests)
│   ├── test_news_analyzer.py   # Clasificador LLM + fallback keywords (5 tests)
│   └── test_graph_nodes.py     # Nodos LangGraph mockeados (6 tests)
└── integration/
    └── test_full_pipeline.py   # Pipeline completo con APIs mockeadas (2 tests)
```

```bash
uv run pytest -m unit        # rápidos, sin I/O real
uv run pytest -m integration # pipeline end-to-end mockeado
```

---

## Licencia

This project is licensed under the GNU General Public License v3.0. You are free to use, modify, and distribute this work, provided all derivatives are licensed under GPL 3.0 and proper attribution is given.
