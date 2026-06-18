# Plan Maestro — de notebook a sistema agéntico direccional USD/COP

> Complemento de [`arquitectura.md`](arquitectura.md). Ese documento define el **qué** (arquitectura, capas de datos, patrones). Este define el **cómo y en qué orden**, mapeado a tu curso real de LangChain/LangGraph (`~/Documents/Education/AI_Master/Langchain_LangGraph`) y configurado para correr sobre OpenAI `gpt-5.4-mini`.

---

## 1. Diagnóstico: dónde estás hoy

Lo que ya existe en el repo (más de lo que crees):

| Pieza | Estado | Archivo |
|---|---|---|
| Fetcher FX (Alpha Vantage + BanRep) | ✅ Completo | `src/cop_fx/data/fx_fetcher.py` |
| Scraper CNN Colombia (RSS + HTML) | ✅ Completo | `src/cop_fx/data/cnn_fetcher.py` |
| Prophet + ARIMA + ensemble + walk-forward eval | ✅ Completo | `src/cop_fx/timeseries/` |
| Clasificador de noticias LLM (topic, severity, bullish_cop) | ✅ Funcional, sin `with_structured_output` | `src/cop_fx/analysis/news_analyzer.py` |
| Grafo LangGraph lineal (6 nodos, fetch paralelo) | ✅ Básico | `src/cop_fx/agents/graph.py` |
| Fábrica LLM provider-agnostic con tiers fast/judge | ✅ Completo | `src/cop_fx/llm.py` |
| Notebook exploratorio CNN (scraping → LLM → SQLite) | ✅ En curso | `notebooks/cnn.ipynb` |
| Notebook walking-skeleton (7 pasos, principio→fin) | ✅ Existe | `notebooks/producto_end_to_end.ipynb` |

Lo que **falta** (y es donde está tu aprendizaje):

1. **Salidas estructuradas con Pydantic** (`with_structured_output`) en todos los puntos donde el LLM responde.
2. **Router** (gate de materialidad) — patrón del módulo 16 de tu curso.
3. **Orchestrator–workers con `Send`** — fan-out dinámico por cluster de tópicos. *Tu curso no cubre `Send`*: este proyecto es donde lo aprendes.
4. **Capa de racionalidad** (adjudicador + abogado del diablo + calibración) — evolución del evaluator.
5. **Tabla `predictions`** para cerrar el loop: predicción de hoy vs movimiento real de mañana → hit-rate direccional.
6. **Persistencia/checkpointer** y HITL antes de publicar — módulos 06, 07 y 12 de tu curso.

---

## 2. Respuestas directas a tus preguntas

Estas respuestas están desarrolladas en `arquitectura.md`; aquí va el veredicto ejecutivo:

**¿Topic y keywords son Raw?** No. Raw (Bronze) es el feed crudo tal cual llegó. Topic/keywords los produce un LLM → son **Gold**. La regla: *el momento en que un LLM toca el dato es la frontera de capa*. Si mejoras el prompt, reprocesas Gold desde Silver sin re-scrapear. (§11 de arquitectura.md)

**¿Traigo acciones de empresas?** No individuales — ruido > señal para un problema *direccional*. Lo que sí robustece: **Brent** (Colombia exporta petróleo), **DXY** (el lado dólar de la ecuación) y opcionalmente **COLCAP** como proxy de riesgo local. Todas son series numéricas gratis, cero tokens. (§2)

**¿Qué patrón agéntico: chaining, routing, paralelización u orchestrator-worker?** Todos, en capas — no compiten, se componen: routing como gate de costo → orchestrator-workers (`Send`) para fan-out por tópico → prompt chaining dentro de cada worker → evaluator-optimizer como capa de racionalidad. El dominante es **orchestrator-workers + evaluator**. (§4)

**¿Dónde empieza LangGraph?** En Silver→Gold. La extracción (Bronze→Silver) es un script determinista aburrido: feedparser, httpx, Pydantic — sin grafo. LangGraph entra cuando la pregunta cambia de "¿cómo bajo el dato?" a "¿qué *significa* este dato para la dirección del dólar?". (§11)

**¿Cuándo monto múltiples agentes?** Cuando una sola llamada ya no alcanza, no antes. Progresión: función simple en notebook → un nodo LLM en StateGraph → multi-agente con `Send` cuando el volumen variable de noticias lo exija. Cada agente nuevo debe justificar sus tokens con una decisión que un nodo determinista no podía tomar. (§11)

**¿Cómo se relaciona la noticia con el precio?** En la **reconciliación**, no en la extracción. Señal-noticias (LLM: suma ponderada de `bullish_cop` × severity) vs señal-serie (determinista: signo de `yhat` del ensemble vs último real) → el adjudicador las cruza: concuerdan = alta confianza; divergen = conflicto explícito, baja confianza o abstención (`neutral`). (§11, paso 3)

---

## 3. Configuración del modelo: gpt-5.4-mini

La fábrica `get_chat_model()` ya es provider-agnostic. Solo cambia `.env`:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-mini          # tier "fast": clasificación por lote, alto volumen
LLM_MODEL_JUDGE=gpt-5.4-mini    # tier "judge": empieza igual; sube de modelo SOLO si
                                 # el hit-rate del adjudicador lo justifica
LLM_TEMPERATURE=0.0              # clasificación direccional = determinismo
```

Principio: **no pagues un modelo grande hasta tener una métrica que demuestre que el chico no alcanza.** La tabla `predictions` (Etapa 5) es la que te dará esa evidencia. En `notebooks/cnn.ipynb` reemplaza la llamada directa a `claude-haiku` por `get_chat_model("fast")` — así el notebook y el pipeline comparten la misma configuración.

---

## 4. Roadmap de construcción mapeado a tu curso

Cada etapa produce algo que funciona y ejercita un módulo concreto del curso. No saltes etapas: el orden es deliberado (primero contratos, luego un nodo, luego el grafo, luego racionalidad).

### Etapa 0 — Contratos Pydantic (1 sesión)
**Curso:** `notebooks/02_schemas/` (parsers y Pydantic)

Define en `src/cop_fx/ingestion/news/schemas.py` y un nuevo `src/cop_fx/contracts.py`:

```python
class EnrichedArticle(BaseModel):          # Gold por artículo
    topic: Literal["monetary_policy", "trade", "political_risk",
                   "commodities", "macro", "other"]
    keywords: list[str] = Field(min_length=1, max_length=5)
    severity: Literal["high", "medium", "low"]
    bullish_cop: bool
    reasoning: str = Field(max_length=200)

class NewsSignal(BaseModel):               # agregado de noticias
    direction: Literal["down", "up", "neutral"]
    score: float                            # suma ponderada por severity
    drivers: list[str]                      # títulos/URLs que sustentan

class TimeSeriesSignal(BaseModel):         # determinista, sin LLM
    direction: Literal["down", "up", "neutral"]
    yhat_delta_pct: float
    models_agree: bool                      # ¿Prophet y ARIMA dan el mismo signo?

class DirectionalCall(BaseModel):          # producto final diario
    direction: Literal["down", "up", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    horizon_days: int
    news_signal: NewsSignal
    ts_signal: TimeSeriesSignal
    reconciliation: Literal["agree", "diverge", "partial"]
    rationale: str
    devils_advocate: str
    caveats: list[str]
```

Esto es también tu entrenamiento de Pydantic: validadores (`field_validator` para que `confidence` baje si `reconciliation="diverge"`), `Field` constraints, `Literal` en vez de strings libres.

### Etapa 1 — Un solo nodo con salida estructurada (1 sesión)
**Curso:** `notebooks/05_intro_langgraph/main.py` (StateGraph básico)

Migra `NewsAnalyzer` a `get_chat_model("fast").with_structured_output(BatchAnalysis)` — se acaba el parsing manual de JSON y los fallbacks frágiles. Envuélvelo en un `StateGraph` de un nodo. Ya tienes trazas y estado tipado. **Este es el momento exacto en que "empieza LangGraph" en tu repo.**

### Etapa 2 — Router: el gate de materialidad (1 sesión)
**Curso:** `notebooks/16_routing/` + `notebooks/06_agent_state/main.py` (conditional edges)

Nodo inicial que responde una sola pregunta con structured output:

```python
class MaterialityGate(BaseModel):
    has_material_news: bool
    reason: str
```

`add_conditional_edges`: si `False` → ruta TS-only (cero tokens adicionales, el forecast decide solo con baja confianza); si `True` → análisis completo. Es el patrón de routing de tu curso aplicado a control de costo, no a departamentos de soporte.

### Etapa 3 — Orchestrator–workers con `Send` (2 sesiones)
**Curso:** `notebooks/13_Multi-Agent_Architecture/` y `15_multiagent_orchestration/` dan la base conceptual (supervisor + workers), pero el curso **no enseña `Send`** — aquí lo aprendes:

```python
from langgraph.types import Send

def fan_out(state: PipelineState):
    clusters = group_by_topic(state["raw_articles"])   # determinista, sin LLM
    return [Send("topic_worker", {"cluster": c}) for c in clusters]

workflow.add_conditional_edges("orchestrator", fan_out)
```

Cada worker hace **prompt chaining** (curso: `notebooks/03_multi-step_workflows/`): extraer → clasificar → puntuar impacto. El número de workers varía cada día con las noticias — eso es lo que `Send` resuelve y un grafo estático no puede. El agregador usa un reducer `Annotated[list[EnrichedArticle], operator.add]` (curso: `notebooks/06_agent_state/`).

### Etapa 4 — La capa de racionalidad: el adjudicador (2 sesiones)
**Curso:** no hay módulo equivalente — es tu aporte original sobre la base del patrón evaluator.

Un nodo `adjudicate` con tier `judge` que recibe `NewsSignal` + `TimeSeriesSignal` y produce `DirectionalCall`. La racionalidad **no es un prompt que diga "razona bien"; es estructura**:

1. **Reconciliación explícita.** El prompt obliga a declarar si las señales concuerdan o divergen *antes* de decidir.
2. **Abogado del diablo obligatorio.** El schema exige `devils_advocate: str` — el modelo no puede emitir un veredicto sin haber construido el mejor argumento contrario. Esto combate el sesgo de confirmación a nivel de contrato, no de cortesía.
3. **Calibración.** `confidence` con reglas duras (un `field_validator` o un nodo determinista posterior): divergencia ⇒ techo de confianza 0.5; concordancia con `models_agree=True` ⇒ piso 0.6. El LLM propone, el código acota.
4. **Abstención como salida válida.** Si confianza < umbral ⇒ `direction="neutral"`. Saber no predecir es la capacidad, no la falla.
5. **Trazabilidad.** Sin `drivers` citados, el veredicto se rechaza y se reintenta (loop evaluator, máx 2 vueltas).

### Etapa 5 — Cerrar el loop: backtesting direccional (1 sesión)
**Curso:** `notebooks/09_databases/` (SQL) y `notebooks/11_onservability/` (métricas)

Tabla `predictions` (SQLite local, BigQuery después): cada día guarda el `DirectionalCall`; al día siguiente un job compara contra el movimiento real. Métricas: **hit-rate direccional, matriz de confusión down/up/neutral, accuracy condicionada a confianza** (¿aciertas más cuando dices "alta confianza"? — eso valida la calibración). Baseline obligatorio: ¿le ganas al random walk ("mañana = hoy")? No MAPE — el objetivo es dirección. (arquitectura.md §7)

### Etapa 6 — Persistencia, HITL y memoria (cuando lo anterior funcione)
**Curso:** `notebooks/06_agent_state/`, `07_agent_memory/`, `12_memory_langgraph/`

`SqliteSaver` como checkpointer, `interrupt` antes de finalizar (tú apruebas el veredicto), y memoria entre corridas: "ayer dije `down` con 0.7 y acerté/fallé" entra como contexto del adjudicador de hoy. RAG sobre noticias históricas (`notebooks/10_agentic_rag/`) entra aquí, no antes: solo cuando el adjudicador necesite contexto que el batch del día no trae.

---

## 5. El grafo objetivo (estado final de las etapas 1–4)

```
START
  ├─► fetch_fx ─► run_forecast ─► ts_signal (determinista) ──────────┐
  └─► fetch_news ─► materiality_router                                │
                       │ no material                                  │
                       ├──────────────────────────────────────────────┤
                       │ material                                     ▼
                       ▼                                          adjudicate
                  orchestrator ─ Send ─► topic_worker × N ─► aggregate ─┘
                                          (chaining:                  │
                                           extraer→clasificar→impacto)│
                                                                      ▼
                                              ¿drivers citados? ──no──► retry (≤2)
                                                                      │ sí
                                                                      ▼
                                                       generate_report ─► publish
                                                                          (HITL)
```

El grafo actual de `agents/graph.py` es el esqueleto correcto — las etapas lo van enriqueciendo nodo a nodo, nunca lo reescriben desde cero.

---

## 6. Plan para las notebooks

Tu instinto de "ver el grafo relacional topics↔noticias" es buen EDA, pero es un *medio*, no el producto. Organización recomendada:

1. **`cnn.ipynb`** (sigue siendo tu laboratorio de ingesta): scraping → enriquecimiento → SQLite. Cambios: (a) usa `get_chat_model("fast")` con `gpt-5.4-mini` en vez del modelo hardcodeado; (b) separa explícitamente las celdas en secciones **BRONZE** (fetch crudo), **SILVER** (DataFrame validado con Pydantic, dedup) y **GOLD** (topic/keywords del LLM) — que la notebook *enseñe* la frontera de capas; (c) el grafo networkx topics↔noticias vive en GOLD como diagnóstico de clustering — es exactamente el insumo del `group_by_topic` del orchestrator (Etapa 3): si en el grafo ves que 8 noticias cuelgan de `political_risk`, ese es un cluster que merece su propio worker.

2. **`producto_end_to_end.ipynb`** (ya existe, es tu walking skeleton): mantenlo como la demo principio→fin. A medida que completes etapas, reemplaza sus heurísticas por los componentes reales: la reconciliación manual de la celda 6 se convierte en el nodo `adjudicate`, el dataclass `DirectionalCall` se convierte en el BaseModel del contrato.

3. **Nueva: `grafo_inteligencia.ipynb`** (al llegar a Etapa 3): construir el grafo LangGraph incrementalmente celda a celda, visualizándolo con `graph.get_graph().draw_mermaid_png()` después de cada nodo añadido. Esa es la notebook donde *dominas* LangGraph.

---

## 7. Orden de ejecución sugerido (próximas 4 semanas)

| Semana | Entregable | Verificación |
|---|---|---|
| 1 | Etapa 0 + 1: contratos Pydantic + nodo único con `with_structured_output` sobre `gpt-5.4-mini` | `cnn.ipynb` corre Bronze→Gold sin parsing manual de JSON |
| 2 | Etapa 2 + 3: router + `Send` fan-out | Día sin noticias materiales cuesta ~1 llamada; día cargado dispara N workers |
| 3 | Etapa 4: adjudicador con abogado del diablo y calibración acotada por código | `producto_end_to_end.ipynb` produce un `DirectionalCall` validado |
| 4 | Etapa 5: tabla `predictions` + script de hit-rate | Primer reporte: accuracy vs random walk |

GCP (Cloud Run + BigQuery + Scheduler) queda para después de la semana 4 — el diseño ya lo contempla (arquitectura.md §3, §9 fases 5–6) y nada de lo anterior habrá que reescribir: SQLite→BigQuery y cron local→Scheduler son swaps de adapter, no de arquitectura.

---

*Principio rector: cada etapa termina con algo ejecutable y una métrica. El multi-agente se gana con necesidad demostrada, no se monta por ambición. La racionalidad vive en los contratos (schemas que obligan al contra-argumento y acotan la confianza), no en los adjetivos del prompt.*
