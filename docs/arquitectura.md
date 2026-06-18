# cop-fx-intelligence — Arquitectura, Datos y Razonamiento

> Documento de orientación. Objetivo del sistema: **determinar la dirección del USD/COP** (¿el dólar baja o no?), no la magnitud. Es un problema de **clasificación direccional**, no de regresión.

---

## 1. El objetivo, bien planteado

No predecimos *cuánto* se mueve el dólar, sino **hacia dónde**:

- `down` → el COP se fortalece / el USD/COP cae
- `up` → el COP se debilita / el USD/COP sube
- `neutral` → señales contradictorias o débiles → **el sistema se abstiene**

La abstención es parte del diseño: forzar una predicción cuando las señales no concuerdan es justo lo contrario de la "racionalidad" que buscamos. Saber cuándo *no* predecir es una capacidad, no una falla.

Consecuencia técnica clave: **el LLM nunca inventa números.** El LLM produce *dirección + razonamiento*; la serie de tiempo produce el *signo de la tendencia*. Separar estas dos responsabilidades elimina la causa #1 de alucinación.

---

## 2. Qué información traer (y qué NO)

Hoy traes solo CNN Colombia. Es necesario pero insuficiente: el USD/COP es **mitad Colombia, mitad dólar global**. Recomendación de fuentes, separando lo cualitativo (noticias) de lo cuantitativo (series numéricas baratas):

### Noticias (cualitativo — alimenta a los agentes)
| Fuente | Aporta | Cómo |
|---|---|---|
| CNN en Español + Portafolio / La República | Riesgo político/fiscal, BanRep, contexto local | RSS |
| Una fuente macro global (Fed / Reuters markets) | El lado **USD**: política monetaria de EE.UU. | RSS |

Mantenlo en 2–3 fuentes. Más fuentes = más tokens = más costo sin más señal direccional.

### Series numéricas (cuantitativo — features baratas, alta señal)
| Serie | Por qué importa para la dirección |
|---|---|
| **USD/COP** (ya la tienes) | La variable objetivo |
| **Petróleo Brent** | Colombia exporta petróleo → Brent ↑ ⇒ COP se fortalece (USD/COP ↓). Probablemente el driver fundamental más fuerte |
| **DXY (índice dólar)** | El lado dólar: DXY ↑ ⇒ USD/COP ↑ |
| **COLCAP** (opcional) | Proxy de apetito de riesgo local |

Estas vienen de APIs gratis (Alpha Vantage, stooq), sin scraping, sin costo de LLM.

### Lo que NO debes traer
**Acciones de empresas individuales.** Para la *dirección* del USD/COP aportan ruido > señal y multiplican costo. El agregado útil (COLCAP) ya cubre el sentimiento de mercado local como una sola serie numérica.

> Reencuadre: esto no es "noticias vs serie de tiempo". Es un **modelo multi-señal**: señal-noticias + señal-petróleo + señal-DXY + señal-tendencia-COP → reconciliadas en una sola llamada direccional.

---

## 3. Bases de datos — diseño económico

Tu intuición es correcta: son **dos formas de dato distintas**.

- **FX y series numéricas** = serie de tiempo (angosta, append-only, numérica) → store columnar.
- **Noticias** = documentos + búsqueda semántica (embeddings para RAG) → store relacional + vectorial.

**Pero** bajo la restricción "lo más económico posible" y dado que el acceso es *batch 1 vez al día* (sin lecturas en tiempo real), un Postgres encendido 24/7 es desperdicio. Actualizo mi recomendación anterior:

### Recomendación (cero costo idle)
```
GCS  (bronze / raw)  ── JSON crudo por fecha, antes de parsear (reproducibilidad)
        │
        ▼
BigQuery (serverless, free tier)
   ├── fx_rates              (serie de tiempo: fecha, par, valor, fuente)
   ├── market_series         (brent, dxy, colcap)
   ├── articles              (estructurado + campos del LLM: topic, severity, signal)
   ├── article_embeddings    (VECTOR — BigQuery tiene VECTOR_SEARCH nativo → RAG sin DB extra)
   └── predictions           (predicción direccional diaria + resultado real → backtesting)
```

- **Sin servidor encendido.** BigQuery cobra por consulta; a este volumen estás en free tier (~$0).
- **RAG sin pieza extra:** los embeddings viven en BigQuery con `VECTOR_SEARCH` nativo.
- **Dev local que espeja la nube:** SQLite + un índice FAISS en disco. Misma interfaz, cero costo mientras desarrollas.

> Postgres + pgvector solo tendría sentido si después quieres lecturas transaccionales/tiempo real. Para un batch diario, no.

---

## 4. Patrón agentic — la respuesta corta: **todos, en capas**

No es "o chaining o routing o parallelization o orchestrator-worker". Cada uno resuelve un nivel distinto. Así se componen:

```
                    ┌─────────────── ROUTER (gate de costo) ───────────────┐
START ──► ingestión │  ¿hay noticia material hoy? structured_output→Literal │
(datos listos)      │   no → ruta TS-only (barata)    sí → análisis completo│
                    └───────────────────────┬──────────────────────────────┘
                                            ▼
                 ┌──────────── ORCHESTRATOR–WORKERS (paralelo) ────────────┐
                 │  orchestrator agrupa noticias por tópico                 │
                 │  y dispara 1 worker por cluster vía Send (cantidad       │
                 │  dinámica). Cada worker hace PROMPT CHAINING:            │
                 │      extraer → clasificar(topic,keywords) → impacto      │
                 │  En paralelo: TS-analyst (signo del forecast)           │
                 └───────────────────────┬─────────────────────────────────┘
                                         ▼
                                    AGGREGATOR
                          (combina señales ponderadas por severidad)
                                         ▼
                 ┌────────── CAPA DE RACIONALIDAD (evaluator loop) ─────────┐
                 │  Adjudicador: reconcilia noticias vs serie de tiempo,    │
                 │  corre abogado del diablo, calibra confianza, decide     │
                 │  dirección o se abstiene. Si confianza baja → 1 RAG más  │
                 │  o degrada a neutral.                                    │
                 └───────────────────────┬─────────────────────────────────┘
                                         ▼
                              reporte → (HITL) publish
```

| Patrón | Dónde vive | Por qué ahí |
|---|---|---|
| **Routing** | Gate inicial | Si no hay noticia material, ruta barata. Control de costo |
| **Prompt chaining** | Dentro de cada worker | Sub-pasos secuenciales: extraer → clasificar → puntuar |
| **Parallelization** | Workers + TS analyst concurrentes | Velocidad; señales independientes |
| **Orchestrator–workers** | Fan-out por cluster de noticias con `Send` | La cantidad de noticias relevantes varía cada día → workers dinámicos |
| **Evaluator–optimizer** | Capa de racionalidad | Reflexión, abogado del diablo, calibración |

El patrón **dominante** es **orchestrator–workers + evaluator**. Los demás son capas de soporte.

---

## 5. La capa de racionalidad

Aquí está el corazón intelectual del proyecto. "Racionalidad" no es un prompt que diga "razona bien"; es **estructura que reduce sesgo**. El nodo *Adjudicador* recibe señales estructuradas y produce un juicio razonado:

1. **Reconciliación de señales.** Cruza dirección-noticias vs dirección-serie-de-tiempo:
   - Concuerdan (ambas `down`) → señal fuerte, alta confianza.
   - Divergen (noticias `down`, TS `up`) → **conflicto explícito**: baja confianza, explica cuál señal pesa más y por qué (p. ej. una noticia de shock domina una tendencia técnica suave).
2. **Abogado del diablo.** Antes de decidir, un paso obligatorio que construye el argumento *contrario* más fuerte. Combate el sesgo de confirmación — la diferencia entre un sistema que "razona" y uno que racionaliza.
3. **Calibración de confianza.** Devuelve `confidence ∈ [0,1]`, no falsa precisión. Confianza baja + señales en conflicto ⇒ `neutral`.
4. **Trazabilidad.** La decisión cita las noticias específicas y el signo del forecast que la sustentan. Sin cita, no hay afirmación.

Esto es el patrón **evaluator–optimizer** de tu curso (`evaluator.py`): el adjudicador *evalúa* la coherencia del juicio y puede hacer loop (pedir más contexto vía RAG) antes de cerrar.

---

## 6. Contrato de salida (lo que el sistema produce cada día)

```python
class DirectionalCall(BaseModel):
    direction: Literal["down", "up", "neutral"]
    confidence: float                      # 0..1
    horizon_days: int                      # ej. 1–5
    news_signal: NewsSignal                # dirección + drivers + citas
    ts_signal: TimeSeriesSignal            # signo del forecast + acuerdo entre modelos
    reconciliation: Literal["agree", "diverge"]
    rationale: str                         # cadena de razonamiento
    devils_advocate: str                   # el contra-argumento más fuerte
    caveats: list[str]
```

Todo vía `with_structured_output` (lo que estudiaste en `07-structural_output`).

---

## 7. Evaluación — direccional, no MAPE

Como el objetivo es dirección, **no evalúes con MAPE/RMSE.** Mide:

- **Hit-rate / accuracy direccional**: predicho vs movimiento real del día siguiente.
- **Matriz de confusión** (down/up/neutral).
- **Accuracy condicionada a confianza**: ¿aciertas más cuando dices "alta confianza"? Eso valida la calibración.
- Baseline honesto: ¿le ganas a "mañana igual que hoy" (random walk)? Si no, el modelo no aporta.

Guarda cada predicción + resultado real en la tabla `predictions` de BigQuery. Eso cierra el loop y te da backtesting gratis con el tiempo.

---

## 8. Economía — dónde está el costo y cómo aplastarlo

El costo NO está en la infra (Cloud Run scale-to-zero + Cloud Scheduler + BigQuery free tier ≈ centavos/mes). Está en **tokens de LLM**:

- **Un modelo de volumen, uno de juicio.** El proyecto corre sobre **OpenAI** (config en `cop_fx.llm` + `settings.llm_provider`). Tier `fast` = `gpt-5-mini` para todo lo que no es juicio — gate de materialidad, extracción/clasificación por artículo, agente editor y juez de keywords (alto volumen); tier `judge` = `gpt-5.4-mini` **solo** en el nodo adjudicador (1 llamada, alto valor). Cambiar de modelo o de proveedor es una línea en `config.yaml`/`.env` — todo pasa por la fábrica `get_chat_model("fast"|"judge")`.
- **Batch, no por artículo.** Manda un digest de N artículos en un prompt, no una llamada por artículo.
- **Embeddings solo de lo nuevo.** Cachea; nunca re-embebas un artículo ya visto.
- **Cap de artículos** (top-K por recencia/relevancia).
- **Cero LLM en lo determinista** (dedup, parsing, signo de la tendencia).

Resultado realista: una corrida diaria cuesta fracciones de centavo. El proyecto entero vive prácticamente en free tier.

---

## 9. Roadmap por fases

| Fase | Entregable | Reusa del curso |
|---|---|---|
| **0** | Contratos Pydantic (`Article`, `FXRate`, `DirectionalCall`) + DDL BigQuery | `07-structural_output` |
| **1** | **Extractor de noticias** (adapter por fuente, RSS, raw→GCS, estructurado→BQ) — *empieza aquí* | — |
| **2** | Extractor FX + series macro (Brent, DXY) → `fx_rates`, `market_series` | — |
| **3** | Capa RAG sobre noticias (embeddings → `article_embeddings`) | `rag.py`, `05-rags` |
| **4** | Grafo de inteligencia: router + orchestrator/`Send` + workers + adjudicador | `support/`, `orchestrator.py`, `evaluator.py` |
| **5** | Contenerizar (Dockerfile, Secret Manager, Artifact Registry) | — |
| **6** | Cloud Run Jobs + Cloud Scheduler (ingest → Pub/Sub → intelligence) | — |
| **7** | HITL (`interrupt` antes de publicar) + LangSmith (trazas) | `make_graph(checkpointer)` |

---

## 10. Mapa directo a tu repo de estudio

| Lo que programaste | Su rol en cop-fx |
|---|---|
| `support/routes/intent/route.py` | Router/gate inicial de costo |
| `orchestrator.py` (`Send` + aggregator) | Fan-out a workers por cluster de noticias |
| `evaluator.py` (generator→evaluator→loop) | Capa de racionalidad / adjudicador |
| `rag.py`, `05-rags` | Analista de noticias con RAG y citas |
| `07-structural_output` | Todos los contratos de salida |
| `support/nodes/*/{node,prompt,tools}` | Layout modular por analista |
| `make_graph(checkpointer)` | Persistencia + HITL |
| `12-paralellization` | Concurrencia de ingestión y análisis |

---

## 11. Capas de datos y dónde entra LangGraph

Tu intuición ("extraigo, le agrego topic/keywords, y eso es Raw") tiene un orden equivocado en una sola cosa: **topic y keywords NO son Raw — son Gold.** El momento en que un LLM toca el dato es la frontera entre capas. Esto importa porque define qué se reprocesa sin re-scrapear y dónde empieza el costo.

### Las tres capas (medallion)

```
BRONZE (raw)            SILVER (limpio)              GOLD (enriquecido por LLM)
─────────────────       ─────────────────────        ────────────────────────────
XML/JSON crudo     →    Article validado        →    + topic, keywords, severity,
tal cual llegó          (Pydantic, dedup, tz,        + bullish_cop, signal, embedding
del feed                URL canónica)                + DirectionalCall del día
                        DETERMINISTA, cero LLM       AQUÍ vive el LLM
```

| Capa | Qué contiene | Quién lo produce | Determinista? |
|---|---|---|---|
| **Bronze** | El feed crudo, byte por byte (`*.xml`), por fecha | `_download` + `persist_bronze` | Sí — solo I/O |
| **Silver** | `Article` parseado, deduplicado, fechas en UTC | `_parse_feed` + `from_raw` | Sí — parsing puro |
| **Gold** | topic, keywords, severity, dirección, embedding, veredicto | `NewsAnalyzer` + grafo agentic | **No — aquí razona el LLM** |

**Por qué separar Bronze de Silver:** si mañana cambias el parser (un selector, una regla de fecha), reprocesas Silver desde Bronze **sin volver a golpear el feed**. El crudo es tu fuente de verdad reproducible. Por eso `extractor.py` guarda el `.xml` *antes* de parsear.

**Por qué topic/keywords son Gold y no Raw:** los produce un modelo (o una heurística que el modelo reemplazará). No son un hecho del feed; son una *interpretación*. Mezclarlos con Raw te impide reprocesar el enriquecimiento cuando mejores el prompt — tendrías que re-scrapear para corregir una clasificación. Mantenerlos en Gold = cambias el prompt, reprocesas Gold desde Silver, cero scraping, cero feeds caídos.

### Dónde empieza LangGraph — exactamente

```
  Bronze  ───────►  Silver  ───────►  Gold
  RSS/HTTP         Pydantic          ╔══════════════════════════╗
  feedparser       dedup, tz         ║   AQUÍ ENTRA LANGGRAPH    ║
  (extractor.py)   (schemas.py)      ║  router → orchestrator →  ║
                                     ║  workers → adjudicador    ║
  ── determinista, SIN grafo ──►     ╚══════════════════════════╝
```

LangGraph **no toca la extracción.** La ingestión (Bronze→Silver) es un script batch aburrido: feedparser, httpx, Pydantic. No hay estado conversacional, no hay decisiones, no hay razonamiento — meterle un grafo sería sobre-ingeniería.

LangGraph empieza en **Silver→Gold**, cuando ya tienes `list[Article]` limpio y la pregunta deja de ser "¿cómo bajo el dato?" y pasa a ser "¿qué *significa* este dato para la dirección del dólar?". Ese "significa" es razonamiento, y razonamiento con estado, ramas y reflexión = LangGraph.

### Cuándo montas UN agente vs MÚLTIPLES

No montas multi-agente desde el día 1. La progresión correcta:

1. **Hoy (notebook):** una sola función de enriquecimiento + reconciliación. Sin grafo todavía. Ves el producto end-to-end.
2. **Un nodo LLM:** envuelves el enriquecimiento en un `StateGraph` de un solo nodo con `with_structured_output`. Ya tienes estado y trazas.
3. **Multi-agente (cuando duele):** lo montas el día que **una sola llamada ya no alcanza** — cuando el volumen de noticias varía y necesitas fan-out por tópico (`Send`), o cuando quieres el loop de abogado-del-diablo (`evaluator.py`). El disparador es la *necesidad*, no la ambición.

> Regla: cada agente nuevo debe justificar su costo en tokens con una decisión que un nodo determinista no podía tomar.

### Cómo se relaciona la noticia con el precio del dólar

Tres pasos, dos de ellos sin LLM:

1. **Señal-noticias (Gold, LLM):** cada artículo → `bullish_cop` (¿fortalece o debilita al peso?) ponderado por `severity`. Sumas → una dirección agregada de noticias.
2. **Señal-serie (determinista, sin LLM):** el ensemble (Prophet+ARIMA) da un `yhat`; comparas contra el último valor real → **signo** de la tendencia. No importa cuánto, solo el signo.
3. **Reconciliación (LLM, el adjudicador):** cruzas ambas señales. Concuerdan → alta confianza. Divergen → conflicto explícito, baja confianza o `neutral`. Aquí es donde "se relaciona la noticia con el precio": no en la extracción, sino en el juicio final que las pone una frente a la otra.

El notebook `producto_end_to_end.ipynb` ejecuta exactamente estos tres pasos para que veas el producto antes de montar nada de infra.

### Decirle a Claude Code "lee este archivo y ejecuta"

Sí — ese es el patrón correcto. Apuntas Claude Code a este doc:

> *"Lee `docs/arquitectura.md`. Implementa la Fase 1 (extractor de noticias) respetando las capas Bronze→Silver de la sección 11: el crudo se persiste antes de parsear, el parsing es determinista y sin LLM, topic/keywords NO van en esta fase (son Gold). Usa los contratos de `schemas.py`."*

Así no improvisa la arquitectura: la lee. El doc es el contrato; Claude Code es el ejecutor.

---

*Principio rector: la ingestión es determinista y aburrida a propósito. Toda la energía de LangGraph va al grafo de inteligencia — router, orchestrator-workers, RAG y adjudicador — que es donde el sistema razona sobre la dirección del dólar.*
