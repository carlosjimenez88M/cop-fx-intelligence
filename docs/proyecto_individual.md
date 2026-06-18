# Proyecto individual — Patrones agénticos para la dirección del USD/COP

> **Modalidad:** individual y auto-dirigido. Este documento te da el mapa, los
> retos y las pistas; **no** te da las soluciones. El objetivo no es "que corra",
> sino que justifiques cada decisión con evidencia.
>
> **Pre-requisitos de lectura:** `README.md` (qué hace el sistema) y
> `docs/arquitectura.md` (por qué está diseñado así). Este proyecto asume que ya
> los leíste.

---

## 0. Cómo usar este documento

El eje del proyecto es **un patrón agéntico**. El sistema ya tiene varios patrones
en su forma de *un solo paso* — sin lazos, sin herramientas, sin planeación. Tu
trabajo es tomar **una** de estas tres pistas y llevarla a su forma completa:

- **Pista 1 — ReAct:** el analista de noticias razona *y actúa* consultando datos.
- **Pista 2 — Evaluator-optimizer:** el adjudicador se critica y se corrige en un lazo.
- **Pista 3 — Orchestrator-workers:** el orquestador *planea* analistas especializados.

**Eliges UNA y la haces a fondo.** Un patrón implementado, medido y defendido vale
más que tres a medias. Cada pista combina una *técnica de prompting* (cómo le
hablas al modelo) con un *cableado de LangGraph* (cómo conectas los nodos): las dos
caras del mismo patrón.

Trabaja en este orden, igual para las tres pistas:

1. **Diagnostica** (§4): lee los prompts y la topología; nombra lo que ya existe.
2. **Hipótesis:** escribe, *antes* de tocar código, "creo que este patrón mejorará
   la métrica Y porque Z".
3. **Línea base:** mide el sistema actual con la infraestructura del repo (§2.1, §6).
4. **Implementa** tu patrón en una rama.
5. **Vuelve a medir** y compara contra tu línea base.
6. **Documenta** qué pasó — incluido lo que *no* funcionó.

---

## 1. El sistema en una pantalla

El pipeline es un `StateGraph` de LangGraph que cada día decide la **dirección**
del USD/COP (`up` / `down` / `neutral`) cruzando tres señales bajo una capa de
racionalidad. No predice magnitud; **el LLM nunca inventa números**.

```
START ─┬─► fetch_fx ──► run_forecast (señal de serie, sin LLM) ──────────┐
       ├─► fetch_market (equity/DXY/Brent — contexto)                    │
       ├─► load_memory (track-record histórico)                         │
       └─► fetch_news ─► check_materiality ─(router)─┐                   │
                            │ material               ▼                   │
                            ▼                    skip_news               │
                       orchestrate                   │                   │
                            │ Send × N clusters      │                   │
                            ▼                        │                   │
                   topic_worker (dinámicos)          │                   │
                            ▼                        ▼                   ▼
                    aggregate_signals ───► pick_top_story ──► adjudicate (defer)
                                                                   │
                                  generate_report → record_prediction → human_review (HITL) → END
```

> Tu producto final es el `DirectionalCall` y el reporte Markdown. El nodo
> `human_review` es un punto de control humano (HITL) opcional que pausa para
> revisar el veredicto antes de cerrar la corrida.

**Dónde vive cada cosa (puntos de entrada al código):**

| Pieza | Archivo |
|---|---|
| Cableado del grafo | `src/cop_fx/agents/graph.py` |
| Nodos (toda la lógica + prompts) | `src/cop_fx/agents/nodes.py` |
| Estado compartido (`PipelineState`) | `src/cop_fx/agents/state.py` |
| Prompt del analista de noticias | `src/cop_fx/analysis/news_analyzer.py` |
| Contratos Pydantic (salida estructurada) | `src/cop_fx/contracts.py` |
| Series de mercado (Brent/DXY/equity) | `src/cop_fx/data/market_fetcher.py` |
| Serie TRM | `src/cop_fx/data/fx_fetcher.py` |
| Memoria / track-record | `src/cop_fx/agents/memory.py` |
| Registro y auto-evaluación de predicciones | `src/cop_fx/tracking/predictions.py` |
| Backtesting direccional (solo serie) | `src/cop_fx/tracking/backtest.py` |

Los prompts ya están en el código como constantes: `_MATERIALITY_PROMPT`,
`_TOP_STORY_PROMPT`, `_ADJUDICATOR_PROMPT` (en `nodes.py`) y `_PROMPT_TEMPLATE`
(en `news_analyzer.py`). **Empieza leyéndolos enteros.**

---

## 2. La intención del proyecto (tu vara de medir)

Todo cambio se juzga contra estos objetivos. Si tu cambio no mueve ninguno, no
sirve aunque "se vea más elegante":

1. **Acierto direccional** — ¿el sistema acierta `up`/`down` más seguido que un
   baseline tonto (momentum, always_up)? (Ojo: `directional_backtest` solo mide la
   *serie*; medir el acierto del **LLM** es harina aparte — ver §2.1.)
2. **Calidad de la abstención** — `neutral` debe aparecer cuando las señales no
   concluyen, **no** como escape fácil. Un sistema que dice `neutral` siempre
   tiene 0 errores y 0 valor.
3. **Cero alucinación numérica/factual** — el razonamiento solo puede citar
   hechos presentes en las señales de entrada.
4. **Auditabilidad** — debes poder explicar *por qué* el sistema decidió lo que
   decidió (señal dominante, abogado del diablo, calibración).
5. **Costo** — menos tokens / menos llamadas a LLM por la misma o mejor señal es
   una mejora legítima y valiosa.

> Regla de oro: **una mejora que no puedas medir contra uno de estos cinco puntos
> no es una mejora — es una opinión.**

### 2.1 Qué se puede medir (y qué NO) — léelo o medirás humo

Esta es la parte que el repo **todavía no te resuelve**, y entenderla es la mitad
del proyecto. Antes de prometer "subí el acierto direccional", verifica con qué
instrumento:

- **`directional_backtest` mide SOLO la pata de serie de tiempo.** Compara
  `arima` / `momentum` / `always_up`, todas deterministas. **No ejecuta ningún
  LLM.** Si cambias un prompt o agregas un patrón agéntico, este número **no se
  mueve** — no porque tu cambio sea malo, sino porque el backtest ni lo toca. Úsalo
  como *baseline de la señal de serie*, no como juez de tu patrón.
- **El acierto direccional del LLM vive en `predictions.db`** (`PredictionStore`),
  que `record_prediction` llena a razón de **~1 fila por corrida** y auto-califica
  con `evaluate_pending` al vencer el horizonte. **No existe un harness de
  backfill/replay**: no puedes recrear 60 días de decisiones del LLM ejecutando un
  comando. Con el repo tal cual, **no obtendrás una muestra estadísticamente útil
  del acierto del LLM en el plazo de un curso.**

Por eso, para medir tu patrón usarás sobre todo **métricas-proxy offline** sobre
entradas congeladas (ver la sección "Cómo lo mides" de tu pista) — y, si tienes
tiempo, el **harness de replay** (§6) que desbloquea la medición de acierto real.

> **Iteración offline sin quemar API keys:** stubbea el LLM con
> `unittest.mock.patch` como en `tests/unit/test_graph_nodes.py`, y aliméntalo con
> artículos guardados como *fixtures* que capturaste tú. Ojo: el GOLD store
> (`data/cnn_articles.db`) guarda la **salida** de la clasificación (`summary` +
> tópico/severidad), **no** el cuerpo crudo que leyó el analista — para congelar
> *entradas* tienes que guardarlas tú mismo. Verifícalo antes de confiar en él.

---

## 3. Reglas del juego

- **Una rama por experimento.** Nombra claro: `exp/react-worker`,
  `exp/evaluator-loop`, `exp/orchestrator-personas`.
- **Sé crítico con el data flow antes de declarar algo terminado.** Verifica
  *qué texto* le llega realmente al LLM (no asumas que el cuerpo del artículo
  llegó completo). Mira el log y la salida estructurada, no solo el reporte final.
- **El LLM produce dirección + razonamiento; la serie produce el signo.** No
  difumines esta frontera: si tu patrón deja que el LLM invente el signo de la
  tendencia o un precio, lo rompiste. (Aplica con fuerza a la Pista 1: una tool
  puede *informar* la severidad, pero la serie sigue dando el signo.)
- **Cuida el costo de LLM.** Los lazos (Pista 2) y el fan-out doble (Pista 3)
  multiplican llamadas. Estima antes de correr; itera sobre un nodo aislado.
- **Documenta lo que no funcionó.** Un experimento con resultado negativo bien
  argumentado es un entregable válido (el repo ya tiene dos lecciones de
  overfitting documentadas; sigue esa cultura).

---

## 4. Diagnóstico común (hazlo antes de elegir pista)

### 4.1 Lee los prompts y nombra las técnicas que YA usan

Construye una tabla: para cada uno de los cuatro prompts, ¿qué técnicas de
prompting ya emplea? Pistas de lo que hay (no es exhaustivo — busca tú):

- **Role prompting** ("You are the independent chair of a Colombian FX committee…").
- **Chain-of-thought estructurado** ("Rules of reasoning — in this order: 1…2…").
- **Prompting adversarial / self-critique** (el `devils_advocate` obligatorio).
- **Salida estructurada por contrato** (`with_structured_output` + Pydantic).
- **Reglas de calibración** (caps de confianza, cuándo `neutral`).
- **Constraints de idioma y de "no inventes"**.

> Si no puedes nombrar la técnica que ya existe, no vas a saber qué le falta.

### 4.2 Lee la topología y responde por escrito

- ¿Por qué `fetch_market` y `load_memory` **no tienen arista de salida** hacia el
  adjudicador? (Pista: `defer=True` y el conteo de *triggers*.)
- ¿Por qué el fan-out usa `Send` y no un `for` dentro de un nodo?
- ¿Por qué los reducers de `worker_analyses` y `errors` usan `operator.add`?
- ¿Qué nodos llaman a un LLM y cuáles son **deterministas a propósito**?
  (`compute_ts_signal`, `aggregate_signals` lo son — entiende por qué.)

> **Trampa documentada en este repo:** `defer=True` NO retiene cuando lo único
> pendiente son tasks de `Send`, y un nodo deferred con 2+ triggers entrantes se
> ejecuta **una vez por trigger**. La solución usada aquí es *un solo trigger
> entrante* al adjudicador. Si tu patrón agrega una arista hacia `adjudicate`,
> escribe un test que cuente cuántas veces se ejecuta. (Crítico para la Pista 2.)

---

## 5. Las tres pistas — elige UNA

Cada pista tiene la misma estructura: **dónde vive hoy → qué construir → la técnica
de prompting que lo sostiene → trampas de LangGraph → cómo lo mides**.

### Pista 1 — ReAct: el analista que razona y actúa

**Dónde vive hoy.** El `topic_worker` (`nodes.py`) y el `NewsAnalyzer` reciben el
texto del artículo y clasifican de **un solo disparo**. La `severity` (high/medium/
low) la *adivina* el modelo: no consulta ningún dato para confirmarla.

**Qué construir.** Dale al analista **herramientas deterministas** y deja que
razone+actúe en un lazo ReAct:
- `get_series(ticker, ventana)` sobre Brent / DXY / equity (ya existen en
  `market_fetcher.py`), `get_trm(rango)` (de `fx_fetcher.py`), o
  `prior_coverage(titular)` (¿esto ya se reportó? penaliza reciclado).
- El analista decide cuándo llamarlas: *"el titular dice shock petrolero → llamo
  `get_series('brent') → observo +4% → canal terms_of_trade, severidad alta"*.

**Técnica de prompting.** ReAct (ciclo Thought → Action → Observation), buenas
*descripciones de tools*, y opcionalmente un *few-shot* de una trayectoria de
herramienta bien hecha. En LangGraph puedes armar el lazo a mano o con un
sub-grafo ReAct prebuilt; el worker pasa de ser un nodo a ser un mini-agente.

**Por qué importa (intención).** Ataca de frente la **alucinación (#3)**: la
severidad deja de ser opinión y se ancla en un número real. De rebote mejora el
**acierto (#1)**.

**Trampas de LangGraph.** El lazo de tools **multiplica llamadas** → pon
`recursion_limit` y un tope de pasos explícito. Envuelve los fetchers como tools
sin romper el fail-soft (una tool que falla no debe tumbar al worker). Respeta el
payload del `Send`. Y la regla dura: la tool *informa*, **no** fija el signo de la
tendencia (eso sigue siendo de la serie).

**Cómo lo mides.** Sobre un set congelado de artículos: ¿la `severity` con ReAct
correlaciona mejor con el movimiento realizado de la TRM que la one-shot? ¿En
cuántos casos la tool *cambió* la clasificación? Costo: tokens/llamadas extra por
artículo vs. la ganancia de señal.

---

### Pista 2 — Evaluator-optimizer: el adjudicador que se corrige

**Dónde vive hoy.** `adjudicate` (`nodes.py`) es **un solo disparo**. Las
incoherencias (dirección que no coincide con `dominant_signal`, confianza fuera de
banda) las arregla **código a posteriori** (`_build_directional_call` / "fix
structural contradictions"). Es un parche, no un razonamiento.

**Qué construir.** El lazo completo del patrón: **generador → evaluador →
(regenera si reprueba)**. Un nodo `critique` (otro prompt, idealmente otro tier de
modelo) puntúa el `AdjudicatorVerdict` contra una **rúbrica explícita**: ¿la
dirección coincide con la señal dominante? ¿el abogado del diablo es de verdad
fuerte o de relleno? ¿la confianza está calibrada a la divergencia? Si reprueba,
devuelve la crítica al generador y se regenera, hasta N vueltas.

**Técnica de prompting.** Reflexión / critique-then-revise, con **separación de
roles** (generador vs evaluador) y una rúbrica de evaluación escrita, no implícita.

**Por qué importa (intención).** Mueve la lógica frágil del código a un
razonamiento **auditable (#4)** y mejora la **calidad de la abstención (#2)**.

**Trampas de LangGraph.** Estás metiendo un **ciclo** y tocando el nodo con
`defer=True`: relee la trampa de §4.2 y añade un test que cuente ejecuciones.
Lleva `revision_count` en el estado y pon `recursion_limit` para no hacer bucle
infinito. **Táctica complementaria (Pydantic):** parte de esas garantías pueden
migrar del prompt al **contrato** en `contracts.py` — `Field(ge=…, le=…)`,
`Literal`, `@model_validator` — para que un veredicto incoherente **no pueda
construirse**. Mide cuánto código de "fix contradictions" puedes borrar; cuidado
con que validadores muy estrictos disparen reintentos de `with_structured_output`
(= más costo).

**Cómo lo mides.** Número de contradicciones que el código *tenía* que corregir
(debe tender a 0 con el lazo); reparto de `dominant_signal` y tasa de `neutral`;
¿cuántas vueltas necesita en promedio? Costo de las llamadas extra del crítico.

---

### Pista 3 — Orchestrator-workers: el orquestador que planea

**Dónde vive hoy.** `orchestrate` → `topic_worker × N` vía `Send`
(`fan_out_clusters` en `nodes.py`). El orquestador solo **parte** los artículos en
clusters; **todos los workers usan el mismo prompt**. No hay especialización.

**Qué construir** (elige una variante):
- **Personas por cluster:** el orquestador decide la *persona* del worker según el
  tipo de cluster — un cluster de política monetaria recibe analista de banca
  central; uno de petróleo, analista de commodities; uno de riesgo fiscal, analista
  de crédito soberano. El payload del `Send` lleva la persona/prompt.
- **Debate alcista vs bajista:** por cada cluster, lanza DOS workers con system
  prompts opuestos (uno construye el caso COP-fuerte, otro el COP-débil) y el
  adjudicador concilia el debate.

**Técnica de prompting.** Role/persona prompting especializado y *templating*
dinámico del prompt; en la variante debate, prompting adversarial estructurado.

**Por qué importa (intención).** Mejora la **calidad de la clasificación (#1)** —
un especialista nombra mejor el canal de transmisión — y la **auditabilidad (#4)**.

**Trampas de LangGraph.** Sigue siendo `Send` dinámico: respeta los reducers
`operator.add` y el tope `MAX_TOPIC_WORKERS`. El `aggregate_signals` determinista
debe seguir cuadrando aunque ahora lleguen análisis de personas distintas. El
**debate duplica el costo** — esa es justo la pregunta a responder con datos.

**Cómo lo mides.** Sobre clusters congelados: ¿la clasificación del especialista
coincide mejor con una etiqueta humana que la del worker genérico? En la variante
debate: ¿sube el acierto/la calidad de la abstención, o solo subió el costo?

---

## 6. Infraestructura de medición compartida (opcional, alto valor)

Las tres pistas chocan con lo mismo: §2.1 dice que **no puedes medir el acierto
real del LLM** con el repo tal cual. Si quieres ir más allá de las métricas-proxy,
el entregable de mayor palanca es un **harness de replay/backfill**:

> Construye un modo que reproduzca el pipeline sobre N fechas históricas: para
> cada fecha, congela los artículos y la serie de ese día, corre el grafo (LLM
> real o stubbeado) y vuelca el `DirectionalCall` a `predictions.db` con su
> `run_date` correcto; luego `evaluate_pending` lo califica contra la TRM
> realizada. Esto **desbloquea medir de verdad** el efecto de tu patrón sobre el
> acierto direccional. Pistas: captura/almacena las entradas por fecha (el GOLD
> store no sirve, guarda salidas); empieza con 5–10 fechas por costo; respeta los
> contratos de estado al inyectar datos congelados.

No es obligatorio construirlo, pero si lo haces, conviértelo en parte central de
tu medición — y reconócelo como el habilitador que es.

---

## 7. Reglas duras al tocar el grafo

- **No reintroduzcas `os.chdir()` ni `ROOT = Path.cwd()`.** Las rutas salen de
  `cop_fx.paths` (derivado de `__file__`). Está marcado como mala práctica.
- **Serialización del estado:** el estado lleva DataFrames (`fx_df`, `ensemble_df`)
  y objetos `ForecastResult`. Por eso el checkpointer usa
  `JsonPlusSerializer(pickle_fallback=True)`. Si agregas campos al estado,
  verifica que sean serializables.
- **Todo nodo nuevo necesita un test.** Mira `tests/unit/test_graph_nodes.py`
  como plantilla. Si tu patrón toca el `defer`/`Send` o introduce un ciclo, añade
  un test de regresión que cuente ejecuciones.
- **Fail-soft:** los nodos de datos no deben tumbar el pipeline; acumulan en
  `errors` (reducer `operator.add`) y siguen. Mantén ese contrato — también en las
  tools de la Pista 1.

---

## 8. Entregables

Entrega una rama y un documento corto (`docs/experimentos/<tu-nombre>.md`) con:

1. **Pista elegida e hipótesis** — qué patrón y qué creías que iba a pasar y por
   qué (escrito *antes* de tocar código).
2. **Línea base** — la métrica de tu pista medida sobre el sistema sin tu cambio,
   con el comando/procedimiento exacto que usaste.
3. **El patrón** — diff acotado + nombra explícitamente la técnica de prompting y
   el cableado de LangGraph que aplicaste.
4. **Resultado** — la misma métrica después. Tablas, no adjetivos. Incluye el costo.
5. **Veredicto honesto** — ¿se queda o se descarta? Si no funcionó, por qué crees
   que no.
6. **Tests** verdes (`uv run pytest`).

### Rúbrica

| Criterio | Peso |
|---|---|
| Diagnóstico correcto de lo que ya existe (§4) | 20% |
| Patrón implementado correctamente y completo (no a medias) | 25% |
| Medición rigurosa contra la intención (§2), no por intuición | 30% |
| Respeto a las reglas duras (serialización, `defer`/ciclos, fail-soft, rutas) | 15% |
| Honestidad intelectual (documentar lo que no funcionó) | 10% |

---

## Apéndice — Arranque rápido

```bash
# Entorno (gestor: uv, NO pip)
uv sync

# Correr el pipeline completo (cuidado con el costo de LLM)
uv run cop-fx run

# Correr con human-in-the-loop (pausa en human_review para inspeccionar el veredicto)
uv run cop-fx run --review

# Backtest direccional de la PATA DE SERIE (arima/momentum/always_up).
# OJO: NO ejecuta el LLM — no mide tu patrón agéntico (ver §2.1).
uv run python -c "from cop_fx.data.fx_fetcher import FXFetcher; from cop_fx.tracking.backtest import directional_backtest; _, summary = directional_backtest(FXFetcher().fetch()); print(summary)"

# Tests
uv run pytest

# Dashboard (para inspeccionar veredicto, noticias GOLD, backtest)
uv run streamlit run dashboard/app.py
```

> Configuración operativa en `config.yaml` (modelo, caps de noticias, bandas).
> `.env` es **solo secretos**. Precedencia: env > .env > yaml > defaults.

**Primer paso recomendado (diagnóstico medible):** antes de elegir pista, arma un
set de ~20 titulares etiquetados a mano (material / no material), corre
`check_materiality` sobre ellos stubbeando el resto, y calcula precisión/recall del
gate actual. Te entrena el flujo completo —congelar entrada, stubbear LLM, medir—
sin depender del backtest (que solo mide la serie) ni de `predictions.db` (que se
llena ~1/día). De ahí, elige tu patrón.
