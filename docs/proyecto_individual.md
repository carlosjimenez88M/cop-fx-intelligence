# Proyecto individual — Razonamiento multiagéntico para la dirección del USD/COP

> **Modalidad:** individual y auto-dirigido. Este documento te da el mapa, los
> retos y las pistas; **no** te da las soluciones. El objetivo no es "que corra",
> sino que justifiques cada decisión con evidencia.
>
> **Pre-requisitos de lectura:** `README.md` (qué hace el sistema) y
> `docs/arquitectura.md` (por qué está diseñado así). Este proyecto asume que ya
> los leíste.

---

## 0. Cómo usar este documento

El proyecto tiene dos frentes que se refuerzan entre sí:

- **Parte A — Prompting:** cómo le hablas a los modelos para que el sistema cumpla
  mejor su intención.
- **Parte B — Arquitectura LangGraph:** cómo cableas los nodos para que el
  razonamiento sea más robusto, barato o auditable.

No tienes que hacer todo. Tienes que **elegir un hilo, llevarlo hasta el final y
demostrar con números que mejoraste algo**. Un cambio pequeño bien medido vale
más que diez cambios sin evidencia.

Trabaja en este orden:

1. **Diagnostica** (Parte A.0 y B.0): lee el código y nombra lo que ya existe.
2. **Elige** uno o dos retos (de A y/o B) y escribe tu hipótesis *antes* de tocar
   código: "creo que X mejorará la métrica Y porque Z".
3. **Mide la línea base** con la infraestructura que ya está en el repo.
4. **Implementa** tu cambio en una rama.
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
                                  generate_report → record_prediction → human_review → (fin del ejercicio)
```

> **Fuera de alcance:** la publicación en redes (el nodo `publish` / Twitter) **no
> es parte de este proyecto**. No necesitas credenciales de X/Twitter ni vas a
> publicar nada. Tu producto final es el `DirectionalCall` y el reporte, no un
> tweet. Si un reto te lleva al final del grafo, detente en el reporte/veredicto.

**Dónde vive cada cosa (puntos de entrada al código):**

| Pieza | Archivo |
|---|---|
| Cableado del grafo | `src/cop_fx/agents/graph.py` |
| Nodos (toda la lógica + prompts) | `src/cop_fx/agents/nodes.py` |
| Estado compartido (`PipelineState`) | `src/cop_fx/agents/state.py` |
| Prompt del analista de noticias | `src/cop_fx/analysis/news_analyzer.py` |
| Contratos Pydantic (salida estructurada) | `src/cop_fx/contracts.py` |
| Memoria / track-record | `src/cop_fx/agents/memory.py` |
| Checkpointer + Store | `src/cop_fx/agents/persistence.py` |
| Registro y auto-evaluación de predicciones | `src/cop_fx/tracking/predictions.py` |
| Backtesting direccional | `src/cop_fx/tracking/backtest.py` |

Los prompts ya están en el código como constantes: `_MATERIALITY_PROMPT`,
`_TOP_STORY_PROMPT`, `_ADJUDICATOR_PROMPT` (en `nodes.py`) y `_PROMPT_TEMPLATE`
(en `news_analyzer.py`). **Empieza leyéndolos enteros.**

---

## 2. La intención del proyecto (tu vara de medir)

Todo cambio se juzga contra estos objetivos. Si tu cambio no mueve ninguno, no
sirve aunque "se vea más elegante":

1. **Acierto direccional** — ¿el sistema acierta `up`/`down` más seguido que un
   baseline tonto (momentum, always_up)? Mídelo con `directional_backtest`.
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
  LLM.** Si cambias un prompt del adjudicador o del analista, este número **no se
  mueve** — no porque tu cambio sea malo, sino porque el backtest ni lo toca. Úsalo
  como *baseline de la señal de serie*, no como juez de tus prompts.
- **El acierto direccional del LLM vive en `predictions.db`** (`PredictionStore`),
  que `record_prediction` llena a razón de **~1 fila por corrida** y auto-califica
  con `evaluate_pending` al vencer el horizonte. **No existe un harness de
  backfill/replay**: no puedes recrear 60 días de decisiones del LLM ejecutando un
  comando. Con el repo tal cual, **no obtendrás una muestra estadísticamente útil
  del acierto del LLM en el plazo de un curso.**

Tienes dos salidas honestas, y debes declarar cuál tomaste:

  **(a) Métricas-proxy offline** (baratas, viables ya): para cambios de prompt,
  mide cosas que sí puedes calcular sobre entradas fijas —
  - precisión/recall del gate sobre un set de titulares etiquetados a mano;
  - varianza de `severity` al re-correr el analista sobre los **mismos** artículos;
  - tasa de `neutral` y reparto de `dominant_signal`;
  - número de contradicciones que hoy **corrige el código** en `adjudicate`
    (si tu prompt es mejor, el código debería corregir menos);
  - tokens/llamadas por decisión.

  **(b) Construir el harness de replay** (ver reto B.2.5) — el entregable más
  valioso del proyecto, porque desbloquea medir TODO lo demás de verdad.

> **Iteración offline sin quemar API keys:** stubbea el LLM con
> `unittest.mock.patch` como en `tests/unit/test_graph_nodes.py`, y aliméntalo con
> artículos guardados como *fixtures* que capturaste tú. Ojo: el GOLD store
> (`data/cnn_articles.db`) guarda la **salida** de la clasificación (`summary` +
> tópico/severidad), **no** el cuerpo crudo que leyó el analista — para congelar
> *entradas* tienes que guardarlas tú mismo. Verifícalo antes de confiar en él.

---

## 3. Reglas del juego

- **Una rama por experimento.** Nombra claro: `exp/few-shot-adjudicator`,
  `exp/reflection-loop`. No mezcles prompting y arquitectura en el mismo commit.
- **Sé crítico con el data flow antes de declarar algo terminado.** Verifica
  *qué texto* le llega realmente al LLM (no asumas que el cuerpo del artículo
  llegó completo). Mira el log y la salida estructurada, no solo el reporte final.
- **El LLM produce dirección + razonamiento; la serie produce el signo.** No
  difumines esta frontera: si tu cambio deja que el LLM invente el signo de la
  tendencia o un precio, lo rompiste.
- **Cuida el costo de LLM.** Antes de correr el pipeline entero, estima cuántas
  llamadas hará. Prefiere iterar sobre un nodo aislado con datos guardados.
- **Documenta lo que no funcionó.** Un experimento con resultado negativo bien
  argumentado es un entregable válido (el repo ya tiene dos lecciones de
  overfitting documentadas; sigue esa cultura).

---

## Parte A — Estrategias de prompting

### A.0 Diagnóstico (hazlo antes de escribir un solo prompt)

Lee los cuatro prompts del sistema y construye una tabla: para cada prompt, ¿qué
**técnicas de prompting** ya usa? Vas a encontrar varias. Identifícalas por
nombre. Pistas de lo que hay (no es lista exhaustiva — busca tú):

- **Role prompting** ("You are the independent chair of a Colombian FX committee…").
- **Chain-of-thought estructurado** ("Rules of reasoning — in this order: 1…2…").
- **Prompting adversarial / self-critique** (el `devils_advocate` obligatorio).
- **Salida estructurada por contrato** (`with_structured_output` + Pydantic) — el
  LLM no puede salirse del esquema.
- **Reglas de calibración** (caps de confianza, cuándo `neutral`).
- **Constraints de idioma y de "no inventes"** ("Do not make claims not in the
  provided signals").

> Si no puedes nombrar la técnica que ya existe, no vas a saber qué le falta.

### A.1 Catálogo de técnicas para aplicar

Elige y aplica **al menos dos** de estas que el sistema todavía NO usa (o usa de
forma débil). Para cada una: aplícala, mídela, decide si se queda.

| Técnica | Idea en una línea | Dónde podría brillar aquí |
|---|---|---|
| **Few-shot / exemplars** | Mostrar 2–3 ejemplos resueltos de entrada→salida | Calibrar `severity` del analista; mostrar un caso `neutral` bien hecho al adjudicador |
| **Contrastive / negative examples** | Mostrar un ejemplo *malo* y por qué lo es | Enseñar al gate a NO marcar como material el ruido global |
| **Self-consistency** | Muestrear N veces y votar la dirección modal (necesita **temperatura > 0** — choca con el `temperature=0.0` actual; súbela solo en ese nodo) | Adjudicador en días ambiguos: ¿converge o se contradice? |
| **Reflexión / critique-then-revise** | El modelo evalúa su propia salida y la corrige | Segundo paso sobre el `DirectionalCall` antes de cerrar el veredicto |
| **Prompt chaining / descomposición** | Partir un prompt grande en pasos más simples | Separar "clasificar relación de señales" de "elegir dirección" |
| **Rúbrica explícita / scoring** | Pedir que puntúe contra criterios antes de decidir | Que el editor puntúe cada candidata a noticia del día |
| **ReAct (razona+actúa)** | Intercalar razonamiento con consultas a herramientas | Si agregas una tool de datos (ver Parte B) |
| **Output forzado de incertidumbre** | Pedir confianza calibrada + qué evidencia la cambiaría | Reforzar la abstención como capacidad |

### A.2 Retos concretos de prompting

Pistas, no recetas. Elige uno como tu hilo principal:

1. **El gate barato sobre-filtra o sub-filtra.** `_MATERIALITY_PROMPT` decide con
   solo titulares. Construye un set de ~20 titulares etiquetados a mano
   (material / no material) y mide precisión/recall del gate *actual*. Luego
   intenta mejorarlo con few-shot + ejemplos contrastivos. ¿Subió el recall sin
   inundar de falsos positivos (= más costo aguas abajo)?

2. **La calibración de `severity` del analista es subjetiva.** En
   `news_analyzer.py`, las reglas de `high/medium/low` son texto. Dale ejemplos
   anclados (anchored examples) de cada nivel. Mide: ¿bajó la varianza de
   `severity` entre corridas sobre los mismos artículos?

3. **El adjudicador a veces debería abstenerse y no lo hace (o al revés).**
   Diseña un prompt que pida *primero* listar qué evidencia faltaría para estar
   seguro, y solo entonces decidir. Compara la tasa de `neutral` y el acierto
   direccional condicionado a no-neutral.

4. **El editor (`pick_top_story`) elige por intuición.** Conviértelo en una
   rúbrica: que puntúe cada candidata en frescura, especificidad y relevancia FX
   antes de elegir el índice. ¿Coincide mejor con lo que un humano elegiría?

### A.3 Cómo medir un prompt (no por intuición)

- **Congela la entrada.** Captura un conjunto de artículos / señales reales como
  *fixtures* propios (recuerda: el GOLD store guarda la *salida*, no el cuerpo de
  entrada — §2.1) y corre el nodo aislado sobre *los mismos datos* antes y después,
  stubbeando el LLM con `patch` como en `tests/unit/test_graph_nodes.py`. Comparar
  sobre entradas distintas no prueba nada.
- **Define la métrica antes de cambiar el prompt** (ver §2).
- **Repite la corrida** (temperatura 0 ayuda pero no garantiza determinismo):
  reporta varianza, no un solo número.
- **Mira la salida estructurada cruda**, no solo el reporte: ¿el `consistency_notes`
  realmente nombra la señal dominante? ¿el `devils_advocate` es de verdad el
  contra-argumento más fuerte o es de relleno?

---

## Parte B — Arquitectura LangGraph

### B.0 Diagnóstico de la topología actual

Antes de añadir nodos, entiende por qué el grafo está así. Responde por escrito:

- ¿Por qué `fetch_market` y `load_memory` **no tienen arista de salida** hacia el
  adjudicador? (Pista: tiene que ver con `defer=True` y el conteo de *triggers*.)
- ¿Por qué el fan-out usa `Send` y no un `for` dentro de un nodo? (Pista: la
  cantidad de workers cambia cada día.)
- ¿Por qué los reducers de `worker_analyses` y `errors` usan `operator.add`?
- ¿Qué nodos llaman a un LLM y cuáles son **deterministas a propósito**?
  (`compute_ts_signal`, `aggregate_signals` lo son — entiende por qué.)

> **Trampa documentada en este repo:** `defer=True` NO retiene cuando lo único
> pendiente son tasks de `Send`, y un nodo deferred con 2+ triggers entrantes se
> ejecuta **una vez por trigger**. La solución usada aquí es *un solo trigger
> entrante* al adjudicador. Si tu cambio agrega una arista hacia `adjudicate`,
> escribe un test que cuente cuántas veces se ejecuta.

### B.1 Catálogo de mejoras arquitectónicas

Elige **una** como hilo principal:

| Patrón LangGraph | Qué te da | Pista de dónde aplicarlo |
|---|---|---|
| **Reflection loop** (ciclo con arista de regreso) | El adjudicador critica y reintenta su veredicto | Nodo `critique` entre `adjudicate` y `generate_report`; arista condicional de regreso si no pasa un check |
| **Retry / fallback con arista condicional** | Robustez ante fallos de LLM o de datos | Hoy el fallback es código dentro del nodo; vuélvelo explícito en el grafo |
| **Subgrafo** | Encapsular la rama de noticias como grafo reusable | Compila `fetch_news…aggregate_signals` como subgrafo con su propio estado |
| **Evaluador-como-nodo** | Un nodo que califica la salida y enruta | Tras `record_prediction`, un nodo que decide si re-correr con más fuentes |
| **Streaming de progreso** | UX y debugging | Usa `stream_mode` para emitir avance por nodo al dashboard |
| **Multi-agente en debate** | Dos analistas con sesgo opuesto (alcista/bajista) | Reemplaza/duplica `topic_worker` y deja que el adjudicador concilie el debate |
| **Tool calling (ReAct)** | El analista consulta una serie numérica bajo demanda | Expón Brent/DXY como tool y deja que el worker la pida cuando la noticia lo amerite |
| **Caché de nodos** | No re-analizar artículos ya vistos | Clave por hash de contenido contra el GOLD store |

### B.2 Retos concretos de arquitectura

1. **Ciclo de reflexión sobre el veredicto.** Añade un nodo `critique_verdict`
   que reciba el `DirectionalCall` y verifique las reglas de coherencia *con otro
   modelo o prompt*. Si falla, arista condicional de regreso a `adjudicate` (con
   un contador para no hacer bucle infinito — pista: lleva `revision_count` en el
   estado). Mide: ¿bajaron las contradicciones que hoy corrige el código a mano?

2. **La rama de noticias como subgrafo.** Extrae `fetch_news → check_materiality
   → orchestrate → topic_worker → aggregate_signals` a un subgrafo compilado.
   Beneficio a demostrar: lo puedes testear y correr aislado. Cuidado con el
   estado: define el contrato de entrada/salida del subgrafo.

3. **Debate alcista vs bajista.** Por cada cluster, lanza DOS workers con system
   prompts opuestos (uno busca el caso COP-fuerte, otro el COP-débil). El
   adjudicador concilia. Usa `Send` para el fan-out doble. Mide si el acierto
   sube o si solo subió el costo.

4. **Tool calling para series bajo demanda.** Hoy `fetch_market` trae todo
   siempre. Conviértelo en una herramienta que el analista invoque solo cuando
   una noticia toca términos de intercambio (petróleo) o el lado dólar. Mide la
   reducción de llamadas/tokens vs la pérdida (o no) de señal.

5. **★ Harness de replay/backfill (reto estrella).** Hoy no puedes medir el
   acierto del LLM con muestra real (§2.1). Construye un modo que reproduzca el
   pipeline sobre N fechas históricas: para cada fecha, congela los artículos y
   la serie de ese día, corre el grafo (con LLM real o stubbeado) y vuelca el
   `DirectionalCall` a `predictions.db` con su `run_date` correcto; luego
   `evaluate_pending` lo califica contra la TRM realizada. **Esto desbloquea medir
   de verdad todo lo demás del proyecto** — por eso es el reto de mayor valor.
   Pistas: necesitas capturar/almacenar las entradas por fecha (el GOLD store no
   sirve, guarda salidas); cuida el costo de LLM (empieza con 5–10 fechas);
   respeta los contratos de estado al inyectar datos congelados.

### B.2.bis — Pydantic: mueve las garantías del prompt al contrato

Tu objetivo es dominar LangGraph **y Pydantic**. Hoy varias garantías viven como
*texto* en los prompts ("confidence below 0.35 turns neutral", "direction MUST
match dominant_signal") y el código las re-corrige a mano en `adjudicate`. Reto:
**llévalas al contrato** en `src/cop_fx/contracts.py`.

- Usa tipos restringidos y `Field(ge=…, le=…)`, `enum`/`Literal`, y
  `@field_validator` / `@model_validator` para que un veredicto incoherente
  (p. ej. `dominant_signal=news` con `direction` que no coincide, o `confidence`
  fuera de rango) **no pueda construirse**, en vez de corregirse después.
- Mide: ¿cuánta lógica de "fix structural contradictions" del nodo `adjudicate`
  puedes borrar porque el contrato ya la garantiza? Menos código defensivo a mano
  = mejora real de robustez y auditabilidad.
- Cuidado: `with_structured_output` reintenta cuando el modelo viola el esquema —
  observa si tus validadores nuevos disparan más reintentos (= más costo) y
  balancéalo.

### B.3 Reglas duras al tocar el grafo

- **No reintroduzcas `os.chdir()` ni `ROOT = Path.cwd()`.** Las rutas salen de
  `cop_fx.paths` (derivado de `__file__`). Está marcado como mala práctica.
- **Serialización del estado:** el estado lleva DataFrames (`fx_df`, `ensemble_df`)
  y objetos `ForecastResult`. Por eso el checkpointer usa
  `JsonPlusSerializer(pickle_fallback=True)`. Si agregas campos al estado,
  verifica que sean serializables o el HITL/checkpoint se rompe.
- **Todo nodo nuevo necesita un test.** Mira `tests/unit/test_graph_nodes.py`
  como plantilla. Si tu cambio toca el `defer`/`Send`, añade un test de regresión
  que cuente ejecuciones.
- **Fail-soft:** los nodos de datos no deben tumbar el pipeline; acumulan en
  `errors` (reducer `operator.add`) y siguen. Mantén ese contrato.

---

## Parte C — Entregables

Entrega una rama y un documento corto (`docs/experimentos/<tu-nombre>.md`) con:

1. **Hipótesis** — qué creías que iba a pasar y por qué (escrita *antes* de tocar
   código).
2. **Línea base** — la métrica de §2 medida sobre el sistema sin tu cambio, con el
   comando exacto que usaste (`directional_backtest`, conteo de tokens, etc.).
3. **El cambio** — diff acotado + qué técnica de prompting o patrón LangGraph
   aplicaste, nombrado.
4. **Resultado** — la misma métrica después. Tablas, no adjetivos.
5. **Veredicto honesto** — ¿se queda o se descarta? Si no funcionó, por qué crees
   que no.
6. **Tests** verdes (`uv run pytest`).

### Rúbrica

| Criterio | Peso |
|---|---|
| Diagnóstico correcto de lo que ya existe (A.0 / B.0) | 20% |
| Técnica/patrón aplicado correctamente y con criterio | 25% |
| Medición rigurosa contra la intención (§2), no por intuición | 30% |
| Respeto a las reglas duras (serialización, `defer`, fail-soft, rutas) | 15% |
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
# OJO: NO ejecuta el LLM — no mide tus cambios de prompt (ver §2.1).
# directional_backtest(df, ...) necesita la serie TRM (columnas ds, y).
uv run python -c "from cop_fx.data.fx_fetcher import FXFetcher; from cop_fx.tracking.backtest import directional_backtest; _, summary = directional_backtest(FXFetcher().fetch()); print(summary)"

# Tests
uv run pytest

# Dashboard (para inspeccionar veredicto, noticias GOLD, backtest)
uv run streamlit run dashboard/app.py
```

> Configuración operativa en `config.yaml` (modelo, caps de noticias, bandas).
> `.env` es **solo secretos**. Precedencia: env > .env > yaml > defaults.

**Primer paso recomendado (calentamiento offline y medible):** arma un set de
~20 titulares etiquetados a mano (material / no material), corre `check_materiality`
sobre ellos stubbeando el resto, y calcula precisión/recall del gate actual. Eso te
da una línea base **real para un cambio de prompt** sin depender del backtest (que
solo mide la serie) ni de `predictions.db` (que se llena ~1/día). De ahí, elige tu
hilo.
