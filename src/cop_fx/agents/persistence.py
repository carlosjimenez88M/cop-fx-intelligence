"""Etapa 6 — persistencia del grafo: checkpointer + store de memoria.

LangGraph separa dos clases de memoria, y este módulo las expone como UNA
fábrica para que el resto del sistema no toque las clases concretas:

  - **Checkpointer** (memoria de corto plazo / por hilo): guarda el estado del
    grafo paso a paso bajo un ``thread_id``. Es lo que habilita el
    ``interrupt`` (HITL) — sin checkpointer no hay dónde "pausar" — y permite
    reanudar una corrida en otro proceso. `SqliteSaver` lo hace durable en
    disco; `InMemorySaver` vive solo en RAM (tests/dev).

  - **Store** (memoria de largo plazo / entre hilos): un almacén
    namespace→key→value que sobrevive a cualquier hilo. Aquí se usa para que
    el adjudicador recuerde su propio track-record entre corridas (ver
    `cop_fx.agents.memory`).

Precedencia de decisión: HITL ⇒ checkpointer durable obligatorio (hay que poder
reanudar en otra invocación). Sin HITL el grafo corre sin checkpointer, como
antes — cero cambios de comportamiento por defecto.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.memory import InMemoryStore

from cop_fx.paths import DATA_DIR

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.db"


def _serde() -> JsonPlusSerializer:
    """Serde con pickle de respaldo.

    El estado del grafo carga objetos que NO son JSON/msgpack: DataFrames de
    pandas (`fx_df`, `ensemble_df`) y `ForecastResult`. Sin un fallback, el
    checkpointer revienta al serializar el estado en el `interrupt`. El
    `pickle_fallback` los serializa por pickle y deja el resto en el formato
    eficiente por defecto.
    """
    return JsonPlusSerializer(pickle_fallback=True)


def get_checkpointer(*, persistent: bool = True) -> BaseCheckpointSaver[Any]:
    """Devuelve el checkpointer del grafo.

    `persistent=True` ⇒ `SqliteSaver` sobre `data/checkpoints.db`: el estado
    sobrevive al proceso, así una corrida pausada por `interrupt` se reanuda
    desde el CLI en otra invocación. `persistent=False` ⇒ `InMemorySaver`.
    """
    if not persistent:
        return InMemorySaver(serde=_serde())

    from langgraph.checkpoint.sqlite import SqliteSaver

    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph puede tocar la conexión desde el hilo
    # del executor; la conexión vive lo que dure el saver.
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    saver = SqliteSaver(conn, serde=_serde())
    saver.setup()
    return saver


def get_store() -> BaseStore:
    """Store de memoria a largo plazo (entre corridas).

    Hoy `InMemoryStore`: la fuente durable real es `predictions.db` (ver
    `cop_fx.agents.memory`), que el nodo de memoria hidrata al store al inicio
    de cada corrida. El store queda como la interfaz namespaced que el grafo
    consulta — y el punto de cambio si mañana se migra a un store nativo
    durable (p. ej. Postgres) sin tocar los nodos.
    """
    return InMemoryStore()
