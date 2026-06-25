"""HTTP API del producto COP/USD — FastAPI modular, async y desacoplada.

Capas (de fuera hacia dentro):

    routers/      → solo HTTP: validan entrada, llaman a un servicio, devuelven schema.
    services/     → la lógica; envuelven el pipeline y los stores del dominio.
    schemas.py    → contratos de entrada/salida (Pydantic v2).
    dependencies  → inyección de dependencias (DIP): el router pide una interfaz.
    app.py        → fábrica `create_app()` que ensambla todo.

El objetivo de SOLID aquí es que los routers no sepan NADA de sqlite, pandas ni
LangGraph: solo conversan con protocolos (`services.protocols`). Cambiar el
almacenamiento o el motor de forecast no toca la capa HTTP.
"""

from __future__ import annotations

from cop_fx.api.app import create_app

__all__ = ["create_app"]
