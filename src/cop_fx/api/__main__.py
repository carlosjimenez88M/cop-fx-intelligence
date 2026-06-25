"""Punto de entrada del servidor: ``cop-fx-api`` o ``python -m cop_fx.api``.

Lee host/puerto/reload de variables de entorno (``COP_FX_API_HOST``, etc.) para
que el mismo binario sirva en local y en contenedor sin cambiar código.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "cop_fx.api.app:app",
        host=os.getenv("COP_FX_API_HOST", "0.0.0.0"),  # bind público (contenedor)
        port=int(os.getenv("COP_FX_API_PORT", "8000")),
        reload=os.getenv("COP_FX_API_RELOAD", "false").lower() == "true",
        log_level=os.getenv("COP_FX_API_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
