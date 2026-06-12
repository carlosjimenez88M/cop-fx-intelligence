"""Rutas absolutas del proyecto — derivadas del paquete, NUNCA del cwd.

Antes, notebooks y dashboard adivinaban la raíz con ``Path.cwd()`` y hacían
``os.chdir()`` — frágil y mala práctica. Este módulo resuelve la raíz desde
la ubicación del propio paquete (instalado en editable desde ``src/``), así
cualquier componente funciona sin importar desde dónde se ejecute.

Uso::

    from cop_fx.paths import PROJECT_ROOT, DATA_DIR, REPORTS_DIR
"""

from __future__ import annotations

from pathlib import Path

# src/cop_fx/paths.py → parents[2] == raíz del repo (instalación editable)
_candidate = Path(__file__).resolve().parents[2]
PROJECT_ROOT: Path = _candidate if (_candidate / "pyproject.toml").exists() else Path.cwd()

DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
ENV_FILE: Path = PROJECT_ROOT / ".env"
