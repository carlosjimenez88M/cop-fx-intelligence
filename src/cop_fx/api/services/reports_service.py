"""Servicio de reportes: lista y sirve los Markdown diarios de ``reports/``.

Los reportes se llaman ``report_YYYY-MM-DD.md``. El servicio expone el listado
(más reciente primero) y el contenido por fecha. Solo lectura de disco.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fastapi.concurrency import run_in_threadpool

from cop_fx.api.errors import ResourceNotFound
from cop_fx.api.schemas import ReportDetail, ReportList, ReportSummary
from cop_fx.paths import REPORTS_DIR

if TYPE_CHECKING:
    from pathlib import Path

_REPORT_RE = re.compile(r"report_(\d{4}-\d{2}-\d{2})\.md$")


class ReportsService:
    """Acceso de solo lectura a los reportes diarios en Markdown."""

    def __init__(self, reports_dir: Path = REPORTS_DIR) -> None:
        self._dir = reports_dir

    async def list_reports(self) -> ReportList:
        return await run_in_threadpool(self._list_sync)

    async def get_report(self, run_date: str) -> ReportDetail:
        return await run_in_threadpool(self._get_sync, run_date)

    # ------------------------------------------------------------------

    def _list_sync(self) -> ReportList:
        if not self._dir.exists():
            return ReportList(count=0, items=[])
        summaries: list[ReportSummary] = []
        for path in sorted(self._dir.glob("report_*.md"), reverse=True):
            match = _REPORT_RE.search(path.name)
            if match:
                summaries.append(ReportSummary(run_date=match.group(1), filename=path.name))
        return ReportList(count=len(summaries), items=summaries)

    def _get_sync(self, run_date: str) -> ReportDetail:
        path = self._dir / f"report_{run_date}.md"
        if not path.exists():
            raise ResourceNotFound(f"No existe reporte para la fecha {run_date}")
        return ReportDetail(
            run_date=run_date,
            filename=path.name,
            markdown=path.read_text(encoding="utf-8"),
        )
