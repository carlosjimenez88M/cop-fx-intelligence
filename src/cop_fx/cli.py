"""CLI entry point: `cop-fx run [--review]` y `cop-fx resume`."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from typing import Any

import structlog

from cop_fx.agents.graph import resume_pipeline, run_pipeline
from cop_fx.config.settings import get_settings


def _configure_logging(level: str) -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logging.basicConfig(level=level, stream=sys.stderr)


def _interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    """Extrae el payload del `interrupt` del estado pausado (HITL)."""
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"value": value}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cop-fx",
        description="COP/USD FX Intelligence Pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Execute the daily analysis pipeline")
    run_parser.add_argument(
        "--review",
        action="store_true",
        default=False,
        help="Human-in-the-loop: pausa en human_review para revisar el veredicto (interrupt)",
    )
    run_parser.add_argument(
        "--thread",
        default=None,
        help="thread_id del checkpointer (por defecto la fecha de la corrida)",
    )
    run_parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Override run date (ISO format, default: today)",
    )

    resume_parser = sub.add_parser(
        "resume", help="Reanuda una corrida HITL pausada y acepta/rechaza el veredicto"
    )
    resume_parser.add_argument("--thread", required=True, help="thread_id de la corrida pausada")
    decision = resume_parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true", help="Aceptar el veredicto")
    decision.add_argument("--reject", action="store_true", help="Rechazar el veredicto")
    return parser


def _cmd_run(args: argparse.Namespace, log: structlog.BoundLogger) -> None:
    log.info(
        "Starting COP/USD pipeline",
        run_date=args.date,
        review=args.review,
    )
    state = run_pipeline(
        run_date=args.date,
        hitl=args.review,
        thread_id=args.thread,
    )

    payload = _interrupt_payload(dict(state))
    if payload is not None:
        thread = args.thread or args.date
        log.info("Pipeline paused for human review", thread_id=thread)
        print("\n⏸️  HITL — revisión humana del veredicto:\n")
        print(
            f"   Dirección: {payload.get('direction')}  ·  confianza: {payload.get('confidence')}"
        )
        print(f"   Reporte: {payload.get('report_path')}\n")
        print("   Aceptar:  cop-fx resume --thread", thread, "--approve")
        print("   Rechazar: cop-fx resume --thread", thread, "--reject")
        return

    errors = state.get("errors", [])
    if errors:
        log.error("Pipeline completed with errors", errors=errors)
        sys.exit(1)

    log.info(
        "Pipeline complete",
        report=state.get("report_path"),
        latest_rate=state.get("latest_rate"),
    )
    print(state.get("report_markdown", ""))


def _cmd_resume(args: argparse.Namespace, log: structlog.BoundLogger) -> None:
    approved = bool(args.approve)
    log.info("Resuming HITL pipeline", thread_id=args.thread, approved=approved)
    state = resume_pipeline(
        thread_id=args.thread,
        approved=approved,
    )
    errors = state.get("errors", [])
    if errors:
        log.error("Resumed pipeline completed with errors", errors=errors)
        sys.exit(1)
    log.info("Resume complete", approved=approved)
    print(f"Veredicto {'aceptado' if approved else 'rechazado'}.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    settings = get_settings()
    _configure_logging(settings.log_level)
    log = structlog.get_logger()

    if args.command == "run":
        _cmd_run(args, log)
    elif args.command == "resume":
        _cmd_resume(args, log)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
