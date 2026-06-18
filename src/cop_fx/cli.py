"""CLI entry point: `cop-fx run [--publish] [--review]` y `cop-fx resume`."""

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
        "--publish",
        action="store_true",
        default=False,
        help="Post the generated tweet (requires Twitter credentials)",
    )
    run_parser.add_argument(
        "--review",
        action="store_true",
        default=False,
        help="Human-in-the-loop: pausa en human_review antes de publicar (interrupt)",
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
        "resume", help="Reanuda una corrida HITL pausada y aprueba/rechaza la publicación"
    )
    resume_parser.add_argument("--thread", required=True, help="thread_id de la corrida pausada")
    decision = resume_parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true", help="Aprobar y publicar")
    decision.add_argument("--reject", action="store_true", help="Rechazar (no publicar)")
    resume_parser.add_argument(
        "--tweet-text", default=None, help="Texto editado opcional para el tweet aprobado"
    )
    return parser


def _cmd_run(args: argparse.Namespace, log: structlog.BoundLogger) -> None:
    log.info(
        "Starting COP/USD pipeline",
        run_date=args.date,
        publish=args.publish,
        review=args.review,
    )
    state = run_pipeline(
        run_date=args.date,
        publish_enabled=args.publish,
        hitl=args.review,
        thread_id=args.thread,
    )

    payload = _interrupt_payload(dict(state))
    if payload is not None:
        thread = args.thread or args.date
        log.info("Pipeline paused for human review", thread_id=thread)
        print("\n⏸️  HITL — revisión humana requerida antes de publicar:\n")
        print(
            f"   Dirección: {payload.get('direction')}  ·  "
            f"confianza: {payload.get('confidence')}"
        )
        print(f"   Tweet propuesto:\n   {payload.get('tweet_text')}\n")
        print("   Aprobar:  cop-fx resume --thread", thread, "--approve")
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
        tweet_id=state.get("tweet_id"),
    )
    print(state.get("report_markdown", ""))


def _cmd_resume(args: argparse.Namespace, log: structlog.BoundLogger) -> None:
    approved = bool(args.approve)
    log.info("Resuming HITL pipeline", thread_id=args.thread, approved=approved)
    state = resume_pipeline(
        thread_id=args.thread,
        approved=approved,
        tweet_text=args.tweet_text,
    )
    errors = state.get("errors", [])
    if errors:
        log.error("Resumed pipeline completed with errors", errors=errors)
        sys.exit(1)
    log.info(
        "Resume complete",
        approved=approved,
        tweet_id=state.get("tweet_id"),
    )
    if approved and not state.get("tweet_id"):
        print("Aprobado, pero no se publicó (revisa publish_enabled/twitter_enabled).")


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
