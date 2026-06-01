"""CLI entry point: `cop-fx run [--publish]`."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import structlog

from cop_fx.agents.graph import run_pipeline
from cop_fx.config.settings import get_settings


def _configure_logging(level: str) -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logging.basicConfig(level=level, stream=sys.stderr)


def main() -> None:
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
        "--date",
        default=date.today().isoformat(),
        help="Override run date (ISO format, default: today)",
    )

    args = parser.parse_args()
    settings = get_settings()
    _configure_logging(settings.log_level)

    log = structlog.get_logger()

    if args.command == "run":
        log.info("Starting COP/USD pipeline", run_date=args.date, publish=args.publish)
        state = run_pipeline(run_date=args.date, publish_enabled=args.publish)

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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
