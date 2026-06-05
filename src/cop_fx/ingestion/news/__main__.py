"""CLI del extractor de noticias.

Uso:
    python -m cop_fx.ingestion.news validate     # ¿qué feeds están vivos hoy?
    python -m cop_fx.ingestion.news run          # descarga + persiste bronze
    python -m cop_fx.ingestion.news run --no-persist
"""

from __future__ import annotations

import sys

from cop_fx.ingestion.news.extractor import run_news_ingestion, validate_sources
from cop_fx.logger import get_logger

logger = get_logger("cop_fx.ingestion.news")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "run"

    if cmd == "validate":
        report = validate_sources()
        print("\nEstado de las fuentes (artículos encontrados):")
        for name, n in report.items():
            mark = "OK " if n > 0 else "FAIL"
            print(f"  [{mark}] {n:>3}  {name}")
        dead = [n for n, c in report.items() if c == 0]
        if dead:
            print(f"\n{len(dead)} feed(s) caído(s): revisa la URL en sources.py")
        return 0

    if cmd == "run":
        persist = "--no-persist" not in argv
        articles = run_news_ingestion(persist=persist)
        print(f"\n{len(articles)} artículos extraídos. Primeros 5:")
        for a in articles[:5]:
            print(f"  [{a.country}] {a.published_at:%Y-%m-%d} · {a.source}")
            print(f"        {a.title[:90]}")
        return 0

    print(f"Comando desconocido: {cmd!r}. Usa 'validate' o 'run'.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
