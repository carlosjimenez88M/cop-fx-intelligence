"""Etapa 5 — la tabla `predictions`: cerrar el loop predicción → realidad.

Cada corrida diaria guarda su `DirectionalCall`; corridas posteriores evalúan
las predicciones pendientes contra la TRM real. Esto convierte el sistema en
algo MEDIBLE: hit-rate direccional, matriz de confusión y accuracy condicionada
a confianza (¿aciertas más cuando dices "alta confianza"? — eso valida la
calibración del adjudicador). Ver docs/arquitectura.md §7: no MAPE — dirección.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from cop_fx.logger import get_logger
from cop_fx.paths import DATA_DIR

if TYPE_CHECKING:
    import pandas as pd

    from cop_fx.contracts import DirectionalCall

logger = get_logger(__name__)

# Misma banda muerta que la señal de serie: |Δ%| menor = movimiento neutral.
EVAL_NEUTRAL_BAND_PCT = 0.10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    run_date          TEXT PRIMARY KEY,
    horizon_days      INTEGER NOT NULL,
    direction         TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    reconciliation    TEXT    NOT NULL,
    news_direction    TEXT,
    news_score        REAL,
    ts_direction      TEXT,
    ts_delta_pct      REAL,
    market_direction  TEXT,
    dominant_signal   TEXT,
    latest_rate       REAL,
    rationale         TEXT,
    devils_advocate   TEXT,
    created_at        TEXT,
    -- Se llenan al evaluar (cuando la TRM del horizonte ya existe):
    actual_rate       REAL,
    actual_change_pct REAL,
    actual_direction  TEXT,
    hit               INTEGER,   -- alias historico de final_hit
    final_hit         INTEGER,
    news_hit          INTEGER,
    ts_hit            INTEGER,
    market_hit        INTEGER,
    evaluated_at      TEXT
)
"""


class PredictionStore:
    """Persistencia SQLite de los DirectionalCall diarios + su evaluación."""

    # Columnas añadidas después del schema inicial — migración liviana.
    _EXTRA_COLS: ClassVar[dict[str, str]] = {
        "top_story_title": "TEXT",
        "top_story_display_title": "TEXT",
        "top_story_source": "TEXT",
        "top_story_why": "TEXT",
        "market_direction": "TEXT",
        "dominant_signal": "TEXT",
        "final_hit": "INTEGER",
        "news_hit": "INTEGER",
        "ts_hit": "INTEGER",
        "market_hit": "INTEGER",
    }

    def __init__(self, db_path: str | Path | None = None) -> None:
        # Default absoluto (cop_fx.paths): independiente del cwd.
        self._db_path = Path(db_path) if db_path is not None else DATA_DIR / "predictions.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(_SCHEMA)
            existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
            for col, sql_type in self._EXTRA_COLS.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {sql_type}")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------

    def save(
        self,
        call: DirectionalCall,
        *,
        run_date: str,
        latest_rate: float | None,
        top_story: dict[str, Any] | None = None,
        market_signal: dict[str, Any] | None = None,
    ) -> None:
        """Upsert por run_date: re-correr el día refresca la predicción."""
        top = top_story or {}
        market = market_signal or (call.market_signal.model_dump() if call.market_signal else {})
        market_direction = market.get("direction")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (run_date, horizon_days, direction, confidence, reconciliation,
                     news_direction, news_score, ts_direction, ts_delta_pct,
                     market_direction, dominant_signal, latest_rate,
                     rationale, devils_advocate, created_at,
                     top_story_title, top_story_display_title, top_story_source, top_story_why)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_date) DO UPDATE SET
                    horizon_days = excluded.horizon_days,
                    direction = excluded.direction,
                    confidence = excluded.confidence,
                    reconciliation = excluded.reconciliation,
                    news_direction = excluded.news_direction,
                    news_score = excluded.news_score,
                    ts_direction = excluded.ts_direction,
                    ts_delta_pct = excluded.ts_delta_pct,
                    market_direction = excluded.market_direction,
                    dominant_signal = excluded.dominant_signal,
                    latest_rate = excluded.latest_rate,
                    rationale = excluded.rationale,
                    devils_advocate = excluded.devils_advocate,
                    created_at = excluded.created_at,
                    top_story_title = excluded.top_story_title,
                    top_story_display_title = excluded.top_story_display_title,
                    top_story_source = excluded.top_story_source,
                    top_story_why = excluded.top_story_why,
                    actual_rate = NULL, actual_change_pct = NULL,
                    actual_direction = NULL, hit = NULL, final_hit = NULL,
                    news_hit = NULL, ts_hit = NULL, market_hit = NULL,
                    evaluated_at = NULL
                """,
                (
                    run_date,
                    call.horizon_days,
                    call.direction,
                    call.confidence,
                    call.reconciliation,
                    call.news_signal.direction,
                    call.news_signal.score,
                    call.ts_signal.direction,
                    call.ts_signal.yhat_delta_pct,
                    market_direction,
                    call.dominant_signal,
                    latest_rate,
                    call.rationale,
                    call.devils_advocate,
                    datetime.now(UTC).isoformat(),
                    top.get("title"),
                    top.get("display_title") or top.get("title"),
                    top.get("source"),
                    top.get("why_it_matters"),
                ),
            )
        logger.info("Prediction saved: %s → %s (%.2f)", run_date, call.direction, call.confidence)

    # ------------------------------------------------------------------
    # Evaluación
    # ------------------------------------------------------------------

    def evaluate_pending(self, fx_df: pd.DataFrame) -> int:
        """Evalúa predicciones cuyo horizonte ya tiene TRM real disponible.

        Compara la tasa en la primera fecha >= run_date + horizon contra la
        tasa al momento de predecir. hit = 1/0 solo para predicciones no
        neutrales — la abstención no se premia ni se castiga.
        """
        import pandas as pd

        series = fx_df.sort_values("ds").reset_index(drop=True)
        evaluated = 0
        with self._conn() as conn:
            pending = conn.execute(
                "SELECT run_date, horizon_days, latest_rate, direction, "
                "news_direction, ts_direction, market_direction "
                "FROM predictions WHERE evaluated_at IS NULL"
            ).fetchall()

            for (
                run_date,
                horizon,
                latest_rate,
                direction,
                news_direction,
                ts_direction,
                market_direction,
            ) in pending:
                target = pd.Timestamp(run_date) + timedelta(days=int(horizon))
                future = series[series["ds"] >= target]
                if future.empty or latest_rate is None:
                    continue  # el horizonte aún no llega

                actual_rate = float(future["y"].iloc[0])
                change_pct = (actual_rate - latest_rate) / latest_rate * 100
                if change_pct > EVAL_NEUTRAL_BAND_PCT:
                    actual_direction = "up"
                elif change_pct < -EVAL_NEUTRAL_BAND_PCT:
                    actual_direction = "down"
                else:
                    actual_direction = "neutral"

                final_hit = _score_direction(direction, actual_direction)
                news_hit = _score_direction(news_direction, actual_direction)
                ts_hit = _score_direction(ts_direction, actual_direction)
                market_hit = _score_direction(market_direction, actual_direction)
                conn.execute(
                    "UPDATE predictions SET actual_rate=?, actual_change_pct=?, "
                    "actual_direction=?, hit=?, final_hit=?, news_hit=?, "
                    "ts_hit=?, market_hit=?, evaluated_at=? WHERE run_date=?",
                    (
                        actual_rate,
                        round(change_pct, 3),
                        actual_direction,
                        final_hit,
                        final_hit,
                        news_hit,
                        ts_hit,
                        market_hit,
                        datetime.now(UTC).isoformat(),
                        run_date,
                    ),
                )
                evaluated += 1

        if evaluated:
            logger.info("Evaluated %d pending predictions", evaluated)
        return evaluated

    # ------------------------------------------------------------------
    # Lectura / métricas
    # ------------------------------------------------------------------

    def all(self) -> pd.DataFrame:
        import pandas as pd

        with self._conn() as conn:
            return pd.read_sql("SELECT * FROM predictions ORDER BY run_date DESC", conn)

    def metrics(self) -> dict[str, Any]:
        """Métricas direccionales sobre las predicciones ya evaluadas."""
        import pandas as pd

        df = self.all()
        done = df[df["evaluated_at"].notna()]
        decided = done[done["direction"] != "neutral"]
        hit_col = "final_hit" if "final_hit" in done.columns else "hit"

        out: dict[str, Any] = {
            "n_total": len(df),
            "n_evaluated": len(done),
            "n_decided": len(decided),
            "n_abstained": int((done["direction"] == "neutral").sum()),
            "hit_rate": float(decided[hit_col].mean()) if len(decided) else None,
        }
        if len(done):
            for name, col in (
                ("final", hit_col),
                ("news", "news_hit"),
                ("ts", "ts_hit"),
                ("market", "market_hit"),
            ):
                if col in done.columns:
                    scored = done[done[col].notna()]
                    out[f"{name}_hit_rate"] = float(scored[col].mean()) if len(scored) else None
                    out[f"{name}_n_decided"] = len(scored)
            out["confusion"] = pd.crosstab(
                done["direction"],
                done["actual_direction"],
                rownames=["predicho"],
                colnames=["real"],
            )
            decided_conf = decided.assign(
                conf_bucket=pd.cut(
                    decided["confidence"],
                    bins=[0, 0.45, 0.6, 1.0],
                    labels=["baja (<0.45)", "media", "alta (>0.6)"],
                )
            )
            out["hit_rate_by_confidence"] = (
                decided_conf.groupby("conf_bucket", observed=True)[hit_col]
                .agg(["mean", "count"])
                .rename(columns={"mean": "hit_rate", "count": "n"})
            )
        return out


def _score_direction(predicted: str | None, actual: str) -> int | None:
    """1/0 for directional calls; None for abstentions or missing signals."""
    if not predicted or predicted == "neutral":
        return None
    return int(predicted == actual)
