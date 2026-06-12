"""Download historical COP/USD exchange rate data."""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pandas as pd

from cop_fx.config.settings import get_settings
from cop_fx.logger import get_logger

logger = get_logger(__name__)

_AV_BASE = "https://www.alphavantage.co/query"


class FXFetcher:
    """Fetches daily COP/USD rates.

    Cadena de fuentes: Alpha Vantage (si hay key) → TRM oficial vía
    datos.gov.co (Socrata, sin key) → Yahoo Finance. El viejo scraper de
    banrep.gov.co quedó bloqueado por bot-manager; datos.gov.co publica la
    MISMA TRM de la Superfinanciera por API abierta.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, lookback_days: int | None = None) -> pd.DataFrame:
        """Return a DataFrame with columns [ds, y] suitable for Prophet.

        ds  — datetime (daily, UTC midnight)
        y   — COP per 1 USD
        """
        days = lookback_days or self._settings.fx_lookback_days
        end = date.today()
        start = end - timedelta(days=days)

        sources: list[tuple[str, object]] = []
        av_key = self._settings.alpha_vantage_api_key
        if av_key:
            sources.append(
                ("Alpha Vantage", lambda: self._fetch_alpha_vantage(av_key.get_secret_value()))
            )
        sources.append(("TRM datos.gov.co", lambda: self._fetch_trm_datos_gov(start, end)))
        sources.append(("Yahoo Finance", lambda: self._fetch_yahoo(days)))

        df: pd.DataFrame | None = None
        for name, fetch_fn in sources:
            try:
                df = fetch_fn()  # type: ignore[operator]
                logger.info("FX source: %s", name)
                break
            except Exception as exc:
                logger.warning("FX source %s failed (%s) — trying next", name, exc)
        if df is None or df.empty:
            raise RuntimeError("All FX sources failed (Alpha Vantage / datos.gov.co / Yahoo)")

        df = df[(df["ds"] >= pd.Timestamp(start)) & (df["ds"] <= pd.Timestamp(end))]
        df = df.sort_values("ds").reset_index(drop=True)
        logger.info("Fetched %d FX rows (start=%s, end=%s)", len(df), start, end)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_alpha_vantage(self, api_key: str) -> pd.DataFrame:
        params = {
            "function": "FX_DAILY",
            "from_symbol": self._settings.fx_base_currency,
            "to_symbol": self._settings.fx_quote_currency,
            "outputsize": "full",
            "apikey": api_key,
        }
        with httpx.Client(timeout=30) as client:
            resp = client.get(_AV_BASE, params=params)
            resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data or "Note" in data:
            raise ValueError(data.get("Error Message") or data.get("Note"))

        series = data["Time Series FX (Daily)"]
        rows = [{"ds": pd.Timestamp(k), "y": float(v["4. close"])} for k, v in series.items()]
        return pd.DataFrame(rows)

    def _fetch_trm_datos_gov(self, start: date, end: date) -> pd.DataFrame:
        """TRM oficial (Superfinanciera) vía la API Socrata de datos.gov.co.

        Dataset 32sa-8pi3 — sin API key, JSON limpio. Es la misma TRM que
        publica el BanRep, pero por un endpoint pensado para máquinas.
        """
        url = "https://www.datos.gov.co/resource/32sa-8pi3.json"
        params = {
            "$select": "vigenciadesde,valor",
            "$where": (
                f"vigenciadesde >= '{start.isoformat()}T00:00:00.000'"
                f" AND vigenciadesde <= '{end.isoformat()}T23:59:59.000'"
            ),
            "$order": "vigenciadesde ASC",
            "$limit": "5000",
        }
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
        rows = resp.json()
        if not rows:
            raise ValueError("datos.gov.co returned no TRM rows")

        df = pd.DataFrame(rows)
        df["ds"] = pd.to_datetime(df["vigenciadesde"], errors="coerce")
        df["y"] = pd.to_numeric(df["valor"], errors="coerce")
        return df[["ds", "y"]].dropna()

    def _fetch_yahoo(self, lookback_days: int) -> pd.DataFrame:
        """USD/COP histórico desde la chart API pública de Yahoo Finance."""
        quote = self._settings.fx_quote_currency
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote}=X"
        params = {"range": f"{max(lookback_days, 30)}d", "interval": "1d"}
        headers = {"User-Agent": "Mozilla/5.0 (cop-fx-intelligence)"}
        with httpx.Client(timeout=30, headers=headers) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]

        df = pd.DataFrame(
            {
                "ds": pd.to_datetime(timestamps, unit="s").normalize(),
                "y": pd.to_numeric(pd.Series(closes), errors="coerce"),
            }
        )
        return df.dropna()
