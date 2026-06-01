"""Download historical COP/USD exchange rate data."""

from __future__ import annotations

from cop_fx.logger import get_logger
from datetime import date, timedelta

import httpx
import pandas as pd

from cop_fx.config.settings import get_settings

logger = get_logger(__name__)

_AV_BASE = "https://www.alphavantage.co/query"


class FXFetcher:
    """Fetches daily COP/USD rates from Alpha Vantage (primary) or Banco de la República (fallback)."""

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

        av_key = self._settings.alpha_vantage_api_key
        if av_key:
            try:
                df = self._fetch_alpha_vantage(av_key.get_secret_value())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Alpha Vantage failed (%s), falling back to BanRep", exc)
                df = self._fetch_banrep(start, end)
        else:
            logger.info("No Alpha Vantage key — using BanRep endpoint")
            df = self._fetch_banrep(start, end)

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
        rows = [
            {"ds": pd.Timestamp(k), "y": float(v["4. close"])}
            for k, v in series.items()
        ]
        return pd.DataFrame(rows)

    def _fetch_banrep(self, start: date, end: date) -> pd.DataFrame:
        """Fetch from Banco de la República's public TRM endpoint."""
        url = (
            f"https://www.banrep.gov.co/es/estadisticas/trm"
            f"?inicio={start.strftime('%d/%m/%Y')}"
            f"&fin={end.strftime('%d/%m/%Y')}"
        )
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        tables = pd.read_html(resp.text)
        if not tables:
            raise ValueError("BanRep returned no parseable tables")

        df = tables[0].copy()
        df.columns = ["date_str", "y"]
        df["ds"] = pd.to_datetime(df["date_str"], dayfirst=True, errors="coerce")
        df["y"] = pd.to_numeric(
            df["y"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
        return df[["ds", "y"]].dropna()
