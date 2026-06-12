"""Application settings loaded from environment variables / .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_provider: Literal["openai", "anthropic"] = Field(
        "openai", description="Proveedor de LLM activo"
    )
    openai_api_key: SecretStr | None = Field(None, description="OpenAI API key")
    anthropic_api_key: SecretStr | None = Field(None, description="Anthropic API key (opcional)")
    # tier 'fast' — alto volumen, barato (clasificar/extraer por noticia)
    llm_model: str = Field("gpt-4o-mini", description="Modelo barato para volumen")
    # tier 'judge' — 1 llamada de alto valor (adjudicador / reconciliación)
    llm_model_judge: str = Field("gpt-4o", description="Modelo fuerte para el juicio final")
    llm_temperature: float = Field(0.1, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # FX data
    # ------------------------------------------------------------------
    alpha_vantage_api_key: SecretStr | None = Field(
        None, description="Alpha Vantage API key for FX data"
    )
    fx_base_currency: str = Field("USD", description="Base currency")
    fx_quote_currency: str = Field("COP", description="Quote currency")
    fx_lookback_days: int = Field(365, ge=30, description="Historical window in days")

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------
    newsapi_key: SecretStr | None = Field(None, description="NewsAPI.org key")
    news_rss_feeds: list[str] = Field(
        default=[
            # ── Lado COP: economía colombiana ─────────────────────────
            "https://feeds.eltiempo.com/rss/economia",
            "https://www.portafolio.co/rss/economia.xml",
            "https://feeds.semana.com/negocios",
            "https://rss.app/feeds/economia-colombia.xml",
            # ── Lado USD: macro global / Fed (el USD/COP es mitad dólar) ──
            "https://www.cnbc.com/id/20910258/device/rss/rss.html",      # CNBC Economy
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",  # MarketWatch
        ],
        description="RSS feeds: economía colombiana (lado COP) + macro global (lado USD)",
    )
    news_max_articles: int = Field(50, ge=1, le=200)

    # ------------------------------------------------------------------
    # Twitter / X
    # ------------------------------------------------------------------
    twitter_api_key: SecretStr | None = Field(None)
    twitter_api_secret: SecretStr | None = Field(None)
    twitter_access_token: SecretStr | None = Field(None)
    twitter_access_token_secret: SecretStr | None = Field(None)
    twitter_bearer_token: SecretStr | None = Field(None)
    twitter_enabled: bool = Field(False, description="Post tweets when True")

    # ------------------------------------------------------------------
    # App behaviour
    # ------------------------------------------------------------------
    environment: Literal["development", "staging", "production"] = Field("development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    report_output_dir: str = Field("reports", description="Local dir for generated reports")
    forecast_horizon_days: int = Field(7, ge=1, le=30)

    @field_validator("llm_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed_prefixes = ("claude-", "gpt-", "o1", "o3")
        if not any(v.startswith(p) for p in allowed_prefixes):
            raise ValueError(f"Unrecognised model id: {v}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
