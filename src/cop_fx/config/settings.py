"""Configuración del sistema — config.yaml (operativa) + .env (secretos)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from cop_fx.paths import ENV_FILE, PROJECT_ROOT, REPORTS_DIR


class Settings(BaseSettings):
    """Configuración del sistema.

    DOS archivos, responsabilidades separadas:
      - ``config.yaml`` (raíz): TODA la configuración operativa — modelo,
        número de artículos, feeds, bandas, workers. Editable sin tocar código.
      - ``.env`` (raíz): SOLO secretos (API keys, tokens de Twitter).

    Precedencia: env vars > .env > config.yaml > defaults del código.
    Rutas absolutas vía cop_fx.paths — funciona sin importar el cwd.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        yaml_file=str(PROJECT_ROOT / "config.yaml"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_provider: Literal["openai", "anthropic"] = Field(
        "openai", description="Proveedor de LLM activo"
    )
    openai_api_key: SecretStr | None = Field(None, description="OpenAI API key (.env)")
    anthropic_api_key: SecretStr | None = Field(None, description="Anthropic API key (.env)")
    # tier 'fast' — alto volumen, barato (clasificar/extraer por noticia)
    llm_model: str = Field("gpt-4o-mini", description="Modelo barato para volumen")
    # tier 'judge' — 1 llamada de alto valor (adjudicador / reconciliación)
    llm_model_judge: str = Field("gpt-4o", description="Modelo fuerte para el juicio final")
    llm_temperature: float = Field(0.1, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # FX data
    # ------------------------------------------------------------------
    alpha_vantage_api_key: SecretStr | None = Field(
        None, description="Alpha Vantage API key (.env, opcional)"
    )
    fx_base_currency: str = Field("USD", description="Base currency")
    fx_quote_currency: str = Field("COP", description="Quote currency")
    fx_lookback_days: int = Field(365, ge=30, description="Historical window in days")

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------
    newsapi_key: SecretStr | None = Field(None, description="NewsAPI.org key (.env, opcional)")
    news_rss_feeds: list[str] = Field(
        default=[
            "https://www.larepublica.co/rss/economia",
            "https://www.portafolio.co/rss/economia.xml",
            "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        ],
        description="RSS feeds: economía colombiana (lado COP) + macro global (lado USD)",
    )
    news_max_articles: int = Field(100, ge=1, le=200)
    gate_headlines_cap: int = Field(
        60, ge=10, le=200, description="Titulares que evalúa el gate de materialidad"
    )
    min_analyzable_chars: int = Field(
        80, ge=0, description="Sin cuerpo ni summary de este tamaño, el artículo se descarta"
    )
    keyword_llm_judge_enabled: bool = Field(
        True, description="Use an LLM to keep only actionable FX terms"
    )
    keyword_top_n: int = Field(25, ge=5, le=100)
    keyword_min_articles: int = Field(1, ge=1, le=20)
    keyword_pairwise_min_correlation: float = Field(0.20, ge=0.0, le=1.0)
    keyword_pairwise_min_joint: int = Field(2, ge=1, le=20)
    keyword_pairwise_top_n: int = Field(80, ge=10, le=500)
    keyword_domain_stopwords: list[str] = Field(
        default=[
            "colombia",
            "colombiano",
            "colombiana",
            "colombianos",
            "colombianas",
            "pais",
            "país",
            "gobierno",
            "nacional",
            "economia",
            "economía",
            "mercado",
            "mercados",
            "cop",
            "usd",
            "dolar",
            "dólar",
        ],
        description="Domain-generic terms removed before keyword ranking and co-occurrence",
    )

    # ------------------------------------------------------------------
    # Agentes
    # ------------------------------------------------------------------
    max_topic_workers: int = Field(
        6, ge=1, le=20, description="Cota del fan-out (Send) — controla el costo"
    )
    ts_neutral_band_pct: float = Field(
        0.10, ge=0.0, description="Banda muerta del forecast (la serie se abstiene)"
    )
    market_dead_band_pct: float = Field(
        0.30, ge=0.0, description="Banda muerta del equity (sin sesgo de mercado)"
    )

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
    report_output_dir: str = Field(
        default=str(REPORTS_DIR),
        description="Dir de reportes; relativo se resuelve contra la raíz",
    )
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
    return Settings()  # type: ignore[call-arg]
