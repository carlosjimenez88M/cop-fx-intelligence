"""Fábrica única de modelos de chat — provider-agnostic.

Toda construcción de LLM en el proyecto pasa por aquí. Cambiar de proveedor
(OpenAI ↔ Anthropic) o de modelo es UNA línea en `.env`, no un refactor.

Dos tiers, por economía (ver docs/arquitectura.md §8):
  - "fast"  → alto volumen, tarea simple (clasificar/extraer por noticia)
  - "judge" → 1 llamada de alto valor (el adjudicador / reconciliación)

Uso::

    from cop_fx.llm import get_chat_model
    llm = get_chat_model("fast").with_structured_output(MiSchema)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain.chat_models import init_chat_model

from cop_fx.config.settings import get_settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

Tier = Literal["fast", "judge"]


def get_chat_model(tier: Tier = "fast", *, temperature: float | None = None) -> BaseChatModel:
    """Devuelve un chat model listo, según el proveedor configurado en settings.

    `tier="fast"` usa `llm_model`; `tier="judge"` usa `llm_model_judge`.
    El proveedor ("openai" | "anthropic") y la API key salen de settings.
    """
    s = get_settings()
    model = s.llm_model if tier == "fast" else s.llm_model_judge

    if s.llm_provider == "openai":
        if s.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY no está en .env y llm_provider='openai'")
        api_key = s.openai_api_key.get_secret_value()
    else:
        if s.anthropic_api_key is None:
            raise RuntimeError("ANTHROPIC_API_KEY no está en .env y llm_provider='anthropic'")
        api_key = s.anthropic_api_key.get_secret_value()

    return init_chat_model(
        model,
        model_provider=s.llm_provider,
        temperature=s.llm_temperature if temperature is None else temperature,
        api_key=api_key,
    )
