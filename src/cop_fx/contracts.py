"""Contratos Pydantic del grafo de inteligencia (Etapa 0).

Estos modelos son la frontera entre el LLM y el resto del sistema:
todo lo que un modelo de lenguaje produce entra por `with_structured_output`
contra uno de estos schemas — nunca por parsing manual de JSON.

La "racionalidad" del sistema vive aquí, no en los prompts:
  - `DirectionalCall` exige `devils_advocate` (no hay veredicto sin contra-argumento),
  - los validadores acotan `confidence` cuando las señales divergen,
  - la abstención (`neutral`) es una salida válida forzada por contrato.

Ver docs/plan_maestro.md (Etapa 0) y docs/arquitectura.md §5-6.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Topic = Literal[
    "monetary_policy",  # BanRep, tasas, inflación
    "trade",            # importaciones / exportaciones / balanza
    "political_risk",   # elecciones, protestas, regulación
    "commodities",      # petróleo, café, carbón
    "macro",            # PIB, empleo, déficit fiscal
    "other",
]

Severity = Literal["high", "medium", "low"]

Direction = Literal["down", "up", "neutral"]
"""down = USD/COP cae (COP se fortalece) · up = USD/COP sube · neutral = abstención."""

# Pesos para agregar señales de noticias: severity → contribución al score
SEVERITY_WEIGHT: dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.2}


class ArticleAnalysis(BaseModel):
    """Veredicto del LLM sobre UN artículo (capa Gold)."""

    index: int = Field(ge=0, description="Posición del artículo en el lote enviado (0-based)")
    topic: Topic
    keywords: list[str] = Field(
        min_length=1, max_length=5, description="3-5 términos clave en español"
    )
    severity: Severity = Field(description="Impacto esperado sobre el USD/COP")
    bullish_cop: bool = Field(description="True si la noticia tiende a fortalecer el COP")
    reasoning: str = Field(max_length=240, description="≤ 20 palabras justificando el veredicto")


class BatchAnalysis(BaseModel):
    """Salida estructurada del clasificador de noticias — una llamada por lote."""

    items: list[ArticleAnalysis]
    market_narrative: str = Field(
        description="3 oraciones sobre el panorama FX del día, basadas SOLO en los artículos"
    )


class MaterialityGate(BaseModel):
    """Salida del router inicial (Etapa 2): ¿hay noticia material hoy?"""

    has_material_news: bool
    reason: str = Field(max_length=300)


class NewsSignal(BaseModel):
    """Señal direccional agregada de las noticias del día."""

    direction: Direction
    score: float = Field(
        description="Suma de bullish_cop ponderada por severity; > 0 ⇒ COP se fortalece"
    )
    drivers: list[str] = Field(
        default_factory=list,
        description="Títulos/URLs de los artículos que sustentan la señal (trazabilidad)",
    )


class TimeSeriesSignal(BaseModel):
    """Señal direccional del ensemble Prophet+ARIMA — determinista, sin LLM."""

    direction: Direction
    yhat_delta_pct: float = Field(description="Δ% del yhat final vs último valor real")
    models_agree: bool = Field(description="¿Prophet y ARIMA dan el mismo signo?")


class DirectionalCall(BaseModel):
    """Producto final diario: el veredicto direccional del adjudicador.

    El LLM propone; los validadores acotan. Reglas duras (no negociables
    por prompt): divergencia ⇒ techo de confianza 0.5; confianza < 0.35
    ⇒ el sistema se abstiene (`neutral`).
    """

    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    horizon_days: int = Field(ge=1, le=30)
    news_signal: NewsSignal
    ts_signal: TimeSeriesSignal
    reconciliation: Literal["agree", "diverge", "partial"]
    rationale: str = Field(description="Cadena de razonamiento que cita los drivers")
    devils_advocate: str = Field(
        min_length=20,
        description="El contra-argumento MÁS FUERTE contra la dirección elegida",
    )
    caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _bound_rationality(self) -> DirectionalCall:
        if self.reconciliation == "diverge" and self.confidence > 0.5:
            self.confidence = 0.5
        if self.direction != "neutral" and self.confidence < 0.35:
            self.direction = "neutral"
        return self


def aggregate_news_signal(
    analyses: list[ArticleAnalysis],
    titles_by_index: dict[int, str] | None = None,
    *,
    threshold: float = 0.5,
) -> NewsSignal:
    """Agrega veredictos por artículo en una señal direccional — determinista, sin LLM.

    score = Σ (±peso_severity); positivo si bullish_cop. |score| ≤ threshold ⇒ neutral.
    """
    score = 0.0
    drivers: list[str] = []
    for a in analyses:
        weight = SEVERITY_WEIGHT[a.severity]
        score += weight if a.bullish_cop else -weight
        if a.severity == "high" and titles_by_index is not None:
            title = titles_by_index.get(a.index)
            if title:
                drivers.append(title)

    direction: Direction
    if score > threshold:
        direction = "down"   # COP se fortalece ⇒ USD/COP cae
    elif score < -threshold:
        direction = "up"
    else:
        direction = "neutral"
    return NewsSignal(direction=direction, score=round(score, 3), drivers=drivers)
