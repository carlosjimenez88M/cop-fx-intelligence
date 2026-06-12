"""Contratos Pydantic del grafo de inteligencia (Etapas 0-3).

Estos modelos son la frontera entre el LLM y el resto del sistema:
todo lo que un modelo de lenguaje produce entra por `with_structured_output`
contra uno de estos schemas — nunca por parsing manual de JSON.

La "racionalidad" del sistema vive aquí, no en los prompts:
  - `DirectionalCall` exige `devils_advocate` (no hay veredicto sin contra-argumento),
  - los validadores acotan `confidence` cuando las señales divergen,
  - la abstención (`neutral`) es una salida válida forzada por contrato,
  - `fx_relevance="none"` excluye la noticia de la señal por construcción
    (un partido de fútbol no puede mover el score ni por error de prompt).

Taxonomía en dos niveles (qué ES la noticia ≠ cómo MUEVE el dólar):
  - `topic`: el dominio de la noticia (15 categorías, incluye salud pública,
    medio ambiente/clima, seguridad, deportes...).
  - `fx_channel`: el MECANISMO de transmisión al USD/COP. Una sequía
    (environment_climate) transmite por `inflation` (alimentos → BanRep);
    una epidemia (public_health) por `growth`/`country_risk`. Obligar al
    modelo a nombrar el canal es lo que separa análisis de opinión.

Ver docs/plan_maestro.md y docs/arquitectura.md §5-6.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Topic = Literal[
    "monetary_policy",      # BanRep, tasas, inflación
    "fiscal_policy",        # presupuesto, reforma tributaria, déficit, deuda pública
    "trade",                # exportaciones / importaciones / aranceles / balanza
    "political_risk",       # elecciones, gobernabilidad, regulación, instituciones
    "security_conflict",    # orden público, conflicto armado, narcotráfico
    "energy_commodities",   # petróleo, carbón, gas, minería, energía
    "agro_commodities",     # café, alimentos, sector agro
    "public_health",        # epidemias, sistema de salud, crisis sanitarias
    "environment_climate",  # clima, El Niño/La Niña, desastres, transición energética
    "labor_social",         # empleo, huelgas, paros, protesta social
    "financial_markets",    # bolsa, deuda, calificadoras, flujos de portafolio
    "us_global_macro",      # Fed, dólar global (DXY), economía mundial
    "sports",               # deportes
    "culture_society",      # cultura, entretenimiento, sociedad
    "other",
]

FxChannel = Literal[
    "interest_rates",   # diferencial de tasas COP vs USD
    "inflation",        # presión de precios → respuesta esperada del BanRep
    "terms_of_trade",   # precio de lo que Colombia exporta/importa
    "country_risk",     # prima de riesgo, percepción institucional
    "capital_flows",    # IED, flujos de portafolio, remesas
    "growth",           # actividad económica / PIB
    "none",             # sin mecanismo de transmisión al FX
]

FxRelevance = Literal["direct", "indirect", "none"]
"""direct = mueve el FX por sí sola · indirect = transmite por un canal de
segundo orden (clima→inflación, salud→crecimiento) · none = sin canal."""

Severity = Literal["high", "medium", "low"]

Direction = Literal["down", "up", "neutral"]
"""down = USD/COP cae (COP se fortalece) · up = USD/COP sube · neutral = abstención."""

# Pesos para agregar señales de noticias
SEVERITY_WEIGHT: dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.2}
RELEVANCE_WEIGHT: dict[str, float] = {"direct": 1.0, "indirect": 0.5, "none": 0.0}


class ArticleAnalysis(BaseModel):
    """Veredicto del LLM sobre UN artículo (capa Gold)."""

    index: int = Field(ge=0, description="Posición del artículo en el lote enviado (0-based)")
    topic: Topic
    keywords: list[str] = Field(
        min_length=1, max_length=5, description="3-5 términos clave en español"
    )
    entities: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Entidades nombradas (personas, instituciones, empresas, lugares)",
    )
    fx_relevance: FxRelevance = Field(
        description="¿La noticia tiene un canal de transmisión al USD/COP?"
    )
    fx_channel: FxChannel = Field(
        description="El mecanismo por el que la noticia mueve el USD/COP"
    )
    severity: Severity = Field(description="Magnitud esperada del impacto sobre el USD/COP")
    bullish_cop: bool = Field(description="True si la noticia tiende a fortalecer el COP")
    reasoning: str = Field(max_length=240, description="≤ 20 palabras justificando el veredicto")

    @model_validator(mode="after")
    def _coherence(self) -> ArticleAnalysis:
        # Sin relevancia FX no hay canal ni severidad — coherencia por contrato,
        # no por cortesía del prompt.
        if self.fx_relevance == "none":
            self.fx_channel = "none"
            self.severity = "low"
        elif self.fx_channel == "none":
            self.fx_relevance = "none"
            self.severity = "low"
        return self


class BatchAnalysis(BaseModel):
    """Salida estructurada del clasificador de noticias — una llamada por lote."""

    items: list[ArticleAnalysis]
    market_narrative: str = Field(
        description="3 oraciones sobre el panorama FX del día, basadas SOLO en los artículos"
    )


class HeadlineTag(BaseModel):
    """Etiqueta gruesa por titular — la produce el gate y siembra el clustering."""

    index: int = Field(ge=0, description="Posición del titular en la lista enviada (0-based)")
    topic: Topic
    material: bool = Field(description="¿Este titular en particular puede mover el USD/COP?")


class MaterialityGate(BaseModel):
    """Salida del router inicial (Etapa 2): ¿hay noticia material hoy?

    Además del veredicto global, etiqueta cada titular con un tópico grueso:
    la misma llamada que decide la ruta siembra los clusters del fan-out
    (Etapa 3) — cero costo adicional.
    """

    has_material_news: bool
    reason: str = Field(max_length=300)
    tags: list[HeadlineTag] = Field(default_factory=list)


class TopStory(BaseModel):
    """La noticia MÁS importante del día para el USD/COP — la elige un agente.

    El agente solo ELIGE entre candidatos ya analizados y justifica; los
    hechos (título, fuente, canal) los inyecta el sistema desde el artículo
    elegido. Importancia = severidad × canal de transmisión × novedad
    (un shock estructural le gana al ruido rutinario).
    """

    chosen_index: int = Field(ge=0, description="Índice del candidato elegido en la lista")
    why_it_matters: str = Field(
        min_length=30, description="Por qué ES la noticia del día (en español)"
    )
    watch_next: str = Field(
        min_length=10, description="Qué vigilar a continuación (en español)"
    )


class NewsSignal(BaseModel):
    """Señal direccional agregada de las noticias del día."""

    direction: Direction
    score: float = Field(
        description="Σ ±(peso_severity × peso_relevance); > 0 ⇒ COP se fortalece"
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


class MarketSignal(BaseModel):
    """Contexto de mercado determinista — validado por el estudio macro.

    El agregado bursátil colombiano (GXG) fue el ÚNICO activo con poder
    adelantado robusto (equity[t-1]→cop[t] ≈ -0.4, ver notebooks/02):
    bolsa arriba ayer ⇒ COP se fortalece ⇒ USD/COP tiende a bajar.
    DXY y Brent van como contexto, no como voto. Café, oro, VIX, tasas US
    y pares LatAm fueron probados y descartados con datos.
    """

    direction: Direction = Field(description="Regla sobre el retorno de ayer del equity")
    equity_ret_1d_pct: float
    dxy_ret_1d_pct: float
    brent_ret_1d_pct: float


class AdjudicatorVerdict(BaseModel):
    """Lo que el LLM adjudicador produce — y NADA más.

    Las señales (`news_signal`, `ts_signal`) y el horizonte los aporta el
    sistema al componer el `DirectionalCall`: el modelo juzga, no inventa
    números. Separar el veredicto del contrato final es lo que impide que
    el LLM "corrija" un score que no le gustó.
    """

    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    reconciliation: Literal["agree", "diverge", "partial"]
    rationale: str = Field(description="Cadena de razonamiento que cita los drivers")
    devils_advocate: str = Field(
        min_length=20,
        description="El contra-argumento MÁS FUERTE contra la dirección elegida",
    )
    caveats: list[str] = Field(default_factory=list)


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

    score = Σ ±(peso_severity × peso_relevance); positivo si bullish_cop.
    Artículos con fx_relevance="none" pesan 0 por construcción.
    |score| ≤ threshold ⇒ neutral.
    """
    score = 0.0
    drivers: list[str] = []
    for a in analyses:
        weight = SEVERITY_WEIGHT[a.severity] * RELEVANCE_WEIGHT[a.fx_relevance]
        if weight == 0.0:
            continue
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
