"""Modelo direccional macro del USD/COP — ¿sube o baja en H días?

Por qué existe
--------------
El forecast univariante (Prophet+ARIMA) solo mira el pasado del propio USD/COP.
El estudio de la notebook 02 mostró que el COP se mueve como **activo de riesgo
emergente**: contemporáneamente correlaciona fuerte con los pares EM (CLP/MXN/BRL),
la bolsa local y el DXY. Este módulo convierte esa estructura en una señal
direccional supervisada: aprende, sobre la historia, el mapeo

    estado macro de HOY  →  P(USD/COP suba en H días)

usando SOLO información conocida en el momento de la predicción (sin fuga de
futuro: las features son retornos pasados de las exógenas, el target es el signo
del COP H días adelante).

Postura crítica
---------------
En FX diario la señal predictiva es débil (cercana a martingala). Por eso:

  - El modelo de producción es **regresión logística regularizada** sobre un set
    compacto de features: rápido, determinista y difícil de sobreajustar con
    ~1000 observaciones. La notebook compara contra gradient boosting y una GRU
    (deep learning) y documenta que la complejidad extra **no** compra accuracy
    out-of-sample estable — de ahí la elección conservadora.
  - La señal sale con una **banda muerta** sobre la probabilidad: si el modelo no
    está suficientemente convencido, se ABSTIENE (`neutral`), coherente con el
    resto del sistema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cop_fx.data.market_fetcher import fetch_yahoo_series
from cop_fx.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

# Universo de drivers (todos con ~5 años en Yahoo). cop = USDCOP spot.
MACRO_SYMBOLS: dict[str, str] = {
    "cop": "COP=X",
    "dxy": "DX-Y.NYB",
    "equity": "GXG",
    "brent": "BZ=F",
    "cobre": "HG=F",
    "oro": "GC=F",
    "eem": "EEM",
    "emb": "EMB",
    "vix": "^VIX",
    "usdmxn": "MXN=X",
    "usdbrl": "BRL=X",
    "usdclp": "CLP=X",
}
RETURN_WINDOWS = (1, 5, 10)


# ---------------------------------------------------------------------------
# Datos y features
# ---------------------------------------------------------------------------


def build_macro_panel(
    *, lookback_days: int = 1500, symbols: dict[str, str] | None = None
) -> pd.DataFrame:
    """Panel [ds, cop, <exógenas>] alineado por fecha (inner join)."""
    syms = symbols or MACRO_SYMBOLS
    panel: pd.DataFrame | None = None
    for name, symbol in syms.items():
        try:
            serie = fetch_yahoo_series(symbol, lookback_days).rename(columns={"y": name})
        except Exception as exc:  # un símbolo caído no debe tumbar el panel
            logger.warning("macro panel: %s (%s) no disponible: %s", name, symbol, exc)
            continue
        panel = serie if panel is None else panel.merge(serie, on="ds", how="inner")
    if panel is None or panel.empty:
        return pd.DataFrame()
    return panel.sort_values("ds").reset_index(drop=True)


def make_features(
    panel: pd.DataFrame,
    *,
    horizon: int = 5,
    windows: Sequence[int] = RETURN_WINDOWS,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Construye (X, y, feature_cols) sin fuga de futuro.

    X[t] = retornos pasados (ventanas `windows`) de todas las series.
    y[t] = 1 si cop[t+horizon] > cop[t] (USD/COP sube), 0 si baja.
    """
    px = panel.set_index("ds")
    feats = pd.DataFrame(index=px.index)
    for col in px.columns:
        for w in windows:
            feats[f"{col}_r{w}"] = px[col].pct_change(w)
    target = (px["cop"].shift(-horizon) > px["cop"]).astype("float")
    feats = feats.dropna()
    target = target.reindex(feats.index)
    valid = target.notna()
    feats, target = feats[valid], target[valid]
    return feats, target, list(feats.columns)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


class LogisticDirectional:
    """Regresión logística regularizada (estándar de producción)."""

    def __init__(self, c: float = 0.25) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=1000),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticDirectional:
        self._model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict_proba(x))[:, 1]


class GradientBoostDirectional:
    """Gradient boosting (sklearn) — referencia no lineal sin deep learning."""

    def __init__(self) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        self._model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> GradientBoostDirectional:
        self._model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict_proba(x))[:, 1]


@dataclass
class GRUConfig:
    seq_len: int = 20
    hidden: int = 16
    epochs: int = 60
    lr: float = 1e-3
    dropout: float = 0.2
    seed: int = 42


class GRUDirectional:
    """GRU pequeña (PyTorch) sobre secuencias de retornos diarios multivariados.

    Deep learning *con humildad*: una sola capa, hidden chico y dropout, porque
    con ~1000 observaciones ruidosas una red grande solo memoriza. Se entrena en
    CPU/MPS en segundos. Pensada para el experimento de la notebook, no para
    correr en cada pipeline (ver módulo: producción usa la logística).
    """

    def __init__(self, n_features: int, config: GRUConfig | None = None) -> None:
        self._cfg = config or GRUConfig()
        self._n_features = n_features
        self._net: Any = None

    def _build(self) -> Any:
        import torch
        from torch import nn

        cfg = self._cfg
        torch.manual_seed(cfg.seed)

        class _Net(nn.Module):
            def __init__(self, n_in: int, hidden: int, dropout: float) -> None:
                super().__init__()
                self.gru = nn.GRU(n_in, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

            def forward(self, x: object) -> object:
                out, _ = self.gru(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _Net(self._n_features, cfg.hidden, cfg.dropout)

    def fit(self, seqs: np.ndarray, y: np.ndarray) -> GRUDirectional:
        import torch
        from torch import nn

        self._net = self._build()
        net = self._net
        opt = torch.optim.Adam(net.parameters(), lr=self._cfg.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        xb = torch.tensor(seqs, dtype=torch.float32)
        yb = torch.tensor(y, dtype=torch.float32)
        net.train()
        for _ in range(self._cfg.epochs):
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, seqs: np.ndarray) -> np.ndarray:
        import torch

        net = self._net
        if net is None:
            raise RuntimeError("GRU no entrenada")
        net.eval()
        with torch.no_grad():
            logits = net(torch.tensor(seqs, dtype=torch.float32))
            return np.asarray(torch.sigmoid(logits).numpy())


# ---------------------------------------------------------------------------
# Señal de producción (para el pipeline)
# ---------------------------------------------------------------------------


@dataclass
class MacroDirectionalSignal:
    direction: str  # "up" | "down" | "neutral"
    prob_up: float
    horizon_days: int
    n_train: int
    top_drivers: list[tuple[str, float]] = field(default_factory=list)


def macro_directional_signal(
    *,
    panel: pd.DataFrame | None = None,
    horizon: int = 5,
    neutral_band: float = 0.06,
    lookback_days: int = 1500,
) -> MacroDirectionalSignal | None:
    """Entrena la logística sobre la historia y predice la dirección de HOY.

    `neutral_band` define la zona de abstención alrededor de 0.5: si
    |P(up) - 0.5| < band, la señal es `neutral`. Devuelve ``None`` si no hay
    datos suficientes (el llamador degrada con elegancia).
    """
    panel = panel if panel is not None else build_macro_panel(lookback_days=lookback_days)
    if panel.empty or "cop" not in panel.columns or len(panel) < 200:
        logger.warning("macro_directional_signal: panel insuficiente")
        return None

    feats, target, cols = make_features(panel, horizon=horizon)
    if len(feats) < 150:
        return None

    # Última fila CON features (target NaN porque mira al futuro): es el "hoy".
    px = panel.set_index("ds")
    all_feats = pd.DataFrame(index=px.index)
    for col in px.columns:
        for w in RETURN_WINDOWS:
            all_feats[f"{col}_r{w}"] = px[col].pct_change(w)
    today = all_feats.dropna().iloc[[-1]][cols]

    model = LogisticDirectional().fit(feats.to_numpy(), target.to_numpy())
    prob_up = float(model.predict_proba(today.to_numpy())[0])

    if prob_up - 0.5 > neutral_band:
        direction = "up"
    elif 0.5 - prob_up > neutral_band:
        direction = "down"
    else:
        direction = "neutral"

    return MacroDirectionalSignal(
        direction=direction,
        prob_up=round(prob_up, 4),
        horizon_days=horizon,
        n_train=len(feats),
    )


# ---------------------------------------------------------------------------
# Evaluación walk-forward (para la notebook y los tests)
# ---------------------------------------------------------------------------


def walk_forward_accuracy(
    panel: pd.DataFrame,
    *,
    horizon: int = 5,
    min_train: int = 252,
    n_eval: int = 252,
    refit_every: int = 21,
    seq_len: int = 20,
    include_gru: bool = True,
) -> pd.DataFrame:
    """Compara estrategias direccionales out-of-sample (ventana expansiva).

    Devuelve un DataFrame [modelo, accuracy, n]. Honesto por diseño: el
    estandarizado y el ajuste se hacen SOLO con datos previos a cada origen.
    """
    feats, target, _cols = make_features(panel, horizon=horizon)
    x_all = feats.to_numpy()
    y_all = target.to_numpy().astype(int)
    cop = panel.set_index("ds")["cop"].reindex(feats.index).to_numpy()
    rets = panel.set_index("ds").pct_change().reindex(feats.index).fillna(0.0).to_numpy()

    n = len(feats)
    start = max(min_train, n - n_eval)
    origins = list(range(start, n))
    if not origins:
        return pd.DataFrame(columns=["modelo", "accuracy", "n"])

    preds: dict[str, list[int]] = {k: [] for k in ("always_up", "momentum", "logistic", "gboost")}
    if include_gru:
        preds["gru"] = []
    actual: list[int] = []

    log_model: LogisticDirectional | None = None
    gb_model: GradientBoostDirectional | None = None
    gru_model: GRUDirectional | None = None

    for k, t in enumerate(origins):
        if k % refit_every == 0:
            log_model = LogisticDirectional().fit(x_all[:t], y_all[:t])
            gb_model = GradientBoostDirectional().fit(x_all[:t], y_all[:t])
            if include_gru:
                seqs, seq_y = _make_sequences(rets[:t], y_all[:t], seq_len)
                gru_model = GRUDirectional(rets.shape[1]).fit(seqs, seq_y)

        assert log_model is not None and gb_model is not None  # set en k==0
        actual.append(int(y_all[t]))
        preds["always_up"].append(1)
        preds["momentum"].append(1 if cop[t] > cop[t - 1] else 0)
        preds["logistic"].append(int(log_model.predict_proba(x_all[t : t + 1])[0] > 0.5))
        preds["gboost"].append(int(gb_model.predict_proba(x_all[t : t + 1])[0] > 0.5))
        if include_gru and gru_model is not None and t >= seq_len:
            seq = rets[t - seq_len + 1 : t + 1][None, :, :]
            preds["gru"].append(int(gru_model.predict_proba(seq)[0] > 0.5))
        elif include_gru:
            preds["gru"].append(1)

    rows = [
        {
            "modelo": name,
            "accuracy": float(np.mean(np.array(p) == np.array(actual))),
            "n": len(actual),
        }
        for name, p in preds.items()
    ]
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def _make_sequences(rets: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    seqs, ys = [], []
    for i in range(seq_len, len(rets)):
        seqs.append(rets[i - seq_len : i])
        ys.append(y[i])
    return np.asarray(seqs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
