"""Regime-gated mean-reversion skill.

Thèse: la réversion à la moyenne n'a un edge QU'EN range. Le piège de l'ancien
`mean_reversion_atr` était de trader la réversion en tendance (on achète la
baisse... qui continue). Ici on gate strictement sur le régime Kaufman: si le
régime n'est pas "range", on renvoie HOLD. En range, on combine RSI(2) (Connors,
très réactif aux extrêmes courts) + Bollinger %B(20,2), avec un filtre de
tendance long terme MA200 (on n'achète l'excès baissier que si le prix reste
au-dessus de sa MA200, et inversement).

Paramètres FIGÉS a priori (valeurs standard, aucune optimisation):
- RSI longueur 2 ; seuils 10 / 90 (Connors).
- Bollinger 20 barres, 2 écarts-types ; seuils %B 0.1 / 0.9.
- Filtre tendance MA200 ; stop/tp via ATR(14).

Pas de look-ahead: tous les indicateurs (rolling mean/std, RSI, ATR) n'utilisent
que des barres <= barre courante. Aucun .shift négatif, aucune fenêtre future.
La décision se lit sur .iloc[-1] (la barre la plus récente, déjà clôturée).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from hermes.data.regime import detect_regime
from hermes.skills.library import SkillSignal

# --- Paramètres figés (standards Connors / Bollinger) ---
_RSI_LEN = 2
_RSI_LOW = 10.0
_RSI_HIGH = 90.0
_BB_LEN = 20
_BB_K = 2.0
_PCTB_LOW = 0.1
_PCTB_HIGH = 0.9
_MA_LONG = 200
_ATR_LEN = 14
_CONFIDENCE = 0.6  # ~0.55-0.65, signal contrarien modéré


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """RSI Connors (moyennes mobiles simples, longueur courte). Causal: .rolling
    n'utilise que des barres passées. Cas limites standard: down=0 & up>0 -> 100
    (que des hausses); up=0 & down>0 -> 0; aucun mouvement -> 50."""
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # down==0: que des hausses -> 100 si up>0, sinon (aucun mouvement) -> 50.
    rsi = rsi.where(~((down == 0) & (up > 0)), 100.0)
    rsi = rsi.where(~((down == 0) & (up == 0)), 50.0)
    return rsi.astype(float).fillna(50.0)


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, c = df["high"], df["close"]
    low = df["low"]
    prev_c = c.shift(1)
    tr = pd.concat([h - low, (h - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def mean_reversion_regime(df: pd.DataFrame) -> SkillSignal:
    """Réversion conditionnée au régime: trade l'excès RSI/%B uniquement en range.

    BUY si range ET RSI(2)<10 ET %B<0.1 ET close>MA200 (réversion haussière sous
    tendance de fond positive). SELL symétrique. Sinon HOLD.
    """
    close = df["close"]
    px = float(close.iloc[-1])

    atr_series = _atr(df, _ATR_LEN)
    atr = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
    stop_pct = float(2.0 * atr / px) if (atr == atr and atr > 0 and px > 0) else 0.02
    tp_pct = float(3.0 * atr / px) if (atr == atr and atr > 0 and px > 0) else 0.04

    side: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    conf = 0.0

    # Historique insuffisant pour MA200 -> jamais de trade sur données partielles.
    enough_history = len(close) >= _MA_LONG + 1

    regime = detect_regime(df)["regime"]

    rsi = _rsi(close, _RSI_LEN)
    ma = close.rolling(_BB_LEN).mean()
    sd = close.rolling(_BB_LEN).std()
    upper = ma + _BB_K * sd
    lower = ma - _BB_K * sd
    width = (upper - lower).replace(0, np.nan)
    pct_b = ((close - lower) / width).astype(float)

    rsi_now = float(rsi.iloc[-1])
    pctb_now = float(pct_b.iloc[-1]) if pct_b.iloc[-1] == pct_b.iloc[-1] else 0.5
    ma_long = float(close.rolling(_MA_LONG).mean().iloc[-1]) if enough_history else float("nan")
    above_ma = enough_history and ma_long == ma_long and px > ma_long
    below_ma = enough_history and ma_long == ma_long and px < ma_long

    if enough_history and regime == "range":
        if rsi_now < _RSI_LOW and pctb_now < _PCTB_LOW and above_ma:
            side, conf = "BUY", _CONFIDENCE
        elif rsi_now > _RSI_HIGH and pctb_now > _PCTB_HIGH and below_ma:
            side, conf = "SELL", _CONFIDENCE

    return SkillSignal(
        skill_id="mean_reversion_regime",
        side=side,
        confidence=conf,
        stop_pct=stop_pct,
        take_profit_pct=tp_pct,
        rationale=(
            f"regime={regime} rsi2={rsi_now:.1f} pctB={pctb_now:.2f} "
            f"px={px:.2f} ma200={ma_long:.2f}"
        ),
        features={
            "rsi2": rsi_now,
            "pct_b": pctb_now,
            "atr_pct": stop_pct / 2,
        },
    )
