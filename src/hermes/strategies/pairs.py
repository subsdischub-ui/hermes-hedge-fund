"""Cross-asset ETH-BTC pairs strategy (Engle-Granger cointegration).

Thesis: ETH and BTC share crypto-beta and tend to move together, so their
price spread is mean-reverting when the two are cointegrated. We estimate the
hedge ratio by OLS, test the residual for stationarity (ADF), and trade the
z-score of the spread: fade extremes (|z|>z_in), exit near the mean (|z|<z_out).
Market-neutral by construction (long one leg, short the hedged other leg).

Anti-overfit: every parameter is frozen a priori (lookback=200, z_in=2.0,
z_out=0.5, p_threshold=0.05) — round, standard pairs-trading values, never
optimized. The skill self-disables (HOLD) when the pair is not cointegrated.

No look-ahead: cointegration, hedge ratio, spread mean/std and the traded
z-score are all computed on the trailing window of `lookback` bars that ENDS
at the current bar (``.iloc[-lookback:]``). No future bar enters any statistic;
the z-score uses the same window's mean/std as the last spread observation, so
the decision at bar t depends only on data <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from hermes.skills.library import SkillSignal

# Frozen parameters (a priori, never optimized).
_LOOKBACK = 200
_Z_IN = 2.0
_Z_OUT = 0.5
_P_THRESHOLD = 0.05


def cointegration(
    y: np.ndarray | pd.Series,
    x: np.ndarray | pd.Series,
    p_threshold: float = _P_THRESHOLD,
) -> tuple[bool, float, float]:
    """Engle-Granger cointegration test of y on x.

    Fits OLS ``y = a + hedge * x`` for the hedge ratio, runs an ADF test on the
    regression residual and declares cointegration when the ADF p-value is below
    ``p_threshold``.

    Returns ``(is_coint, hedge_ratio, pvalue)``.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.shape[0] != x_arr.shape[0] or y_arr.shape[0] < 3:
        return False, 0.0, 1.0
    if not (np.all(np.isfinite(y_arr)) and np.all(np.isfinite(x_arr))):
        return False, 0.0, 1.0

    design = sm.add_constant(x_arr)
    model = sm.OLS(y_arr, design).fit()
    hedge = float(model.params[-1])
    resid = y_arr - design @ model.params

    if np.std(resid) == 0:
        return False, hedge, 1.0
    pvalue = float(adfuller(resid, autolag="AIC")[1])
    return bool(pvalue < p_threshold), hedge, pvalue


def pairs_eth_btc(
    df_eth: pd.DataFrame,
    df_btc: pd.DataFrame,
    lookback: int = _LOOKBACK,
    z_in: float = _Z_IN,
    z_out: float = _Z_OUT,
) -> SkillSignal:
    """Trade the mean-reverting ETH-BTC spread on the trailing ``lookback`` bars.

    SELL the spread when z>z_in (ETH rich vs BTC), BUY when z<-z_in (ETH cheap),
    HOLD (exit) when |z|<z_out. If the pair is not cointegrated on the window,
    the skill is disabled (HOLD). Parameters are frozen.
    """
    skill_id = "pairs_eth_btc"

    def _hold(rationale: str, features: dict[str, float]) -> SkillSignal:
        return SkillSignal(
            skill_id=skill_id,
            side="HOLD",
            confidence=0.0,
            stop_pct=0.02,
            take_profit_pct=0.04,
            rationale=rationale,
            features=features,
        )

    eth = pd.Series(df_eth["close"].to_numpy(dtype=float))
    btc = pd.Series(df_btc["close"].to_numpy(dtype=float))
    n = min(len(eth), len(btc))
    if n < lookback:
        return _hold(f"insufficient history n={n}<{lookback}", {"z": 0.0})

    # Trailing window ENDING at the current bar — strictly <= bar t, no look-ahead.
    eth_w = eth.iloc[n - lookback : n].to_numpy()
    btc_w = btc.iloc[n - lookback : n].to_numpy()

    is_coint, hedge, pvalue = cointegration(eth_w, btc_w)
    if not is_coint:
        return _hold(
            f"not cointegrated p={pvalue:.3f}>={_P_THRESHOLD} (disabled)",
            {"z": 0.0, "pvalue": pvalue, "hedge": hedge},
        )

    spread = eth_w - hedge * btc_w
    mean = float(np.mean(spread))
    std = float(np.std(spread))
    if std == 0:
        return _hold("degenerate spread (std=0)", {"z": 0.0, "pvalue": pvalue})

    z = (float(spread[-1]) - mean) / std
    confidence = float(min(1.0, abs(z) / 3.0))
    features = {"z": z, "pvalue": pvalue, "hedge": hedge, "spread": float(spread[-1])}

    side: str
    if z > z_in:
        side = "SELL"  # spread rich -> fade: short ETH leg / long BTC hedge
    elif z < -z_in:
        side = "BUY"  # spread cheap -> long ETH leg / short BTC hedge
    elif abs(z) < z_out:
        return _hold(f"z={z:.2f} within exit band |z|<{z_out}", features)
    else:
        # In the dead-band between exit and entry thresholds: no new position.
        return _hold(f"z={z:.2f} in dead-band ({z_out}<=|z|<={z_in})", features)

    return SkillSignal(
        skill_id=skill_id,
        side=side,  # type: ignore[arg-type]
        confidence=confidence,
        stop_pct=0.02,
        take_profit_pct=0.04,
        rationale=f"z={z:.2f} hedge={hedge:.3f} p={pvalue:.3f} -> {side} spread",
        features=features,
    )
