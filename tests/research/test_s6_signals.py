"""Stream 6 — frozen base-signal builders (B2 real-rate shock, B5 flow divergence).

The mandate forbids re-tuning the dead base signals: B2 and B5 are reproduced VERBATIM
from the prior hunts (digest of hunt_X3_conditioning / hunt_b5_gldflow). These tests pin
the side convention and the publication-lag mechanics so a later refactor cannot silently
drift the frozen recipe. Gold rises when real rates FALL and when flow outruns price, so
a negative real-rate shock and a positive flow-vs-price divergence both map to LONG gold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "fxgate",
    reason="s6_gold_regime imports the standalone lab module fxgate (research_fx/, gitignored) — absent in CI",
)
from hermes.research import s6_gold_regime as s6


# --------------------------------------------------------------------------- #
# realrate_shock_events — frozen B2 (DFII10 5-day shock z, +/-1.5, +1bday entry)
# --------------------------------------------------------------------------- #
def _dfii_with_shock(direction):
    dates = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(1)
    x = 2.0 + rng.normal(0, 0.005, 300)
    x[286:291] += direction * np.linspace(0.02, 0.10, 5)  # a 5-day ramp shock
    return pd.Series(x, index=dates), dates


def test_realrate_falling_shock_is_long_gold():
    dfii, dates = _dfii_with_shock(direction=-1)  # real rates fall
    events = s6.realrate_shock_events(dfii, dates, z_thr=1.5, zwin=252)
    assert any(side == 1 for _, side in events)   # long gold on a real-rate drop


def test_realrate_rising_shock_is_short_gold():
    dfii, dates = _dfii_with_shock(direction=+1)  # real rates rise
    events = s6.realrate_shock_events(dfii, dates, z_thr=1.5, zwin=252)
    assert any(side == -1 for _, side in events)  # short gold on a real-rate spike


def test_realrate_events_respect_warmup_and_are_unique_bars():
    dfii, dates = _dfii_with_shock(direction=-1)
    events = s6.realrate_shock_events(dfii, dates, z_thr=1.5, zwin=252)
    bars = [i for i, _ in events]
    assert all(i >= 252 for i in bars)            # no event before the 252d z warm-up
    assert len(bars) == len(set(bars))            # one event per bar (deduped)


# --------------------------------------------------------------------------- #
# flow_divergence — frozen B5 variant B (div = zf - zp, 60d growth z's)
# --------------------------------------------------------------------------- #
def _flow_price(flow_dir, price_dir, n=240):
    """Ramping (accelerating) daily log-growth + tiny noise, so the 60d growth rate
    itself trends and the z-score is well-defined (a constant growth rate would give a
    zero-variance, all-NaN z). The SIGN of each leg's ramp sets the sign of its terminal
    z, so opposing legs give an unambiguous divergence sign."""
    idx = pd.bdate_range("2010-01-01", periods=n)
    rng = np.random.default_rng(0)
    t = np.arange(n)
    d_oz = flow_dir * 0.30 * (t / n) / 60.0 + rng.normal(0, 1e-4, n)
    d_px = price_dir * 0.30 * (t / n) / 60.0 + rng.normal(0, 1e-4, n)
    oz = pd.Series(1_000_000.0 * np.exp(np.cumsum(d_oz)), index=idx)
    px = pd.Series(100.0 * np.exp(np.cumsum(d_px)), index=idx)
    return oz, px


def test_flow_divergence_positive_when_flow_outruns_price():
    oz, px = _flow_price(flow_dir=+1, price_dir=-1)   # flow accumulating, price falling
    div = s6.flow_divergence(oz, px, lookback=60, zwin=60)["div"]
    assert div.dropna().iloc[-1] > 0


def test_flow_divergence_negative_when_price_outruns_flow():
    oz, px = _flow_price(flow_dir=-1, price_dir=+1)   # flow draining, price rising
    div = s6.flow_divergence(oz, px, lookback=60, zwin=60)["div"]
    assert div.dropna().iloc[-1] < 0


def test_flow_divergence_columns_are_causal_zscores():
    oz, px = _flow_price(+1, +1)
    out = s6.flow_divergence(oz, px, lookback=60, zwin=60)
    for col in ("zf", "zp", "div"):
        assert col in out.columns
    # first lookback+zwin rows are warm-up NaN (no future leak)
    assert out["div"].iloc[: 60].isna().all()


# --------------------------------------------------------------------------- #
# combined_etf_ounces — GLD ounces + IAU (shares x nav / gold price) in ounces
# --------------------------------------------------------------------------- #
def test_combined_etf_ounces_sums_gld_and_iau_in_ounces():
    idx = pd.bdate_range("2010-01-01", periods=3)
    gld_oz = pd.Series([1000.0, 1000.0, 1000.0], index=idx)
    iau_shares = pd.Series([100.0, 100.0, 100.0], index=idx)
    iau_nav = pd.Series([50.0, 50.0, 50.0], index=idx)       # $/share
    gold_px = pd.Series([2000.0, 2000.0, 2000.0], index=idx)  # $/oz
    out = s6.combined_etf_ounces(gld_oz, iau_shares, iau_nav, gold_px)
    # IAU ounces = 100*50/2000 = 2.5 -> combined = 1002.5
    assert out.iloc[0] == 1002.5
