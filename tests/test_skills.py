import numpy as np
import pandas as pd

from hermes.skills import get_skill_library


def _trending_df(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    drift = np.linspace(0, 0.10, n)
    noise = rng.standard_normal(n) * 0.002
    rets = drift + noise
    px = 50_000 * np.exp(rets.cumsum() / n)
    high = px * (1 + np.abs(rng.standard_normal(n) * 0.001))
    low = px * (1 - np.abs(rng.standard_normal(n) * 0.001))
    open_ = px * (1 + rng.standard_normal(n) * 0.0005)
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": px, "volume": vol})


def test_skills_return_valid_signals():
    lib = get_skill_library()
    df = _trending_df()
    sigs = lib.evaluate_all(df)
    assert sigs
    for s in sigs:
        assert s.side in {"BUY", "SELL", "HOLD"}
        assert 0.0 <= s.confidence <= 1.0
        assert s.stop_pct > 0
        assert s.take_profit_pct > 0


def _funding_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    px = 100.0 + rng.standard_normal(n) * 0.1
    fr = rng.standard_normal(n) * 0.0001
    fr[-1] = 0.02  # extreme-positive funding on the last bar -> z >> 2
    return pd.DataFrame({
        "open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
        "volume": rng.uniform(10, 100, n), "funding_rate": fr,
    })


def test_funding_strategy2_registered_live_and_longs_extreme_funding():
    """Strategy #2 is wired into the LIVE skill library and fires LONG on an
    extreme-positive funding tail (the validated continuation sign)."""
    lib = get_skill_library()
    sigs = {s.skill_id: s for s in lib.evaluate_all(_funding_df())}
    assert "funding_squeeze_continuation" in sigs  # registered in the live slate
    assert sigs["funding_squeeze_continuation"].side == "BUY"


def test_funding_strategy2_holds_without_funding_column():
    """No funding feed (column absent) -> graceful HOLD, never an error."""
    lib = get_skill_library()
    sigs = {s.skill_id: s for s in lib.evaluate_all(_trending_df())}
    assert sigs["funding_squeeze_continuation"].side == "HOLD"
