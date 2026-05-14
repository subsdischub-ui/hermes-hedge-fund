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
