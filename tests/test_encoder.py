import numpy as np
import pandas as pd

from hermes.trm.encoder import (
    Action,
    ActionCodec,
    MarketStateEncoder,
    N_SIZE_BUCKETS,
)


def _fake_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    px = 50_000 + rng.standard_normal(n).cumsum() * 50
    high = px + np.abs(rng.standard_normal(n) * 20)
    low = px - np.abs(rng.standard_normal(n) * 20)
    open_ = px + rng.standard_normal(n) * 5
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": px, "volume": vol})


def test_encoder_shape_and_range():
    enc = MarketStateEncoder()
    df = _fake_ohlcv()
    grid = enc.encode(df)
    assert grid.shape == (enc.cfg.state_len,)
    assert grid.min() >= 0
    assert grid.max() < enc.cfg.state_bins


def test_action_roundtrip():
    codec = ActionCodec()
    a = Action(side="BUY", size_pct=0.5, stop_pct=0.02, tp_pct=0.04)
    tokens = codec.encode(a)
    assert tokens.shape == (4,)
    b = codec.decode(tokens)
    assert b.side == "BUY"
    assert abs(b.size_pct - 0.5) <= 1 / (N_SIZE_BUCKETS - 1)
    assert abs(b.stop_pct - 0.02) < 0.01
    assert abs(b.tp_pct - 0.04) < 0.02
