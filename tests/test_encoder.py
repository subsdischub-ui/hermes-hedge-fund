import numpy as np
import pandas as pd

from hermes.trm.encoder import (
    Action,
    ActionCodec,
    MarketStateEncoder,
    N_SIZE_BUCKETS,
    STATE_BASE,
    VOCAB_SIZE,
)


def _fake_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    px = 50_000 + rng.standard_normal(n).cumsum() * 50
    high = px + np.abs(rng.standard_normal(n) * 20)
    low = px - np.abs(rng.standard_normal(n) * 20)
    open_ = px + rng.standard_normal(n) * 5
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": px, "volume": vol})


def test_encoder_state_tokens_in_disjoint_region():
    """State tokens MUST be in [STATE_BASE, VOCAB_SIZE) so they don't collide
    with PAD_ID=0 or SIDE_HOLD=1 in the action vocab."""
    enc = MarketStateEncoder()
    df = _fake_ohlcv()
    grid = enc.encode(df)
    assert grid.shape == (enc.cfg.state_len,)
    assert grid.min() >= STATE_BASE, f"state token below STATE_BASE: {grid.min()}"
    assert grid.max() < VOCAB_SIZE, f"state token >= VOCAB_SIZE: {grid.max()}"


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


def test_small_stop_pct_not_silently_inflated():
    """A small stop_pct (0.1%) should round to the closest log-spaced bucket,
    not to a value 10x its magnitude."""
    codec = ActionCodec()
    a = Action(side="BUY", size_pct=0.1, stop_pct=0.001, tp_pct=0.002)
    decoded = codec.decode(codec.encode(a))
    # 0.001 is below the lowest bucket (0.0005); the nearest in log space is 0.0005.
    assert decoded.stop_pct < 0.005, f"stop_pct silently inflated to {decoded.stop_pct}"
