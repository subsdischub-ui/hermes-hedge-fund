"""Strategy #2 R&D: candidate signal #1 (weekend fade) + the evaluation harness
incl. the orthogonality gate vs momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.research.harness import evaluate_signal, orthogonality, signal_report
from hermes.research.signals import weekend_liquidity_fade
from hermes.skills.library import SkillSignal


def _weekend_df(last_ts: str, prev_up: bool = True, thin: bool = True, n: int = 200) -> pd.DataFrame:
    base = pd.Timestamp(last_ts, tz="UTC")
    ts = [base - pd.Timedelta(hours=4) * (n - 1 - i) for i in range(n)]
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    high = close * 1.001
    low = close * 0.999
    nt = np.full(n, 1000.0)
    # Make bar t-1 an outsized move on a thin book.
    move = 105.0 if prev_up else 95.0
    open_[-2], close[-2] = 100.0, move
    high[-2], low[-2] = max(100.0, move), min(100.0, move)
    nt[-2] = 100.0 if thin else 5000.0
    return pd.DataFrame(
        {"ts": ts, "open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(n, 10.0), "num_trades": nt}
    )


def test_weekend_fade_shorts_outsized_up_move_on_thin_book():
    df = _weekend_df("2020-01-04 00:00", prev_up=True, thin=True)  # Saturday
    sig = weekend_liquidity_fade(df)
    assert sig.side == "SELL"
    assert sig.stop_pct > 0 and sig.take_profit_pct > 0


def test_weekend_fade_buys_outsized_down_move():
    df = _weekend_df("2020-01-04 00:00", prev_up=False, thin=True)
    assert weekend_liquidity_fade(df).side == "BUY"


def test_weekend_fade_holds_off_window():
    df = _weekend_df("2020-01-08 00:00", prev_up=True, thin=True)  # Wednesday
    assert weekend_liquidity_fade(df).side == "HOLD"


def test_weekend_fade_holds_when_book_not_thin():
    df = _weekend_df("2020-01-04 00:00", prev_up=True, thin=False)
    assert weekend_liquidity_fade(df).side == "HOLD"


# ------------------------------- harness ----------------------------------- #
def _always_buy(df: pd.DataFrame) -> SkillSignal:
    return SkillSignal("x", "BUY", 0.6, 0.02, 0.04, "r", {})


def test_evaluate_signal_produces_trades():
    n = 300
    close = 100.0 * (1.01 ** np.arange(n))
    df = pd.DataFrame({
        "ts": pd.to_datetime([pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=4) * i for i in range(n)]),
        "open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
        "volume": np.full(n, 10.0), "num_trades": np.full(n, 1000.0),
    })
    trades = evaluate_signal({"BTC/USDC": df}, _always_buy, horizon=12, warmup=200, no_overlap=True)
    assert len(trades) > 0
    assert all(isinstance(r, float) for _, r in trades)
    rep = signal_report(trades)
    assert rep["n"] == len(trades)


def test_orthogonality_identical_and_opposite():
    d = [pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i) for i in range(5)]
    ns = [int(x.value) for x in d]
    a = [(ns[0], 1.0), (ns[1], -1.0), (ns[2], 2.0), (ns[3], -0.5), (ns[4], 1.5)]
    assert orthogonality(a, a) > 0.99
    b = [(t, -r) for t, r in a]
    assert orthogonality(a, b) < -0.99


def test_orthogonality_empty_is_zero():
    assert orthogonality([], [(1, 1.0)]) == 0.0


# ----------------------------- signal #2 VR -------------------------------- #
def test_variance_ratio_random_walk_near_one():
    from hermes.research.signals import variance_ratio

    rng = np.random.default_rng(0)
    r = rng.standard_normal(2000) * 0.01  # iid -> VR ~ 1
    vr, _ = variance_ratio(r, 4)
    assert 0.7 < vr < 1.3


def test_variance_ratio_mean_reverting_below_threshold():
    from hermes.research.signals import variance_ratio

    r = np.array([0.01, -0.01] * 600)  # perfectly alternating -> strong mean reversion
    vr, z = variance_ratio(r, 4)
    assert vr < 0.8
    assert z < -1.64


def test_vr_antitrend_fade_shorts_stretched_meanreverting():
    from hermes.research.signals import vr_antitrend_fade

    closes = [100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(245)]
    closes[-1] = 104.0  # stretched up at the end
    df = pd.DataFrame({"close": closes})
    assert vr_antitrend_fade(df).side == "SELL"


def test_vr_antitrend_fade_holds_in_trend():
    from hermes.research.signals import vr_antitrend_fade

    rng = np.random.default_rng(3)
    # Random walk -> VR ≈ 1 (neither mean-reverting nor trending) -> must NOT fire.
    closes = 100.0 * np.exp(np.cumsum(rng.standard_normal(245) * 0.01))
    assert vr_antitrend_fade(pd.DataFrame({"close": closes})).side == "HOLD"


# ----------------------------- signal #4 flow ------------------------------ #
def _flow_df(n=160, stretch_up=True, ofi_extreme=True):
    rng = np.random.default_rng(1)
    close = np.full(n, 100.0) + rng.standard_normal(n) * 0.2
    close[-1] = 103.0 if stretch_up else 97.0  # stretched last bar
    vol = np.full(n, 1000.0)
    nt = np.full(n, 500.0)
    tbb = np.full(n, 500.0)  # ofi = 500/1000-0.5 = 0 baseline
    if ofi_extreme:
        tbb[-1] = 950.0 if stretch_up else 50.0  # extreme aggressor side
        vol[-1] = 1000.0
        nt[-1] = 200.0  # larger avg trade size -> high atsz
    return pd.DataFrame({"close": close, "volume": vol, "taker_buy_base": tbb, "num_trades": nt})


def test_flow_exhaustion_fade_shorts_unconfirmed_up_extension():
    from hermes.research.signals import flow_exhaustion_fade

    assert flow_exhaustion_fade(_flow_df(stretch_up=True, ofi_extreme=True)).side == "SELL"


def test_flow_exhaustion_fade_holds_without_flow_extreme():
    from hermes.research.signals import flow_exhaustion_fade

    assert flow_exhaustion_fade(_flow_df(stretch_up=True, ofi_extreme=False)).side == "HOLD"


# ----------------------------- funding ------------------------------------- #
def test_attach_funding_asof_backward_no_lookahead():
    from hermes.research.funding import attach_funding

    ohlcv = pd.DataFrame({"ts": [100, 200, 300], "close": [1.0, 2.0, 3.0]})
    funding = pd.DataFrame({"ts": [150, 250], "funding_rate": [0.01, -0.02]})
    out = attach_funding(ohlcv, funding)
    assert list(out["funding_rate"]) == [0.0, 0.01, -0.02]  # bar 100 pre-funding -> 0


def test_funding_extreme_fade_shorts_high_funding():
    from hermes.research.signals import funding_extreme_fade

    n = 200
    rng = np.random.default_rng(2)
    fr = rng.standard_normal(n) * 0.0001
    fr[-1] = 0.02  # extreme positive funding -> crowd long -> fade SHORT
    close = np.full(n, 100.0) + rng.standard_normal(n) * 0.1
    df = pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99,
                       "funding_rate": fr})
    assert funding_extreme_fade(df).side == "SELL"


def test_funding_extreme_fade_holds_when_neutral():
    from hermes.research.signals import funding_extreme_fade

    n = 200
    rng = np.random.default_rng(5)
    fr = rng.standard_normal(n) * 0.0001  # no extreme
    close = np.full(n, 100.0) + rng.standard_normal(n) * 0.1
    df = pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99,
                       "funding_rate": fr})
    assert funding_extreme_fade(df).side == "HOLD"
