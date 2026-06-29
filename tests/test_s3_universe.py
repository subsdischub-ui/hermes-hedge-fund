"""Stream 3 — universe expansion + per-instrument calibration.

TDD for ``hermes.research.s3_universe``. The unit under test is the *generalization
+ calibration* machinery applied to the ALREADY-VALIDATED S2 sleeve
(``funding_squeeze_continuation``) — NOT a new signal. Tests pin:

* causal as-of construction of a per-pair frame from the frozen lake schema,
* that the calibration probe ``funding_squeeze_at_z`` is byte-identical to the
  frozen S2 rule at its frozen (z=2.0, window=180) point — i.e. no rule change,
* the per-asset random-entry MC-null (determinism + power to detect / reject),
* the dated joint book-null,
* the bear/bull drift-beta diagnostic,
* that my richer trade walk is EQUIVALENT to the committed ``evaluate_signal``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.research.harness import evaluate_signal
from hermes.research.signals import funding_squeeze_continuation
from hermes.research.s3_universe import (
    arm_threshold_native,
    bear_bull_split,
    build_pair_frame,
    calibration_row,
    funding_squeeze_at_z,
    joint_dated_null,
    normalize_symbol,
    per_asset_null_pvalue,
    walk_trades,
)


# --------------------------------------------------------------------------- #
# synthetic frame builders
# --------------------------------------------------------------------------- #
def _flat_df(n: int = 400, px: float = 100.0) -> pd.DataFrame:
    ts = (np.arange(n) * 86_400_000 + 1_600_000_000_000).astype("int64")  # daily ms
    close = np.full(n, px)
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, unit="ms", utc=True),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 10.0),
            "funding_rate": np.zeros(n),
            "symbol": "BTCUSDT",
        }
    )


def _spike_df(
    n: int = 400,
    spikes: tuple[int, ...] = (210, 240, 270, 300, 330, 360),
    horizon: int = 4,
    drift: float = 0.02,
) -> pd.DataFrame:
    """Flat tape with funding spikes; price drifts UP for ``horizon`` bars after
    each spike (the planted continuation edge). ``drift``=0 ⇒ no edge."""
    df = _flat_df(n)
    fr = np.full(n, 0.0001)  # tiny positive baseline so std>0
    close = df["close"].to_numpy(dtype=float)
    for s in spikes:
        fr[s] = 0.05  # a single outlier vs the ~0 baseline ⇒ funding_z ≫ 2
        for k in range(1, horizon + 1):
            if s + k < n:
                close[s + k] = close[s + k - 1] * (1 + drift)
    df["funding_rate"] = fr
    df["close"] = close
    df["open"] = close
    df["high"] = close * 1.001
    df["low"] = close * 0.999
    return df


def _noisy_spike_df(seed: int, n: int = 400, drift: float = 0.02) -> pd.DataFrame:
    """A spike df with a noisy price path so per-trade returns have real dispersion
    (a degenerate zero-variance edge gives Sharpe 0 and tests nothing)."""
    rng = np.random.default_rng(seed)
    df = _spike_df(n=n, drift=drift)
    close = df["close"].to_numpy(dtype=float) * (1 + rng.normal(0, 0.002, n))
    df["close"] = close
    df["open"] = close
    df["high"] = close * 1.002
    df["low"] = close * 0.998
    return df


def _moderate_z_window(z_sigmas: float = 3.0, n: int = 200, seed: int = 7) -> pd.DataFrame:
    """A 200-bar frame whose final funding bar sits ~``z_sigmas`` above a NOISY
    baseline ⇒ funding_z ≈ z_sigmas (not the ~13 a single outlier vs a flat
    baseline produces) — so a threshold sweep can actually straddle it."""
    rng = np.random.default_rng(seed)
    df = _flat_df(n)
    fr = rng.normal(0.0, 0.01, n)
    fr[-1] = fr.mean() + z_sigmas * fr.std()
    df["funding_rate"] = fr
    return df


# --------------------------------------------------------------------------- #
# normalize_symbol — funding ('BTCUSDT') vs klines ('BTC/USDT') join key
# --------------------------------------------------------------------------- #
def test_normalize_symbol_unifies_slash_and_no_slash():
    assert normalize_symbol("BTC/USDT") == normalize_symbol("BTCUSDT") == "BTCUSDT"
    assert normalize_symbol("avax/usdt") == "AVAXUSDT"


# --------------------------------------------------------------------------- #
# build_pair_frame — causal as-of merge of funding onto klines
# --------------------------------------------------------------------------- #
def test_build_pair_frame_is_causal_and_normalizes_symbol():
    klines = pd.DataFrame(
        {
            "ts": [1000, 2000, 3000, 4000],
            "open": [10, 11, 12, 13],
            "high": [10, 11, 12, 13],
            "low": [10, 11, 12, 13],
            "close": [10, 11, 12, 13],
            "volume": [1, 1, 1, 1],
            "symbol": ["BTC/USDT"] * 4,
        }
    )
    funding = pd.DataFrame({"ts": [2500], "close": [0.01], "symbol": ["BTCUSDT"]})
    out = build_pair_frame(klines, funding)
    # bars at 1000/2000 are BEFORE the first funding settlement -> neutral 0.0;
    # bars at 3000/4000 inherit the 2500 settlement (never a future one).
    assert list(out["funding_rate"]) == [0.0, 0.0, 0.01, 0.01]
    assert set(out["symbol"]) == {"BTCUSDT"}
    assert pd.api.types.is_datetime64_any_dtype(out["ts"])


# --------------------------------------------------------------------------- #
# funding_squeeze_at_z — calibration probe == frozen S2 rule at its frozen point
# --------------------------------------------------------------------------- #
def test_calibration_probe_is_identical_to_frozen_rule_at_z2():
    df = _spike_df()
    window = df.iloc[:211]  # ends on a spike bar -> arms
    a = funding_squeeze_at_z(window, z_thresh=2.0, window=180)
    b = funding_squeeze_continuation(window)
    assert (a.side, a.confidence, a.stop_pct, a.take_profit_pct) == (
        b.side,
        b.confidence,
        b.stop_pct,
        b.take_profit_pct,
    )
    assert a.side == "BUY"


def test_calibration_probe_higher_threshold_arms_less_often():
    # a window whose final funding bar is ~3σ above a noisy baseline (z≈3):
    # arms at the lower threshold, holds at the higher one.
    win = _moderate_z_window(z_sigmas=3.0)
    lo = funding_squeeze_at_z(win, z_thresh=1.5, window=180)
    hi = funding_squeeze_at_z(win, z_thresh=4.0, window=180)
    assert lo.side == "BUY"
    assert hi.side == "HOLD"


# --------------------------------------------------------------------------- #
# walk_trades — EQUIVALENT to the committed harness (faithful reuse, not a rebuild)
# --------------------------------------------------------------------------- #
def test_walk_trades_matches_committed_evaluate_signal():
    df = _spike_df()
    recs = walk_trades(
        df, funding_squeeze_continuation, horizon=4, fee_bps=7.0, slippage_bps=3.0
    )
    mine = [(r["event_ns"], round(r["ret"], 12)) for r in recs]
    theirs = [
        (ev, round(r, 12))
        for ev, r in evaluate_signal(
            {"BTCUSDT": df}, funding_squeeze_continuation, horizon=4
        )
    ]
    assert mine == theirs
    assert len(recs) >= 3  # the planted spikes fired


# --------------------------------------------------------------------------- #
# per_asset_null_pvalue — random-entry MC null (the §3 floor)
# --------------------------------------------------------------------------- #
def test_per_asset_null_detects_planted_edge():
    df = _spike_df(drift=0.02)
    out = per_asset_null_pvalue(
        df,
        funding_squeeze_continuation,
        horizon=4,
        n_boot=500,
        rng=np.random.default_rng(0),
    )
    assert out["n_trades"] >= 3
    assert out["obs_mean"] > out["null_mean"]
    assert out["p_value"] < 0.05


def test_per_asset_null_rejects_no_edge():
    df = _spike_df(drift=0.0)  # spikes arm the sleeve but there is NO continuation
    out = per_asset_null_pvalue(
        df,
        funding_squeeze_continuation,
        horizon=4,
        n_boot=500,
        rng=np.random.default_rng(0),
    )
    assert out["p_value"] > 0.05


def test_per_asset_null_is_deterministic_under_seed():
    df = _spike_df(drift=0.02)
    kw = dict(horizon=4, n_boot=200)
    a = per_asset_null_pvalue(
        df, funding_squeeze_continuation, rng=np.random.default_rng(42), **kw
    )
    b = per_asset_null_pvalue(
        df, funding_squeeze_continuation, rng=np.random.default_rng(42), **kw
    )
    assert a == b


# --------------------------------------------------------------------------- #
# bear_bull_split — drift-beta diagnostic (is the edge just secular long-beta?)
# --------------------------------------------------------------------------- #
def test_bear_bull_split_classifies_by_ma200():
    df = _spike_df(drift=0.02)
    recs = walk_trades(df, funding_squeeze_continuation, horizon=4)
    out = bear_bull_split(recs)
    assert out["n_bull"] + out["n_bear"] == len(recs)
    # the planted tape is a rising series sitting above its own MA200 -> bull-side
    assert out["n_bull"] >= 1


# --------------------------------------------------------------------------- #
# arm_threshold_native — what z=2.0 means in NATIVE funding units, per instrument
# --------------------------------------------------------------------------- #
def test_arm_threshold_native_is_mean_plus_z_sigma():
    vals = np.array([0.001, 0.002, 0.003, 0.004, 0.005] * 40, dtype=float)  # 200 obs
    df = _flat_df(n=len(vals))
    df["funding_rate"] = vals
    thr = arm_threshold_native(df, z=2.0, window=180)
    tail = vals[-180:]
    assert abs(thr - (tail.mean() + 2.0 * tail.std(ddof=0))) < 1e-9


# --------------------------------------------------------------------------- #
# calibration_row — IS/OOS table; lower z arms more often (the whipsaw risk)
# --------------------------------------------------------------------------- #
def test_calibration_row_lower_threshold_yields_more_trades_oos():
    df = _spike_df(drift=0.01, spikes=tuple(range(205, 395, 8)))
    hi = calibration_row(df, z_thresh=3.0, split_frac=0.6, horizon=4)
    lo = calibration_row(df, z_thresh=1.0, split_frac=0.6, horizon=4)
    assert lo["oos"]["n"] >= hi["oos"]["n"]
    for col in ("n", "win_rate", "sharpe"):
        assert col in lo["oos"] and col in lo["is"]


# --------------------------------------------------------------------------- #
# joint_dated_null — book-level dated null over the ensemble of survivors
# --------------------------------------------------------------------------- #
def test_joint_dated_null_detects_real_book_edge_and_is_deterministic():
    dfs = {f"A{i}": _noisy_spike_df(seed=i, drift=0.02) for i in range(3)}
    kw = dict(
        signal_fn=funding_squeeze_continuation, horizon=4, n_boot=300, weighting="equal"
    )
    out = joint_dated_null(dfs, rng=np.random.default_rng(1), **kw)
    again = joint_dated_null(dfs, rng=np.random.default_rng(1), **kw)
    assert out == again
    assert out["obs_sharpe"] > 0
    assert out["p_value"] < 0.05
