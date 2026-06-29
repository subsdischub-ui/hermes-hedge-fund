"""TDD for Stream 7 — central-bank tone DRIFT (not the print).

The LLM call is I/O (injected in production via hermes.llm.router); everything tested here
is the PURE signal/gate machinery. The keystone test is causality: the trailing z-score at
time t must not move when FUTURE values change — the unit-test embodiment of "lague tout".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.research import s7_cb_tone as s7


# --------------------------------------------------------------------------- #
# classifier: parse + clamp + fail-soft (the LLM is injected, never called here)
# --------------------------------------------------------------------------- #
def test_parse_tone_reads_a_well_formed_score():
    ts = s7.parse_tone({"score": 0.6, "confidence": 0.8, "stance": "hawkish"}, "abc")
    assert ts.id == "abc"
    assert ts.score == 0.6
    assert ts.confidence == 0.8
    assert ts.stance == "hawkish"


def test_parse_tone_clamps_score_into_unit_interval():
    assert s7.parse_tone({"score": 4.2, "stance": "hawkish"}, "x").score == 1.0
    assert s7.parse_tone({"score": -9.0, "stance": "dovish"}, "y").score == -1.0


def test_parse_tone_failsoft_to_neutral_zero_on_garbage():
    ts = s7.parse_tone({"oops": True}, "z")
    assert ts.score == 0.0
    assert ts.confidence == 0.0
    assert ts.stance == "neutral"


def test_classify_one_uses_the_injected_scorer():
    # the LLM is a pure injected callable here — no network, no router
    captured = {}

    def fake_scorer(excerpt: str) -> dict:
        captured["excerpt"] = excerpt
        return {"score": -0.7, "confidence": 0.9, "stance": "dovish"}

    ts = s7.classify_one("a speech about slack and easing", "id7", score_fn=fake_scorer)
    assert captured["excerpt"] == "a speech about slack and easing"
    assert ts.id == "id7"
    assert ts.score == -0.7
    assert ts.stance == "dovish"


# --------------------------------------------------------------------------- #
# daily_tone: collapse many speeches on a day to one confidence-weighted mean
# --------------------------------------------------------------------------- #
def test_daily_tone_handles_empty_input_without_crashing():
    out = s7.daily_tone(pd.DataFrame(columns=["date", "bank", "score", "confidence"]))
    assert list(out.columns) == ["date", "bank", "tone", "n"]
    assert len(out) == 0


def test_daily_tone_confidence_weighted_mean_per_bank_day():
    df = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-02", "2020-01-03"],
            "bank": ["FED", "FED", "FED"],
            "score": [1.0, -1.0, 0.5],
            "confidence": [0.9, 0.1, 1.0],  # the +1.0 dominates 2020-01-02
        }
    )
    out = s7.daily_tone(df)
    d0 = out[(out["bank"] == "FED") & (out["date"] == pd.Timestamp("2020-01-02"))]["tone"].iloc[0]
    assert d0 == pytest.approx((1.0 * 0.9 + -1.0 * 0.1) / (0.9 + 0.1))  # = 0.8


# --------------------------------------------------------------------------- #
# ewma_drift: slow, recency-weighted drift
# --------------------------------------------------------------------------- #
def test_ewma_drift_tracks_a_sustained_shift():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    s = pd.Series([-0.5] * 100 + [0.5] * 100, index=idx)
    drift = s7.ewma_drift(s, halflife_days=10)
    assert drift.iloc[90] < -0.2   # still dovish in the first regime
    assert drift.iloc[-1] > 0.2    # converged hawkish after the shift
    assert len(drift) == len(s)


# --------------------------------------------------------------------------- #
# causal_z: THE keystone — no look-ahead
# --------------------------------------------------------------------------- #
def test_causal_z_is_strictly_backward_looking():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=120), index=idx)

    z_full = s7.causal_z(base, window=30, min_obs=10)

    # mutate the FUTURE (everything strictly after position 60) and recompute
    poisoned = base.copy()
    poisoned.iloc[61:] = 999.0
    z_poison = s7.causal_z(poisoned, window=30, min_obs=10)

    # z at t=60 must be identical: it may only use data up to and including t=60
    assert z_full.iloc[60] == pytest.approx(z_poison.iloc[60], nan_ok=True)


def test_causal_z_is_nan_until_min_obs():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    s = pd.Series(np.arange(20.0), index=idx)
    z = s7.causal_z(s, window=30, min_obs=10)
    assert z.iloc[:9].isna().all()       # < min_obs -> undefined
    assert not np.isnan(z.iloc[-1])


# --------------------------------------------------------------------------- #
# tone_events: fire on |z|>=thr with the right direction
# --------------------------------------------------------------------------- #
def test_tone_events_fire_on_threshold_with_sign():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    z = pd.Series([0.0, 1.5, 0.2, -2.0, np.nan], index=idx)
    evs = s7.tone_events(z, thr=1.0)
    dirs = {pd.Timestamp(ms, unit="ms").normalize(): d for ms, d in evs}
    assert dirs[pd.Timestamp("2020-01-02")] == 1     # hawkish drift
    assert dirs[pd.Timestamp("2020-01-04")] == -1    # dovish drift
    assert pd.Timestamp("2020-01-03") not in dirs    # below threshold
    assert pd.Timestamp("2020-01-05") not in dirs    # NaN never fires


# --------------------------------------------------------------------------- #
# drop_blackout: never hold a position through a policy print
# --------------------------------------------------------------------------- #
def test_drop_blackout_removes_entries_whose_hold_straddles_a_meeting():
    base = pd.Timestamp("2020-03-01").value // 1_000_000  # ms
    events = [(base, 1)]                       # enters 2020-03-01, holds 'horizon' days
    meeting = pd.Timestamp("2020-03-10").value // 1_000_000
    # horizon 15 days -> hold window covers 2020-03-10 -> dropped
    assert s7.drop_blackout(events, [meeting], horizon_days=15) == []
    # horizon 5 days -> hold ends 2020-03-06, before the meeting -> kept
    assert s7.drop_blackout(events, [meeting], horizon_days=5) == events


def test_drop_blackout_keeps_entries_with_no_nearby_meeting():
    base = pd.Timestamp("2020-06-01").value // 1_000_000
    events = [(base, -1)]
    far = pd.Timestamp("2021-01-01").value // 1_000_000
    assert s7.drop_blackout(events, [far], horizon_days=20) == events


# --------------------------------------------------------------------------- #
# bond_proxy: a monotone price for the rates channel (hawkish -> yields up -> proxy down)
# --------------------------------------------------------------------------- #
def test_bond_proxy_falls_when_yield_rises():
    y = pd.Series([2.0, 3.0, 4.0])
    p = s7.bond_proxy(y)
    assert (p.diff().dropna() < 0).all()   # strictly falling as yields rise


# --------------------------------------------------------------------------- #
# 7.3 conditioner: dot-plot vs Fed-funds-futures disagreement (data-to-collect)
# --------------------------------------------------------------------------- #
def test_fed_market_disagreement_is_dots_minus_market_aligned():
    idx = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30"])
    dots = pd.Series([4.6, 4.1, 3.9], index=idx)        # SEP median projected fed funds
    mkt = pd.Series([4.5, 4.4, 4.2], index=idx)         # FF-futures implied for same horizon
    gap = s7.fed_market_disagreement(dots, mkt)
    # positive gap = Fed projects HIGHER than the market prices (Fed more hawkish than market)
    assert gap.iloc[0] == pytest.approx(0.1)
    assert gap.iloc[1] == pytest.approx(-0.3)


def test_fed_market_disagreement_nan_where_either_side_missing():
    a = pd.Series([4.0, np.nan], index=pd.to_datetime(["2024-03-31", "2024-06-30"]))
    b = pd.Series([3.8, 4.1], index=pd.to_datetime(["2024-03-31", "2024-06-30"]))
    gap = s7.fed_market_disagreement(a, b)
    assert gap.iloc[0] == pytest.approx(0.2)
    assert np.isnan(gap.iloc[1])
