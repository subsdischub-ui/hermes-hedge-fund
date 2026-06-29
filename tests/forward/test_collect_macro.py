"""FORWARD — macro forward collection: SEP dot-plot vs Fed-Funds-futures path.

S7.3 named the dot-plot-vs-FF gap as a forward-collectable frontier: the Fed's SEP
median projected policy path versus the market-implied path from Fed-Funds futures.
The gap (Fed minus market) is a PIT measure of policy disagreement; accruing it daily
opens the frontier for free. Only the matching/gap transform is unit-tested.
"""

from __future__ import annotations

from hermes.forward import collect_macro as mc

_NOW = 1_780_000_000_000


def test_gap_matches_horizons_and_signs():
    sep = [{"horizon": "2026", "median": 3.9}, {"horizon": "2027", "median": 3.4}]
    ff = [{"horizon": "2026", "implied": 3.6}, {"horizon": "2027", "implied": 3.5}]
    out = mc.dot_vs_ff_gap(sep, ff, _NOW)
    g = dict(zip(out["horizon"], out["gap"]))
    assert abs(g["2026"] - 0.3) < 1e-9   # Fed above market → hawkish gap
    assert abs(g["2027"] - (-0.1)) < 1e-9  # Fed below market → dovish gap
    assert (out["ts"] == _NOW).all()


def test_unmatched_horizons_dropped():
    sep = [{"horizon": "2026", "median": 3.9}, {"horizon": "2028", "median": 3.0}]
    ff = [{"horizon": "2026", "implied": 3.6}, {"horizon": "2027", "implied": 3.5}]
    out = mc.dot_vs_ff_gap(sep, ff, _NOW)
    assert list(out["horizon"]) == ["2026"]  # only the common horizon survives


def test_empty_inputs_yield_empty_frame():
    assert mc.dot_vs_ff_gap([], [], _NOW).empty


def test_collect_is_fail_soft():
    def fetch_sep():
        return [{"horizon": "2026", "median": 3.9}]

    def fetch_ff():
        raise RuntimeError("CME FedWatch unreachable")

    out = mc.collect(fetch_sep=fetch_sep, fetch_ff=fetch_ff, now_ms=_NOW)
    assert out["gap"].empty
    assert "ff" in out["blockers"]
