"""TDD for the shared illiquid-tail universe selector (Stream 0-β).

The selector encodes the universe policy for Boucle β: maximize liquidity VARIANCE
across the whole spectrum (NOT pick the most-liquid), deliberately over-sampling the
illiquid bottom because that is where H1 predicts carry persists. Pure + deterministic
(no RNG) so the chosen slice is reproducible.
"""
from __future__ import annotations

from hermes.datalake.sources import tail_select as ts


def _rows(n, *, key="symbol"):
    # liquidity 1..n (strictly increasing, positive) so deciles are unambiguous.
    return [{key: f"C{i:03d}", "vol": float(i)} for i in range(1, n + 1)]


def test_returns_n_target_when_enough_candidates():
    out = ts.stratified_decile_sample(_rows(100), liquidity_key="vol", n_target=30)
    assert len(out) == 30


def test_returns_all_when_fewer_candidates_than_target():
    out = ts.stratified_decile_sample(_rows(12), liquidity_key="vol", n_target=40)
    assert len(out) == 12


def test_oversamples_the_illiquid_bottom():
    # ~40% of picks must come from the bottom 3 deciles (the H1-bearing illiquid end).
    out = ts.stratified_decile_sample(
        _rows(200), liquidity_key="vol", n_target=40,
        n_deciles=10, bottom_deciles=3, bottom_fraction=0.4,
    )
    bottom = [r for r in out if r["decile"] < 3]
    assert 14 <= len(bottom) <= 18  # ~16 of 40 = 40%, with rounding tolerance


def test_spans_the_whole_spectrum_not_just_one_end():
    out = ts.stratified_decile_sample(_rows(200), liquidity_key="vol", n_target=40)
    deciles = {r["decile"] for r in out}
    # must reach both the illiquid floor (decile 0) AND the liquid top (decile 9)
    assert 0 in deciles
    assert 9 in deciles
    assert len(deciles) >= 8  # broad coverage, not clustered


def test_annotates_each_selected_row_with_its_decile():
    out = ts.stratified_decile_sample(_rows(50), liquidity_key="vol", n_target=20)
    for r in out:
        assert "decile" in r and 0 <= r["decile"] <= 9


def test_is_deterministic():
    a = ts.stratified_decile_sample(_rows(150), liquidity_key="vol", n_target=33)
    b = ts.stratified_decile_sample(_rows(150), liquidity_key="vol", n_target=33)
    assert [r["symbol"] for r in a] == [r["symbol"] for r in b]


def test_no_duplicate_selections():
    out = ts.stratified_decile_sample(_rows(120), liquidity_key="vol", n_target=37)
    syms = [r["symbol"] for r in out]
    assert len(syms) == len(set(syms))


def test_empty_input_returns_empty():
    assert ts.stratified_decile_sample([], liquidity_key="vol", n_target=10) == []
