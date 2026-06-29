"""Tests for the v-next provenance guardrails (src/hermes/research/s10_provenance.py).

Two reusable checks that catch the silent data-provenance traps S0 surfaced:
  P1  funding-source check: a backtest that reads the SHALLOW ``binance_funding`` stub
      (~200 rows, forward-only — the /fapi/v1/fundingRate startTime=0 clamp) instead of the DEEP
      ``binance_funding_native`` view is silently training on 2 months of data → flag it.
  P2  symbol-format join validation: the lake mixes 'BTC/USDT' (binance_klines/native),
      bare 'BTC' (coinglass_multi_*) and 'BTCUSDT' (coinglass_funding_rate, binance_funding_info);
      a cross-format join silently produces zero rows → flag/normalize before joining.
"""

from __future__ import annotations

import pytest

from hermes.research import s10_provenance as prov


# --------------------------------------------------------------------------- #
# P1 — shallow vs deep funding view
# --------------------------------------------------------------------------- #
def test_check_funding_provenance_flags_shallow_stub():
    res = prov.check_funding_provenance(
        "SELECT ts, funding_rate FROM binance_funding WHERE symbol = 'BTC/USDT'"
    )
    assert res["flagged"] is True
    assert "binance_funding" in res["shallow_views_found"]
    assert "binance_funding_native" in res["recommendation"]


def test_deep_native_view_is_not_flagged():
    for sql in (
        "SELECT * FROM binance_funding_native WHERE symbol='BTC/USDT'",
        "SELECT * FROM binance_funding_on_4h",
        "SELECT * FROM coinglass_funding_rate",  # a different deep source, must not false-positive
        "SELECT * FROM coinglass_multi_funding_rate",
    ):
        res = prov.check_funding_provenance(sql)
        assert res["flagged"] is False, sql
        assert res["shallow_views_found"] == []


def test_native_substring_does_not_false_positive():
    """The shallow check must match the bare 'binance_funding' token, NOT the deep views that
    merely contain it as a prefix."""
    res = prov.check_funding_provenance(
        "SELECT * FROM binance_funding_native n JOIN binance_funding_info i USING(symbol)"
    )
    assert res["flagged"] is False


def test_assert_deep_funding_raises_on_stub():
    with pytest.raises(prov.ProvenanceError, match="binance_funding_native"):
        prov.assert_deep_funding("SELECT funding_rate FROM binance_funding")


# --------------------------------------------------------------------------- #
# P2 — symbol-format join validation
# --------------------------------------------------------------------------- #
def test_detect_symbol_format():
    assert prov.detect_symbol_format("BTC/USDT") == "slash"
    assert prov.detect_symbol_format("BTC") == "bare"
    assert prov.detect_symbol_format("BTCUSDT") == "concat"
    assert prov.detect_symbol_format("WIFUSDT") == "concat"


def test_cross_format_join_is_flagged():
    """binance_klines_4h ('BTC/USDT') joined against coinglass_multi_* (bare 'BTC') silently yields
    zero overlap — the formats are incompatible and must be normalized first."""
    left = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]   # binance_klines / native
    right = ["BTC", "ETH", "SOL"]                  # coinglass_multi_*
    res = prov.validate_symbol_join(left, right)
    assert res["left_format"] == "slash"
    assert res["right_format"] == "bare"
    assert res["compatible"] is False
    assert res["n_overlap"] == 0
    assert "normaliz" in res["recommendation"].lower()


def test_same_format_join_is_compatible():
    res = prov.validate_symbol_join(["BTC/USDT", "ETH/USDT"], ["BTC/USDT", "SOL/USDT"])
    assert res["compatible"] is True
    assert res["n_overlap"] == 1  # BTC/USDT


def test_normalize_symbol_between_formats():
    assert prov.normalize_symbol("BTC/USDT", "bare") == "BTC"
    assert prov.normalize_symbol("BTC", "slash") == "BTC/USDT"
    assert prov.normalize_symbol("BTCUSDT", "bare") == "BTC"
    assert prov.normalize_symbol("BTC", "concat") == "BTCUSDT"
    assert prov.normalize_symbol("BTC/USDT", "concat") == "BTCUSDT"


def test_normalized_join_becomes_compatible():
    left = ["BTC/USDT", "ETH/USDT"]
    right = ["BTC", "ETH"]
    norm_left = [prov.normalize_symbol(s, "bare") for s in left]
    res = prov.validate_symbol_join(norm_left, right)
    assert res["compatible"] is True
    assert res["n_overlap"] == 2


def test_assert_joinable_raises_on_cross_format():
    with pytest.raises(prov.ProvenanceError):
        prov.assert_joinable(["BTC/USDT"], ["BTC"])
