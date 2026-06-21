"""Triple-barrier labeling (López de Prado): from an entry, which barrier is hit
first within the horizon — take-profit (win), stop-loss (loss), or time-out
(labelled by the sign of the final return)."""

import pytest

from hermes.validation.triple_barrier import triple_barrier_label


def test_long_hits_take_profit():
    # rises straight through +5% TP before the -3% stop
    highs = [100, 103, 106]
    lows = [100, 99.5, 104]
    closes = [100, 103, 106]
    out, ret = triple_barrier_label(highs, lows, closes, 0, 100.0, "BUY", 0.03, 0.05, 5)
    assert out == 1
    assert ret == pytest.approx(0.05)


def test_long_hits_stop():
    highs = [100, 101, 101]
    lows = [100, 96, 95]    # -3% stop breached at bar 1
    closes = [100, 97, 96]
    out, ret = triple_barrier_label(highs, lows, closes, 0, 100.0, "BUY", 0.03, 0.05, 5)
    assert out == 0
    assert ret == pytest.approx(-0.03)


def test_long_timeout_labelled_by_final_return():
    highs = [100, 100.5, 101, 101]
    lows = [100, 99.5, 99, 100]
    closes = [100, 100.2, 100.8, 101.0]   # neither barrier hit; ends +1%
    out, ret = triple_barrier_label(highs, lows, closes, 0, 100.0, "BUY", 0.03, 0.05, 3)
    assert out == 1
    assert ret == pytest.approx(0.01, abs=1e-9)


def test_short_hits_take_profit_when_price_falls():
    highs = [100, 100.5, 99]
    lows = [100, 98, 94]     # short TP at -5% reached
    closes = [100, 98, 95]
    out, ret = triple_barrier_label(highs, lows, closes, 0, 100.0, "SELL", 0.03, 0.05, 5)
    assert out == 1
    assert ret == pytest.approx(0.05)


def test_same_bar_stop_and_tp_resolves_to_stop_conservatively():
    highs = [100, 106]      # would hit +5% TP
    lows = [100, 96]        # but also hits -3% stop same bar
    closes = [100, 100]
    out, ret = triple_barrier_label(highs, lows, closes, 0, 100.0, "BUY", 0.03, 0.05, 5)
    assert out == 0
    assert ret == pytest.approx(-0.03)
