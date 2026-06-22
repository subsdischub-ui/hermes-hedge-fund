"""Fee selection routes by stage instrument (spot vs perp)."""

from hermes.execution.factory import select_fees


def test_select_fees_routes_by_instrument():
    cfg = {
        "spot_taker": 7,
        "spot_maker": 4,
        "perp_taker": 4.5,
        "perp_maker": 1.5,
        "slippage_bps": 3,
    }
    assert select_fees(cfg, "spot") == (7.0, 3.0)
    assert select_fees(cfg, "perp") == (4.5, 3.0)


def test_select_fees_defaults_to_none_when_missing():
    # Empty/partial config -> None so PaperExchange falls back to its own default.
    assert select_fees({}, "spot") == (None, None)
    assert select_fees({"slippage_bps": 2}, "perp") == (None, 2.0)


def test_select_fees_maker_uses_maker_bps():
    cfg = {
        "spot_taker": 7,
        "spot_maker": 4,
        "perp_taker": 4.5,
        "perp_maker": 1.5,
        "slippage_bps": 3,
    }
    # maker=True picks the (lower) maker fee for the instrument.
    assert select_fees(cfg, "perp", maker=True) == (1.5, 3.0)
    assert select_fees(cfg, "spot", maker=True) == (4.0, 3.0)
    # maker=False (default) keeps taker behaviour.
    assert select_fees(cfg, "perp", maker=False) == (4.5, 3.0)
