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
