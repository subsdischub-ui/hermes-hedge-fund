"""Fee selection routes by stage instrument (spot vs perp)."""

from hermes.execution.factory import get_exchange, select_fees
from hermes.settings import get_settings


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


def test_stage0_paper_charges_taker_not_maker_by_default():
    """Regression guard: maker_orders once defaulted to true, so get_exchange() fed
    select_fees(maker=True) for EVERY selective-mode order (not just the structurally-
    hedged carry pair) — silently undercharging ~9bps RT instead of ~15bps RT, directly
    contradicting fees:'s own documented intent ("the paper exchange mirrors the taker
    fee ... under-charging here would open the gates on a fictional edge"). No fill-rate
    model ever backed the maker assumption. Pin that the real settings produce a taker
    fee_bps, so a future config edit can't silently reopen this gap."""
    get_settings.cache_clear()
    get_exchange.cache_clear()
    try:
        s = get_settings()
        assert s.yaml("strategy", default={}).get("maker_orders") is False
        px = get_exchange()
        fees_cfg = s.yaml("fees", default={})
        assert px.fee_bps == fees_cfg["perp_taker"]
    finally:
        get_settings.cache_clear()
        get_exchange.cache_clear()
