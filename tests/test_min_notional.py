"""Venue-aware minimum notional: Hyperliquid rejects orders < $10, so an order
sized between $5 and $10 that the guardrail "approved" would zero-fill on the
exchange. Reject it upfront with the right per-venue floor."""

from hermes.risk.guardrails import venue_min_notional


def test_venue_min_notional():
    assert venue_min_notional("paper") == 5.0
    assert venue_min_notional("hyperliquid") == 10.0
    assert venue_min_notional("bybit") == 5.0
    assert venue_min_notional("binanceusdm") == 5.0
    assert venue_min_notional("unknown") == 10.0  # conservative default for live venues
