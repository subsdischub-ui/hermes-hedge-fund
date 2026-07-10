"""Pure Binance USDT-M helpers (testable without ccxt / credentials / network):
- derive_client_order_id: deterministic clientOrderId so a network retry never
  doubles a position (idempotency) — mirrors hyperliquid.derive_cloid.
- build_reduce_only_legs: reduce-only STOP_MARKET/TAKE_PROFIT_MARKET legs so a
  live position is protected on the exchange even if the agent/host dies —
  mirrors hyperliquid.build_tpsl_orders.
"""

from hermes.execution.binance import build_reduce_only_legs, derive_client_order_id


def test_derive_client_order_id_deterministic_and_well_formed():
    c1 = derive_client_order_id("trade-123")
    assert c1 == derive_client_order_id("trade-123")   # deterministic -> idempotent
    assert c1.startswith("hermes-")
    assert len(c1) <= 36   # Binance clientOrderId hard limit
    assert all(ch in "0123456789abcdef" for ch in c1[len("hermes-"):])
    assert derive_client_order_id("trade-123") != derive_client_order_id("trade-124")


def test_derive_client_order_id_suffix_gives_distinct_stable_ids():
    base = derive_client_order_id("trade-123")
    sl = derive_client_order_id("trade-123", suffix="sl")
    tp = derive_client_order_id("trade-123", suffix="tp")
    # The main order and each reduce-only leg get distinct ids...
    assert len({base, sl, tp}) == 3
    # ...but each one is itself deterministic (idempotent across retries).
    assert sl == derive_client_order_id("trade-123", suffix="sl")
    assert tp == derive_client_order_id("trade-123", suffix="tp")


def test_build_reduce_only_legs_for_long_closes_with_sell():
    legs = build_reduce_only_legs("BUY", qty=0.01, stop_price=90000.0, take_profit=110000.0)
    assert len(legs) == 2
    assert all(leg["side"] == "SELL" and leg["amount"] == 0.01 for leg in legs)
    assert all(leg["params"]["reduceOnly"] is True for leg in legs)
    sl = next(leg for leg in legs if leg["tag"] == "sl")
    tp = next(leg for leg in legs if leg["tag"] == "tp")
    assert sl["params"]["stopLossPrice"] == 90000.0
    assert tp["params"]["takeProfitPrice"] == 110000.0


def test_build_reduce_only_legs_for_short_closes_with_buy():
    legs = build_reduce_only_legs("SELL", qty=0.5, stop_price=4000.0, take_profit=3000.0)
    assert len(legs) == 2
    assert all(leg["side"] == "BUY" for leg in legs)


def test_build_reduce_only_legs_skips_missing_prices():
    legs = build_reduce_only_legs("BUY", qty=1.0, stop_price=None, take_profit=150.0)
    assert len(legs) == 1 and legs[0]["tag"] == "tp"
    assert build_reduce_only_legs("BUY", qty=1.0, stop_price=None, take_profit=None) == []
