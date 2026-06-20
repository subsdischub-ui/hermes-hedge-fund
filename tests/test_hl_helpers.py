"""Pure Hyperliquid helpers (testable without the SDK / credentials / network):
- derive_cloid: deterministic client-order-id so a network retry never doubles a
  position (idempotency).
- build_tpsl_orders: reduce-only stop/take-profit trigger orders so a live
  position is protected on the exchange even if the agent/host dies.
"""

from hermes.execution.hyperliquid import build_tpsl_orders, derive_cloid


def test_derive_cloid_deterministic_and_well_formed():
    c1 = derive_cloid("trade-123")
    assert c1 == derive_cloid("trade-123")          # deterministic -> idempotent
    assert c1.startswith("0x") and len(c1) == 34     # 0x + 128-bit hex
    assert all(ch in "0123456789abcdef" for ch in c1[2:])
    assert derive_cloid("trade-123") != derive_cloid("trade-124")


def test_build_tpsl_orders_for_long_closes_with_sell():
    orders = build_tpsl_orders("BTC", is_long=True, qty=0.01, stop_px=90000.0, tp_px=110000.0)
    assert len(orders) == 2
    assert all(o["coin"] == "BTC" and o["is_buy"] is False and o["reduce_only"] for o in orders)
    assert all(o["sz"] == 0.01 for o in orders)
    sl = next(o for o in orders if o["order_type"]["trigger"]["tpsl"] == "sl")
    tp = next(o for o in orders if o["order_type"]["trigger"]["tpsl"] == "tp")
    assert sl["order_type"]["trigger"]["triggerPx"] == 90000.0
    assert tp["order_type"]["trigger"]["triggerPx"] == 110000.0
    assert sl["order_type"]["trigger"]["isMarket"] is True


def test_build_tpsl_orders_for_short_closes_with_buy():
    orders = build_tpsl_orders("ETH", is_long=False, qty=0.5, stop_px=4000.0, tp_px=3000.0)
    assert all(o["is_buy"] is True for o in orders)


def test_build_tpsl_orders_skips_missing_prices():
    orders = build_tpsl_orders("SOL", is_long=True, qty=1.0, stop_px=None, tp_px=150.0)
    assert len(orders) == 1 and orders[0]["order_type"]["trigger"]["tpsl"] == "tp"
    assert build_tpsl_orders("SOL", is_long=True, qty=1.0, stop_px=None, tp_px=None) == []
