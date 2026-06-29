"""Offline tests for the CoinGlass MULTI-VENUE lake client (Stream 0 scout +).

NO live network. The scout proved the subscribed tier serves DEEP multi-exchange
funding + OI (OKX/Bybit/Bitget all page back to 2021), so we ingest a cross-venue
dispersion substrate for S2.3. The only NEW pure logic is the per-(exchange,coin)
perpetual-instrument resolver — the trap is Bybit, which lists the perpetual
``BTCUSDT`` ALONGSIDE many dated futures ``BTCUSDT-03JUL26``; we must pick the perp.
"""

from __future__ import annotations

from hermes.datalake.sources.coinglass_multi import (
    COINS,
    EXCHANGES,
    resolve_perp_instrument,
)

# A realistic slice of /api/futures/supported-exchange-pairs (perp + dated +
# inverse mixed together, exactly as the live API returns them).
_SUPPORTED = {
    "OKX": [
        {"instrument_id": "BTC-USD-SWAP", "base_asset": "BTC", "quote_asset": "USD",
         "settlement_currency": "BTC", "funding_interval": 8},
        {"instrument_id": "BTC-USDT-SWAP", "base_asset": "BTC", "quote_asset": "USDT",
         "settlement_currency": "USDT", "funding_interval": 8},
        {"instrument_id": "BTC-USD-260703", "base_asset": "BTC", "quote_asset": "USD",
         "settlement_currency": "BTC", "expiry_time": 1783065600000},
    ],
    "Bybit": [
        {"instrument_id": "BTCPERP", "base_asset": "BTC", "quote_asset": "USDC",
         "settlement_currency": "USDC", "funding_interval": 8},
        {"instrument_id": "BTCUSDT", "base_asset": "BTC", "quote_asset": "USDT",
         "settlement_currency": "USDT", "funding_interval": 8},
        {"instrument_id": "BTCUSDT-03JUL26", "base_asset": "BTC", "quote_asset": "USDT",
         "settlement_currency": "USDT", "expiry_time": 1783065600000},
        {"instrument_id": "BTCUSDT-25DEC26", "base_asset": "BTC", "quote_asset": "USDT",
         "settlement_currency": "USDT", "expiry_time": 1798675200000},
    ],
    "Bitget": [
        {"instrument_id": "BTCUSDT_UMCBL", "base_asset": "BTC", "quote_asset": "USDT",
         "settlement_currency": "USDT", "funding_interval": 8},
        {"instrument_id": "BTCUSD_DMCBL", "base_asset": "BTC", "quote_asset": "USD",
         "settlement_currency": "BTC", "funding_interval": 8},
    ],
}


def test_resolver_picks_usdt_perp_over_inverse_and_dated():
    # OKX: the USDT swap, not the USD inverse, not the dated future
    assert resolve_perp_instrument(_SUPPORTED, "OKX", "BTC") == "BTC-USDT-SWAP"


def test_resolver_bybit_picks_perpetual_not_dated_future():
    # the heart of the trap: BTCUSDT (perp) beats BTCUSDT-03JUL26 (dated)
    assert resolve_perp_instrument(_SUPPORTED, "Bybit", "BTC") == "BTCUSDT"


def test_resolver_bitget_picks_usdt_linear():
    assert resolve_perp_instrument(_SUPPORTED, "Bitget", "BTC") == "BTCUSDT_UMCBL"


def test_resolver_returns_none_when_absent():
    assert resolve_perp_instrument(_SUPPORTED, "OKX", "PEPE") is None
    assert resolve_perp_instrument(_SUPPORTED, "NoSuchExchange", "BTC") is None


def test_resolver_excludes_dated_even_if_only_option():
    # a coin that ONLY has a dated future (no perp) must resolve to None, never a
    # future masquerading as the perp
    m = {"Bybit": [{"instrument_id": "FOOUSDT-25DEC26", "base_asset": "FOO",
                    "quote_asset": "USDT", "settlement_currency": "USDT",
                    "expiry_time": 1798675200000}]}
    assert resolve_perp_instrument(m, "Bybit", "FOO") is None


def test_universe_constants_sane():
    # cross-venue dispersion needs >=2 non-Binance venues and a handful of coins
    assert "OKX" in EXCHANGES and "Bybit" in EXCHANGES
    assert len([e for e in EXCHANGES if e != "Binance"]) >= 2
    assert {"BTC", "ETH", "SOL"}.issubset(set(COINS))
