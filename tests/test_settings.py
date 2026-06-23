from hermes.settings import get_settings


def test_settings_load_yaml_and_universe():
    s = get_settings()
    assert s.bankroll_usd == 100
    assert "BTC/USDT:USDT" in s.universe
    assert s.stage == "stage_0_paper"
    assert "advance_when" in s.stage_config


def test_universe_reads_yaml_expanded_pairs():
    """Universe is code-controlled via the yaml `universe:` list (env HERMES_UNIVERSE
    overrides). 7 Binance USDT-M perp symbols incl. the 4 validated alts."""
    s = get_settings()
    u = s.universe
    for sym in ("BTC/USDT:USDT", "AVAX/USDT:USDT", "SUI/USDT:USDT",
                "ARB/USDT:USDT", "WIF/USDT:USDT"):
        assert sym in u, f"{sym} missing from universe {u}"
    assert len(u) == 7


def test_data_venue_is_binance():
    """Data feed = the operator's actual venue (Binance USDT-M), yaml-driven."""
    assert get_settings().exchange_primary == "binanceusdm"


def test_yaml_helper():
    s = get_settings()
    assert s.yaml("loops", "market_tick") is not None
    assert s.yaml("nope", "missing", default=42) == 42
