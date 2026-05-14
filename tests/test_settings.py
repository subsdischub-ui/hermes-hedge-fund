from hermes.settings import get_settings


def test_settings_load_yaml_and_universe():
    s = get_settings()
    assert s.bankroll_usd == 100
    assert "BTC/USDC" in s.universe
    assert s.stage == "stage_0_paper"
    assert "advance_when" in s.stage_config


def test_yaml_helper():
    s = get_settings()
    assert s.yaml("loops", "market_tick") is not None
    assert s.yaml("nope", "missing", default=42) == 42
