"""Dashboard endpoints serve the HTML shell and a read-only telemetry feed."""

from fastapi.testclient import TestClient

from hermes.api import app


def test_dashboard_html_and_data():
    with TestClient(app) as client:
        page = client.get("/dashboard")
        assert page.status_code == 200
        assert "Hermès" in page.text

        data = client.get("/dashboard/data")
        assert data.status_code == 200
        j = data.json()
        for key in ("stage", "equity_usd", "journal", "open_positions", "scheduler", "models"):
            assert key in j


def test_manual_tick_resolves_base_from_universe():
    """/tick/{base} resolves the bare coin to its full universe symbol (Binance perp
    format); an unknown base 404s instead of fabricating a hardcoded /USDC symbol."""
    with TestClient(app) as client:
        r = client.post("/tick/ZZZ")  # not in universe
        assert r.status_code == 404
        assert "ZZZ" in r.json()["detail"]


def test_strategies_ab_attributes_funding_to_strategy_2():
    """A/B split: funding trades land under Strategy #2 (the validated brick),
    momentum under #1."""
    from types import SimpleNamespace

    from hermes.api import _strategies_ab

    rows = [
        SimpleNamespace(skill_id="ts_mom_vol_scaled", pnl_usd=1.0),
        SimpleNamespace(skill_id="funding_squeeze_continuation", pnl_usd=2.0),
        SimpleNamespace(skill_id="funding_squeeze_continuation", pnl_usd=-0.5),
    ]
    ab = _strategies_ab(rows)
    assert ab["strategy_1"]["live"]["n"] == 1
    assert ab["strategy_2"]["live"]["n"] == 2
    assert ab["strategy_2"]["status"] == "live"  # has closed funding trades
    assert "funding" in ab["strategy_2"]["label"].lower()
    assert ab["strategy_2"]["research"]["oos_validated_bricks"] == 1
    # Strategy #3 (carry) present, validated, but never traded (no closed rows).
    assert "carry" in ab["strategy_3"]["label"].lower()
    assert ab["strategy_3"]["status"] == "validated"
    assert ab["strategy_3"]["live"]["n"] == 0
