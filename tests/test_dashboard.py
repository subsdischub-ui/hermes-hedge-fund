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
        for key in (
            "stage", "equity_usd", "journal", "open_positions", "scheduler", "models",
            "forward", "governance", "restructuration",
        ):
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
    mean_reversion_regime under the LIVE mean-reversion card."""
    from types import SimpleNamespace

    from hermes.api import _strategies_ab

    rows = [
        SimpleNamespace(skill_id="mean_reversion_regime", pnl_usd=1.0),
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


def test_strategies_ab_does_not_conflate_mean_reversion_with_retired_skills():
    """RESTRUCTURATION 2026-07-02: S1 (ts_mom_vol_scaled) was retired from the live
    slate; mean_reversion_regime is a DISTINCT skill that stays live. Before the fix,
    both were folded into one "Strategy 1 (Momentum)" card, so a future
    mean_reversion_regime trade would display as if it belonged to a retired
    strategy. They must now be attributed to separate cards/buckets."""
    from types import SimpleNamespace

    from hermes.api import _strategies_ab

    rows = [
        SimpleNamespace(skill_id="mean_reversion_regime", pnl_usd=1.5),
        SimpleNamespace(skill_id="ts_mom_vol_scaled", pnl_usd=2.74),
        SimpleNamespace(skill_id="breakout_vol_expansion", pnl_usd=-0.3),
        SimpleNamespace(skill_id="trend_follow_ema", pnl_usd=-0.2),
    ]
    ab = _strategies_ab(rows)
    # The live mean-reversion card carries ONLY the live skill.
    assert ab["strategy_1"]["live"]["n"] == 1
    assert ab["strategy_1"]["live"]["pnl_usd"] == 1.5
    assert "momentum" not in ab["strategy_1"]["label"].lower()
    assert "mean" in ab["strategy_1"]["label"].lower() or "revers" in ab["strategy_1"]["label"].lower()
    # The retired bucket carries the legacy/retired skills, correctly attributed,
    # and is clearly labelled historical/retired — not "live".
    retired = ab["retired_skills"]
    assert retired["live"]["n"] == 3
    assert round(retired["live"]["pnl_usd"], 2) == round(2.74 - 0.3 - 0.2, 2)
    assert retired["status"] != "live"
    label = retired["label"].lower()
    assert "retir" in label or "historiq" in label or "legacy" in label


def test_dashboard_data_forward_block_fail_soft_when_no_data(monkeypatch, tmp_path):
    """No forward/ directory on disk (fresh local checkout) -> 'no_data' status per
    flux, never an exception that breaks the whole endpoint."""
    from hermes.settings import get_settings

    monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            data = client.get("/dashboard/data")
            assert data.status_code == 200
            fwd = data.json()["forward"]
            assert isinstance(fwd, dict)
            assert fwd  # at least the known flux names are reported
            for _flux, info in fwd.items():
                assert info.get("status") == "no_data"
    finally:
        get_settings.cache_clear()


def test_dashboard_data_governance_block_fail_soft():
    with TestClient(app) as client:
        data = client.get("/dashboard/data")
        assert data.status_code == 200
        gov = data.json()["governance"]
        assert "trials_by_book" in gov
        assert "dead_ends_verdicts" in gov


def test_dashboard_data_restructuration_block_has_required_facts():
    with TestClient(app) as client:
        data = client.get("/dashboard/data")
        assert data.status_code == 200
        r = data.json()["restructuration"]
        assert "mean_reversion_regime" in r["slate"]
        assert "funding_squeeze_continuation" in r["slate"]
        assert "ts_mom_vol_scaled" not in r["slate"]
        assert r["retired"]
        retired_ids = [x["skill_id"] if isinstance(x, dict) else x for x in r["retired"]]
        assert "ts_mom_vol_scaled" in retired_ids
        assert r["gate"]["edge_paper_trades_target"] == 40
        assert "edge_paper_trades_observed" in r["gate"]
        assert r["metals_collect_now"]["symbols"] == ["XAUUSDT", "XAGUSDT"]
        assert r["metals_collect_now"]["retest_deadline"] == "2027-07-02"
        assert r["fx"]["status"].lower() in ("fermé", "ferme", "closed")
        assert r["fx"]["reopen_check"] == "2027-07-02"
        assert r["global_cutoff"] == "2028-01-02"
        assert "RESTRUCTURATION" in r["dossier"]
