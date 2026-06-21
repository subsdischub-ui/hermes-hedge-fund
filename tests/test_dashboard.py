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
