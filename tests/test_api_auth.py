"""API key gate: /tick and /learn/* must not be open to the internet once a key
is configured. Open (with a warning) when unconfigured so paper isn't broken."""

from hermes.api import _api_key_ok


def test_open_when_no_key_configured():
    assert _api_key_ok(None, "") is True
    assert _api_key_ok("anything", "") is True


def test_requires_exact_match_when_configured():
    assert _api_key_ok("secret", "secret") is True
    assert _api_key_ok("wrong", "secret") is False
    assert _api_key_ok(None, "secret") is False
    assert _api_key_ok("", "secret") is False


def test_dependency_blocks_requests_without_valid_key(monkeypatch):
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from hermes import settings as settings_mod
    from hermes.api import require_api_key

    monkeypatch.setenv("HERMES_API_KEY", "topsecret")
    settings_mod.get_settings.cache_clear()
    try:
        app = FastAPI()

        @app.get("/p", dependencies=[Depends(require_api_key)])
        def protected():
            return {"ok": True}

        client = TestClient(app)
        assert client.get("/p").status_code == 401
        assert client.get("/p", headers={"x-api-key": "wrong"}).status_code == 401
        assert client.get("/p", headers={"x-api-key": "topsecret"}).status_code == 200
    finally:
        settings_mod.get_settings.cache_clear()
