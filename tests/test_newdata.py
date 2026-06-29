"""Live new-data source: pure-logic correctness, the BTC/ETH-only + graceful-None
guarantees, caching, and an offline end-to-end fetch via a fake httpx (so the live
shadow features are exercised without hitting Deribit/DefiLlama)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from hermes.data.newdata import _zscore_last, dvol_z_now, reset_cache, stbl_supply_growth_z_now


def setup_function(_):
    reset_cache()


def test_zscore_last_matches_manual():
    g = np.arange(1.0, 101.0)  # ramp -> last z is positive, finite
    s = pd.Series(g)
    mu = s.rolling(60).mean()
    sd = s.rolling(60).std(ddof=0)
    expected = float(((s - mu) / sd).iloc[-1])
    assert _zscore_last(g, 60) == pytest.approx(expected)


def test_zscore_last_none_on_thin_or_flat():
    assert _zscore_last(np.array([1.0, 2.0]), 60) is None      # too short
    assert _zscore_last(np.full(80, 5.0), 60) is None          # zero std -> NaN


@pytest.mark.asyncio
async def test_dvol_z_none_for_non_btc_eth():
    """No options-IV signal off BTC/ETH — returns before any network call."""
    assert await dvol_z_now("SOL") is None
    assert await dvol_z_now("WIF") is None


def _fake_httpx(dvol_pts: int = 2300, stbl_days: int = 220) -> types.ModuleType:
    """A minimal fake httpx whose AsyncClient.get returns synthetic Deribit DVOL or
    DefiLlama stablecoin payloads depending on the URL."""
    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            if "deribit" in url:
                base_ts = 1_600_000_000_000
                data = [[base_ts + i * 3_600_000, 0, 0, 0, 50.0 + (i % 23) * 0.5]
                        for i in range(dvol_pts)]
                return _Resp({"result": {"data": data, "continuation": None}})
            # DefiLlama /stablecoin/{id}: rising supply
            sid = int(url.rstrip("/").split("/")[-1])
            day = 86400
            base = 1_500_000_000
            mult = 1.0 if sid == 1 else 0.5
            tokens = [{"date": base + i * day,
                       "circulating": {"peggedUSD": (1e9 + i * 1e6) * mult}}
                      for i in range(stbl_days)]
            return _Resp({"tokens": tokens})

    mod = types.ModuleType("httpx")
    mod.AsyncClient = _Client
    return mod


@pytest.mark.asyncio
async def test_dvol_z_now_computes_offline(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", _fake_httpx())
    z = await dvol_z_now("BTC")
    assert z is not None and np.isfinite(z)


@pytest.mark.asyncio
async def test_stbl_z_now_computes_offline(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", _fake_httpx())
    z = await stbl_supply_growth_z_now()
    assert z is not None and np.isfinite(z)


@pytest.mark.asyncio
async def test_fetch_failure_returns_none_not_raise(monkeypatch):
    """A broken feed must yield None (sleeves HOLD), never propagate an exception
    into the tick."""
    boom = types.ModuleType("httpx")

    class _Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise RuntimeError("network down")

        async def __aexit__(self, *a):
            return False

    boom.AsyncClient = _Boom
    monkeypatch.setitem(sys.modules, "httpx", boom)
    assert await dvol_z_now("BTC") is None
    assert await stbl_supply_growth_z_now() is None


@pytest.mark.asyncio
async def test_cache_returns_without_refetch(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", _fake_httpx())
    first = await dvol_z_now("ETH")
    # Now break httpx: a cached hit must NOT refetch, so the value is still returned.
    boom = types.ModuleType("httpx")
    boom.AsyncClient = None  # any use would TypeError
    monkeypatch.setitem(sys.modules, "httpx", boom)
    assert await dvol_z_now("ETH") == first
