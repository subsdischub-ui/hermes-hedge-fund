"""Fidélité du tick aux barres CLÔTURÉES — leçon du 2026-07-16.

Constat prod (audit exécuté, pas raisonné) : le market_tick tirait via le cron
"0 */4" dans la TZ APPLICATIVE (HERMES_TZ, ex. Europe/Paris = UTC+2) → à H+2 du
cycle des barres 4h Binance (00/04/08/12/16/20 UTC), ET `MarketData.ohlcv`
gardait la barre EN COURS en dernière position. Résultat : chaque décision
évaluait une demi-barre jamais backtestée ; sur 90 j rejoués, ~18 % des setups
mean-reversion validés disparaissaient, et les entrées restantes étaient prises
sur des valeurs intra-barre non validées (sélection adverse pour un signal de
réversion : l'extrême qui a déjà rebondi n'arme plus, celui qui continue de
chuter arme encore).

Deux mécanismes imposent désormais la règle validée (R3 — mécanisme, pas
déclaration) :
  1. `MarketData.ohlcv(closed_only=True)` coupe la barre non clôturée ;
  2. le trigger du market_tick est ANCRÉ UTC (clôtures Binance), immune à
     HERMES_TZ et au DST, à la minute 1 (marge de consolidation du venue).
"""

from __future__ import annotations

import asyncio
import time

from hermes.data.market import MarketData
from hermes.scheduler import market_tick_trigger

_TF_MS = 4 * 3_600_000


class _StubClient:
    """fetch_ohlcv figé, sans réseau ; parse_timeframe comme ccxt (secondes)."""

    def __init__(self, rows: list[list[float]]):
        self._rows = rows

    def parse_timeframe(self, tf: str) -> int:
        assert tf == "4h"
        return 4 * 3600

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=200):
        return list(self._rows)


def _md(rows: list[list[float]]) -> MarketData:
    md = MarketData.__new__(MarketData)
    md.venue = "paper"
    md.client = _StubClient(rows)
    return md


def _rows_with_open_last_bar() -> list[list[float]]:
    """3 barres 4h dont la DERNIÈRE est en cours (ouverte il y a ~1h)."""
    now = int(time.time() * 1000)
    open_cur = now - 3_600_000  # en cours: open+4h > now
    return [
        [open_cur - 2 * _TF_MS, 1.0, 2.0, 0.5, 1.5, 10.0],
        [open_cur - _TF_MS, 1.5, 2.5, 1.0, 2.0, 11.0],
        [open_cur, 2.0, 3.0, 1.5, 2.5, 3.0],
    ]


def test_ohlcv_drops_unclosed_last_bar_by_default():
    rows = _rows_with_open_last_bar()
    df = asyncio.run(_md(rows).ohlcv("BTC/USDT:USDT", timeframe="4h", limit=3))
    assert len(df) == 2, "la barre en cours doit être coupée par défaut (fidélité backtest)"
    assert int(df["ts"].iloc[-1].timestamp() * 1000) == rows[1][0]


def test_ohlcv_keeps_unclosed_bar_when_explicitly_asked():
    rows = _rows_with_open_last_bar()
    df = asyncio.run(
        _md(rows).ohlcv("BTC/USDT:USDT", timeframe="4h", limit=3, closed_only=False)
    )
    assert len(df) == 3


def test_ohlcv_keeps_last_bar_when_it_is_closed():
    now = int(time.time() * 1000)
    open_last = now - _TF_MS - 5_000  # clôturée il y a 5 s
    rows = [
        [open_last - _TF_MS, 1.0, 2.0, 0.5, 1.5, 10.0],
        [open_last, 1.5, 2.5, 1.0, 2.0, 11.0],
    ]
    df = asyncio.run(_md(rows).ohlcv("BTC/USDT:USDT", timeframe="4h", limit=2))
    assert len(df) == 2, "une barre déjà clôturée ne doit jamais être coupée"


def test_market_tick_trigger_is_utc_anchored_to_4h_closes():
    trig = market_tick_trigger("1 0,4,8,12,16,20 * * *")
    assert str(trig.timezone) == "UTC", (
        "le market_tick doit être ancré UTC (clôtures Binance), pas en HERMES_TZ — "
        "c'est le mécanisme qui a tiré à H+2 pendant 3 semaines"
    )
    s = str(trig)
    assert "hour='0,4,8,12,16,20'" in s
    assert "minute='1'" in s
