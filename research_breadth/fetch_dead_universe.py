"""Étape durcissement — fetch complet 1h+funding pour les MORTS QUALIFIÉS point-in-time
(PREREG_HARDENED.md), écrit avec suffixe _hardened (n'écrase RIEN du run original).

Univers des morts testés : 122 candidats /fapi/v1/exchangeInfo status=SETTLING (USDT-M
PERPETUAL, hors stablecoins/wrapped) + 9 délistings historiques plus anciens (hors format
exchangeInfo actuel, cherchés via web pour honorer la source (iii) du PREREG car (i)+(ii)
n'ont donné que 4 morts qualifiés < 5 : SRM/TOMO/HNT/COCOS/ANC/DODO cités par le PREREG comme
preuve de faisabilité API, + LUNA/BTT/ANT trouvés via recherche d'annonces de délisting).
Écran point-in-time (volume 30j-glissant >= $200M ET historique funding >= 3 ans, les DEUX
simultanément, testés sur la donnée PROPRE de chaque perp jusqu'à sa vraie date de mort).

Découverte critique d'instrument (2026-07-16) : /fapi/v1/klines PAD en avant indéfiniment
avec des barres gelées (volume=0, close=dernier prix connu) après l'arrêt réel du trading —
`deliveryDate` d'exchangeInfo est ADDITIONNELLEMENT non fiable (ex. OMGUSDT deliveryDate
métadonnée = 2025-01-31 mais funding réel continue jusqu'à 2026-06-18, ~16 mois d'écart).
La vraie date de mort = max(dernière barre kline à volume>0, dernière entrée funding réelle),
vérifiée par recoupement : ce fetch s'arrête à cette date (+buffer 2j), PAS à "aujourd'hui",
pour ne jamais aspirer les barres de padding gelées dans les fichiers _hardened.

Seuls 4 candidats qualifient (sur 131 testés au total, cf. mission) : MKR, BLZ, FTM, REEF.

Usage: python3 fetch_dead_universe.py
Écrit: research_breadth/data/{COIN}_1h_hardened.csv, {COIN}_funding_hardened.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FAPI = "https://fapi.binance.com"
KLINE_LIMIT = 1500
FUND_LIMIT = 1000
HDRS = {"User-Agent": "hermes-research-breadth/1.0"}

# Morts qualifiés point-in-time (a: vol30j>=$200M, b: historique funding>=3ans, c: exclusions
# stablecoin/wrapped/metal deja appliquees en amont) -- entry_ts_ms/delisting_ts_ms = bornes
# de recherche fetch (buffer +2j sur delisting pour capturer la toute derniere barre reelle
# sans deborder dans le padding gele).
DEAD_QUALIFIED = {
    "MKR": {"symbol": "MKRUSDT", "onboard_ms": 1597302000000, "delisting_ts_ms": 1757318400000,
            "entry_ts_ms": 1692057600000},
    "BLZ": {"symbol": "BLZUSDT", "onboard_ms": 1600326000000, "delisting_ts_ms": 1750320000003,
            "entry_ts_ms": 1694995200000},
    "FTM": {"symbol": "FTMUSDT", "onboard_ms": 1600930800000, "delisting_ts_ms": 1750320000003,
            "entry_ts_ms": 1703548800000},
    "REEF": {"symbol": "REEFUSDT", "onboard_ms": 1611558000000, "delisting_ts_ms": 1750320000003,
             "entry_ts_ms": 1726185600000},
}
BUFFER_MS = 2 * 24 * 3600 * 1000


def get_json(url: str, params: dict, timeout: float = 25.0):
    for attempt in range(6):
        try:
            r = httpx.get(url, params=params, timeout=timeout, headers=HDRS)
            if r.status_code == 200:
                return r.json()
            print(f"    HTTP {r.status_code} on {params}, attempt {attempt}: {r.text[:200]}")
        except (httpx.TransportError, httpx.TimeoutException) as e:
            print(f"    transport error {e!r}, attempt {attempt}")
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url} params={params}")


def fetch_klines_1h(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cur = start_ms
    while cur < end_ms:
        data = get_json(f"{FAPI}/fapi/v1/klines",
                         {"symbol": symbol, "interval": "1h", "limit": KLINE_LIMIT, "startTime": cur})
        if not data:
            break
        # borne stricte a end_ms (ne pas deborder dans le padding gele post-delisting)
        data = [d for d in data if int(d[0]) <= end_ms]
        if not data:
            break
        rows.extend(data)
        last_open = int(data[-1][0])
        if len(data) < KLINE_LIMIT or last_open >= end_ms:
            break
        cur = last_open + 1
        time.sleep(0.15)
    return rows


def fetch_funding_all(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    rows: list[dict] = []
    cur = start_ms
    while cur < end_ms:
        data = get_json(f"{FAPI}/fapi/v1/fundingRate",
                         {"symbol": symbol, "limit": FUND_LIMIT, "startTime": cur})
        if not data:
            break
        data = [d for d in data if int(d["fundingTime"]) <= end_ms]
        if not data:
            break
        rows.extend(data)
        if len(data) < FUND_LIMIT:
            break
        cur = int(data[-1]["fundingTime"]) + 1
        time.sleep(0.15)
    return rows


def main():
    report = {}
    for coin, meta in DEAD_QUALIFIED.items():
        symbol = meta["symbol"]
        onboard = meta["onboard_ms"]
        end_ms = meta["delisting_ts_ms"] + BUFFER_MS
        print(f"\n=== {coin} ({symbol}) onboard={onboard} end={end_ms} ===")

        raw_k = fetch_klines_1h(symbol, onboard, end_ms)
        print(f"  klines: fetched {len(raw_k)} rows")
        kdf = pd.DataFrame(raw_k, columns=[
            "ts", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_base", "taker_buy_quote", "ignore",
        ])[["ts", "open", "high", "low", "close", "volume"]]
        for c in ["open", "high", "low", "close", "volume"]:
            kdf[c] = kdf[c].astype(float)
        kdf["ts"] = kdf["ts"].astype("int64")
        kdf = kdf.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
        kpath = DATA_DIR / f"{coin}_1h_hardened.csv"
        kdf.to_csv(kpath, index=False)
        print(f"  wrote {kpath} ({len(kdf)} rows, "
              f"{pd.to_datetime(kdf['ts'].min(), unit='ms')} -> {pd.to_datetime(kdf['ts'].max(), unit='ms')})")

        raw_f = fetch_funding_all(symbol, max(0, onboard - 3 * 24 * 3600 * 1000), end_ms)
        print(f"  funding: fetched {len(raw_f)} rows")
        fdf = pd.DataFrame(raw_f)[["fundingTime", "fundingRate"]].rename(
            columns={"fundingTime": "ts", "fundingRate": "funding_rate"})
        fdf["ts"] = fdf["ts"].astype("int64")
        fdf["funding_rate"] = fdf["funding_rate"].astype(float)
        fdf = fdf.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
        fpath = DATA_DIR / f"{coin}_funding_hardened.csv"
        fdf.to_csv(fpath, index=False)
        print(f"  wrote {fpath} ({len(fdf)} rows, "
              f"{pd.to_datetime(fdf['ts'].min(), unit='ms')} -> {pd.to_datetime(fdf['ts'].max(), unit='ms')})")

        # sanity: last few klines should NOT be frozen-zero-volume padding (would indicate
        # end_ms boundary miscalculated and we accidentally captured dead padding bars)
        tail_vol = kdf["volume"].tail(5).tolist()
        report[coin] = {
            "symbol": symbol, "n_klines_1h": len(kdf), "n_funding": len(fdf),
            "klines_start": str(pd.to_datetime(kdf["ts"].min(), unit="ms")),
            "klines_end": str(pd.to_datetime(kdf["ts"].max(), unit="ms")),
            "funding_start": str(pd.to_datetime(fdf["ts"].min(), unit="ms")),
            "funding_end": str(pd.to_datetime(fdf["ts"].max(), unit="ms")),
            "last_5_hourly_volumes": tail_vol,
            "entry_ts_ms": meta["entry_ts_ms"],
            "delisting_ts_ms": meta["delisting_ts_ms"],
        }

    (DATA_DIR / "dead_universe_fetch_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("\n\nwrote", DATA_DIR / "dead_universe_fetch_report.json")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
