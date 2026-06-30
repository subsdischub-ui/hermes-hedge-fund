"""Boucle β — H1 driver: is realized funding-APR a DECREASING function of liquidity?

Descriptive measurement only (no registry trial). Regresses funding-APR on
log10(per-bar dollar-volume) across the full liquidity spectrum (23 majors -> Binance
tail -> HL exotic).

Red-team corrections folded in:
  * funding interval is DATA-DERIVED per symbol (Binance moved many alts to 4h; assuming
    8h understated their APR 2x) — illiquid_carry.infer_funding_interval_hours.
  * funding is WINDOW-MATCHED to the liquidity (perp) view span per symbol, so a relisted
    coin's deep funding history is not paired with a short liquidity window.
  * both the SIGNED slope (cash-and-carry harvestable) and the |APR| MAGNITUDE slope
    (sign-agnostic harvestable, reverse-carry territory) are reported.
  * the per-year slope uses the native MEAN (realized carry, corr 0.997), NOT the _on_4h
    backward-carried MEDIAN (a modal-clamp artifact, corr 0.28).

Run: .venv/bin/python research_streams/BETA/run_h1.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from hermes.datalake.carry_join import annualize_funding
from hermes.research import illiquid_carry as ic

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
_DUCKDB = Path(__file__).resolve().parents[2] / "data" / "hermes_datalake.duckdb"

# (venue, native/hourly funding view, interval-style, liquidity view carrying volume*close)
NATIVE_SPECS = [
    ("binance_major", "binance_funding_native", "binance_funding_on_4h"),
    ("binance_tail", "binance_tail_funding_native", "binance_tail_funding_on_4h"),
    ("hl_exotic", "hyperliquid_exotic_funding_1h", "hyperliquid_exotic_funding_on_4h"),
]
JAN_2025 = 1_735_689_600_000


def theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats
    return float(stats.theilslopes(y, x)[0])


def native_per_market(con: duckdb.DuckDBPyConnection, *, start_ts: int | None = None) -> pd.DataFrame:
    """One row per market: realized APR (mean native rate x DATA-DERIVED annualization),
    window-matched to the per-symbol liquidity (perp) view span."""
    wf = f" AND ts >= {start_ts}" if start_ts else ""
    rows = []
    for venue, fview, lview in NATIVE_SPECS:
        # per-symbol liquidity (the EXPLOITABLE window) + its ts span
        liqd = con.execute(
            f"SELECT symbol, median(volume*close) dollar_volume, count(*) n_bars, "
            f"min(ts) lo_perp, max(ts) hi_perp FROM {lview} WHERE 1=1{wf} GROUP BY symbol"
        ).df()
        fund = con.execute(f"SELECT symbol, funding_rate, ts FROM {fview} WHERE 1=1{wf}").df()
        span = dict(zip(liqd.symbol, zip(liqd.lo_perp, liqd.hi_perp)))
        agg = []
        for sym, g in fund.groupby("symbol"):
            if sym not in span:
                continue
            lo, hi = span[sym]
            gm = g[(g.ts >= lo) & (g.ts <= hi)]  # window-match funding to the perp window
            if len(gm) < 2:
                continue
            interval = ic.infer_funding_interval_hours(gm["ts"].to_numpy())
            agg.append({"symbol": sym, "interval_h": interval,
                        "funding_apr": annualize_funding(gm["funding_rate"].mean(), interval),
                        "n_settle": len(gm)})
        m = pd.DataFrame(agg).merge(liqd[["symbol", "dollar_volume", "n_bars", "lo_perp"]], on="symbol")
        m["venue"] = venue
        rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    return out[out["dollar_volume"] > 0].reset_index(drop=True)


def report(tag: str, frame: pd.DataFrame) -> None:
    signed = ic.liquidity_funding_regression(frame, per_market=True)
    absf = frame.assign(funding_apr=frame["funding_apr"].abs())
    mag = ic.liquidity_funding_regression(absf, per_market=True)
    x = np.log10(frame.groupby("symbol")["dollar_volume"].median().to_numpy())
    y = frame.groupby("symbol")["funding_apr"].median().to_numpy()
    ts_slope = theil_sen(x, y) if len(x) >= 3 else float("nan")
    print(f"  {tag:<30} n={signed['n']:>3}  SIGNED slope={signed['slope']:+.4f} (t={signed['t_stat']:+.2f}) "
          f"TS={ts_slope:+.4f}  |  |APR| slope={mag['slope']:+.4f} (t={mag['t_stat']:+.2f})")


def main() -> None:
    con = duckdb.connect(str(_DUCKDB), read_only=True)
    print("=" * 96)
    print("H1 — funding-APR vs log10(dollar_volume).  SIGNED<0 = H1 (cash-and-carry harvest rises in tail);")
    print("     |APR| = sign-agnostic magnitude (reverse-carry territory).  APR in fraction/yr.")
    print("=" * 96)

    nat = native_per_market(con)
    nat["lo_d"] = pd.to_datetime(nat["lo_perp"], unit="ms").dt.strftime("%Y-%m")
    print("\n[NATIVE per-market — DATA-DERIVED interval, WINDOW-MATCHED] realized APR vs median $-vol")
    show = nat[["venue", "symbol", "interval_h", "funding_apr", "dollar_volume", "n_settle", "lo_d"]].copy()
    show = show.sort_values("dollar_volume")
    show["funding_apr"] = show["funding_apr"].map(lambda v: f"{v:+.3f}")
    show["dollar_volume"] = show["dollar_volume"].map(lambda v: f"{v:,.0f}")
    print(show.to_string(index=False))

    print("\n[SIGNED + |MAGNITUDE| regressions — FULL history]")
    report("ALL venues", nat)
    report("Binance only (major+tail)", nat[nat.venue != "hl_exotic"])
    for v in ("binance_major", "binance_tail", "hl_exotic"):
        report(f"within {v}", nat[nat.venue == v])

    natf = native_per_market(con, start_ts=JAN_2025)
    print("\n[SIGNED + |MAGNITUDE| regressions — FORWARD 2025+ (deployable regime §4b)]")
    report("ALL venues (2025+)", natf)
    report("Binance only (2025+)", natf[natf.venue != "hl_exotic"])
    for v in ("binance_major", "binance_tail", "hl_exotic"):
        report(f"within {v} (2025+)", natf[natf.venue == v])

    # absolute forward levels at the illiquid end (the core)
    print("\n[ABSOLUTE forward 2025+ funding-APR by band] (does the tail carry POSITIVE forward?)")
    for v in ("binance_major", "binance_tail", "hl_exotic"):
        g = natf[natf.venue == v]
        print(f"  {v:<14} median={g.funding_apr.median():+.3f}  mean={g.funding_apr.mean():+.3f}  "
              f"frac>0={ (g.funding_apr>0).mean():.2f}  median|APR|={g.funding_apr.abs().median():.3f}")

    # honest per-year (native MEAN, NOT the _on_4h modal-clamp median)
    print("\n[PER-YEAR cross-sectional slope — native MEAN realized carry (not the _on_4h median artifact)]")
    fund_all = []
    for _venue, fview, lview in NATIVE_SPECS:
        f = con.execute(f"SELECT symbol, funding_rate, ts FROM {fview}").df()
        ivl = f.groupby("symbol")["ts"].apply(lambda s: ic.infer_funding_interval_hours(s.to_numpy()))
        f["year"] = pd.to_datetime(f["ts"], unit="ms").dt.year
        f["apr"] = f.apply(lambda r, ivl=ivl: annualize_funding(r.funding_rate, ivl[r.symbol]), axis=1)
        lv = con.execute(f"SELECT symbol, ts, volume*close dv FROM {lview}").df()
        lv["year"] = pd.to_datetime(lv["ts"], unit="ms").dt.year
        fy = f.groupby(["symbol", "year"], as_index=False)["apr"].mean().rename(columns={"apr": "funding_apr"})
        ly = lv.groupby(["symbol", "year"], as_index=False)["dv"].median().rename(columns={"dv": "dollar_volume"})
        fund_all.append(fy.merge(ly, on=["symbol", "year"]))
    peryear = pd.concat(fund_all, ignore_index=True)
    for yr, g in peryear.groupby("year"):
        if g["symbol"].nunique() < 5:
            continue
        r = ic.liquidity_funding_regression(g, per_market=True)
        print(f"  {int(yr)}: n={r['n']:>3}  SIGNED slope={r['slope']:+.4f}  t={r['t_stat']:+.2f}  R²={r['r2']:.3f}")
    con.close()


if __name__ == "__main__":
    main()
