"""Boucle β — H2 driver (THE killer test): does NET carry survive the illiquid tail's
liquidity-dependent slippage + brutal basis tail?

Runs the alpha cash-and-carry harness (carry_harvest, UNCHANGED) on the 13 Binance-tail
markets that have a spot leg, with the pre-registered √-impact slippage charged at the
prevailing (trailing) liquidity (illiquid_carry.liquidity_slippage / variable-cost engine).
Reports GROSS vs NET carry-APR per market, per liquidity band, and the forward (2025+)
regime — the exhibit that shows the negative is robust ACROSS bands (worse at the bottom).

Pre-registration frozen in research_streams/BETA/H2_PREREG.md.
Run: .venv/bin/python research_streams/BETA/run_h2.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from hermes.research import carry_harvest as ch
from hermes.research import illiquid_carry as ic

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
_DUCKDB = Path(__file__).resolve().parents[2] / "data" / "hermes_datalake.duckdb"

TAIL_SPOT = dict(
    spot_view="binance_tail_spot_klines_4h",
    perp_view="binance_tail_funding_on_4h",
    native_view="binance_tail_funding_native",
)
LOOKBACK = ch.MONTHLY_BARS   # 180 bars (monthly), alpha's low-whipsaw choice
CADENCE = ch.MONTHLY_BARS
TAUS = (0.05, 0.10)
BARS_YR = ch.BARS_PER_YEAR_4H
JAN_2025 = 1_735_689_600_000


def _maxdd(cum: np.ndarray) -> float:
    if cum.size == 0:
        return 0.0
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def harvest_one(con, symbol: str, tau: float, *, start_ts: int | None = None) -> dict:
    """Run the gated cash-and-carry on one tail market with the pre-registered slippage."""
    stream = ch.cash_and_carry_stream(con, symbol, **TAIL_SPOT)
    # per-bar dollar-volume from the perp view, aligned to the stream grid
    dv = con.execute(
        "SELECT ts, volume*close AS dv FROM binance_tail_funding_on_4h "
        "WHERE symbol = ? ORDER BY ts", [symbol]
    ).df()
    s = stream.merge(dv, on="ts", how="left").sort_values("ts").reset_index(drop=True)
    if start_ts is not None:
        s = s[s["ts"] >= start_ts].reset_index(drop=True)
    if len(s) < LOOKBACK + 5:
        return {}
    s["dv"] = s["dv"].ffill().bfill()

    # trailing median liquidity -> per-bar slippage (causal, leak-guarded)
    s = s.assign(symbol=symbol)
    from hermes.datalake import liquidity as liq
    s = liq.attach_trailing_liquidity(s.rename(columns={"dv": "dollar_volume"}), lookback_bars=LOOKBACK)
    slip = ic.liquidity_slippage(s["trail_dollar_volume"].to_numpy())

    tdf = ch.add_trailing_funding_apr(s, LOOKBACK, col="funding")
    armed = ch.arm_schedule(tdf["trail_apr"].to_numpy(), tau, CADENCE, warmup_bars=LOOKBACK)
    basis_ret = s["basis_ret"].to_numpy()
    funding = s["funding"].to_numpy()
    net = ic.gated_net_returns_variable_cost(basis_ret, funding, armed, slip)
    gross = ch.gated_net_returns(basis_ret, funding, armed, 0.0)

    years = max((s["ts"].iloc[-1] - s["ts"].iloc[0]) / (1000 * 3600 * 24 * 365), 1e-9)
    cum = np.cumsum(net)
    return {
        "symbol": symbol, "n": len(s), "years": years,
        "med_dv": float(s["dollar_volume"].median()),
        "arm_rate": float(armed.mean()),
        "gross_apr": float(gross.sum() / years),
        "net_apr": float(net.sum() / years),
        "mean_slip_bps": float(slip[armed].mean() * 1e4) if armed.any() else float("nan"),
        "worst_bar": float(net.min()),
        "maxdd": _maxdd(cum),
    }


def spot_leg_markets(con) -> list[str]:
    rows = con.execute("""
        SELECT u.symbol FROM binance_tail_universe u
        WHERE EXISTS (SELECT 1 FROM binance_tail_spot_klines_4h s WHERE s.symbol=u.symbol)
        ORDER BY u.symbol
    """).df()
    return rows["symbol"].tolist()


def band_of(med_dv: float, edges: tuple[float, float]) -> str:
    if med_dv < edges[0]:
        return "illiquid"
    if med_dv < edges[1]:
        return "mid"
    return "upper-tail"


def run(con, *, start_ts=None, label="FULL"):
    syms = spot_leg_markets(con)
    print(f"\n{'='*96}\nH2 net-carry — {label} window  ({len(syms)} Binance-tail markets WITH a spot leg)\n{'='*96}")
    for tau in TAUS:
        recs = [harvest_one(con, sym, tau, start_ts=start_ts) for sym in syms]
        recs = [r for r in recs if r]
        df = pd.DataFrame(recs).sort_values("med_dv").reset_index(drop=True)
        if df.empty:
            print(f"  tau={tau:.0%}: no market has enough bars in this window"); continue
        # liquidity-band edges at the tercile boundaries of median per-bar dv
        e1, e2 = df["med_dv"].quantile([1/3, 2/3]).tolist()
        df["band"] = df["med_dv"].map(lambda d, e1=e1, e2=e2: band_of(d, (e1, e2)))
        print(f"\n--- tau={tau:.0%}  (arm when trailing funding-APR >= {tau:.0%}; monthly cadence) ---")
        show = df.copy()
        show["med_dv"] = show["med_dv"].map(lambda v: f"{v:,.0f}")
        for c in ("gross_apr", "net_apr", "worst_bar", "maxdd"):
            show[c] = show[c].map(lambda v: f"{v:+.3f}")
        show["arm_rate"] = show["arm_rate"].map(lambda v: f"{v:.2f}")
        show["mean_slip_bps"] = show["mean_slip_bps"].map(lambda v: f"{v:.0f}")
        print(show[["band", "symbol", "med_dv", "years", "arm_rate", "mean_slip_bps",
                    "gross_apr", "net_apr", "worst_bar", "maxdd"]].to_string(index=False))
        print("  [band means]  net-APR by liquidity band (the exhibit — robust ≤0 ?):")
        for b in ("illiquid", "mid", "upper-tail"):
            g = df[df.band == b]
            if len(g):
                print(f"    {b:<11} n={len(g)}  gross_APR={g.gross_apr.mean():+.3f}  "
                      f"net_APR={g.net_apr.mean():+.3f}  worst_bar={g.worst_bar.min():+.3f}  "
                      f"maxDD={g.maxdd.min():+.3f}")
        print(f"  [pooled]  gross_APR={df.gross_apr.mean():+.3f}  net_APR={df.net_apr.mean():+.3f}  "
              f"frac markets net>0: {(df.net_apr>0).mean():.2f}")


def main():
    con = duckdb.connect(str(_DUCKDB), read_only=True)
    run(con, label="FULL history")
    run(con, start_ts=JAN_2025, label="FORWARD 2025+ (the deployable regime, mandate §4b)")
    con.close()


if __name__ == "__main__":
    main()
