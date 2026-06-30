"""Boucle β — H2 reverse-carry CLOSURE (red-team steelman, reproduced + closed).

The illiquid tail is dominated by LARGE NEGATIVE funding. A sign-agnostic REVERSE carry
(SHORT spot + LONG perp) harvests that negative funding and LOOKS very profitable. This
driver reproduces it AND shows it is the SHORT-BORROW risk premium, not deployable carry:
to short the spot you must borrow the (hard-to-borrow) illiquid token, and by perp-spot
no-arbitrage the borrow rate ~ the negative funding itself. Charging borrow = lambda x
captured-funding collapses the apparent edge; the lake cannot observe borrow, so it cannot
be claimed. This visibly SHUTS the bi-directional door (rather than leaving it untested).

Run: .venv/bin/python research_streams/BETA/run_h2_reverse.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from hermes.datalake import liquidity as liq
from hermes.research import carry_harvest as ch
from hermes.research import illiquid_carry as ic

_DUCKDB = Path(__file__).resolve().parents[2] / "data" / "hermes_datalake.duckdb"
TAIL_SPOT = dict(spot_view="binance_tail_spot_klines_4h", perp_view="binance_tail_funding_on_4h",
                 native_view="binance_tail_funding_native")
LOOKBACK = ch.MONTHLY_BARS
JAN_2025 = 1_735_689_600_000
TAU = 0.05
LAMBDAS = (0.0, 0.5, 0.7, 1.0)


def reverse_one(con, symbol: str, *, start_ts: int) -> dict:
    """SHORT-spot + LONG-perp reverse carry: harvest NEGATIVE funding, charge sqrt-impact."""
    stream = ch.cash_and_carry_stream(con, symbol, **TAIL_SPOT)
    dv = con.execute("SELECT ts, volume*close dv FROM binance_tail_funding_on_4h "
                     "WHERE symbol = ? ORDER BY ts", [symbol]).df()
    s = stream.merge(dv, on="ts", how="left").sort_values("ts").reset_index(drop=True)
    s = s[s["ts"] >= start_ts].reset_index(drop=True)
    if len(s) < LOOKBACK + 5:
        return {}
    s["dv"] = s["dv"].ffill().bfill()
    s = liq.attach_trailing_liquidity(s.assign(symbol=symbol).rename(columns={"dv": "dollar_volume"}),
                                      lookback_bars=LOOKBACK)
    slip = ic.liquidity_slippage(s["trail_dollar_volume"].to_numpy())
    tdf = ch.add_trailing_funding_apr(s, LOOKBACK, col="funding")
    # reverse: arm when trailing funding-APR <= -TAU  (feed -trail to the >=TAU arm logic)
    rev_armed = ch.arm_schedule(-tdf["trail_apr"].to_numpy(), TAU, ch.MONTHLY_BARS, warmup_bars=LOOKBACK)
    funding = s["funding"].to_numpy()
    basis_ret = s["basis_ret"].to_numpy()
    # reverse position collects -funding (long perp receives when funding<0); basis flips sign
    net = ic.gated_net_returns_variable_cost(-basis_ret, -funding, rev_armed, slip)
    captured = float((-funding)[rev_armed].sum())   # the harvested (positive) funding = borrow premium
    years = max((s["ts"].iloc[-1] - s["ts"].iloc[0]) / (1000 * 3600 * 24 * 365), 1e-9)
    return {"symbol": symbol, "net_sum": float(net.sum()), "captured": captured, "years": years,
            "arm_rate": float(rev_armed.mean())}


def main() -> None:
    con = duckdb.connect(str(_DUCKDB), read_only=True)
    syms = con.execute("""SELECT u.symbol FROM binance_tail_universe u
        WHERE EXISTS (SELECT 1 FROM binance_tail_spot_klines_4h s WHERE s.symbol=u.symbol)""").df()["symbol"].tolist()
    recs = [r for r in (reverse_one(con, s, start_ts=JAN_2025) for s in syms) if r]
    print("=" * 80)
    print("REVERSE-CARRY (short-spot/long-perp) on the NEGATIVE-funding tail — FORWARD 2025+")
    print("=" * 80)
    print("per-market forward net APR (raw, gross of spot-borrow):")
    for r in sorted(recs, key=lambda x: -x["net_sum"] / x["years"]):
        print(f"  {r['symbol']:<16} net_APR={r['net_sum']/r['years']:+.3f}  "
              f"captured_funding_APR={r['captured']/r['years']:+.3f}  arm={r['arm_rate']:.2f}")
    # pooled net APR at each no-arb borrow level (borrow = lambda * captured funding)
    print("\nThe captured funding IS the short-borrow premium (no-arbitrage). Charge borrow = lambda x captured:")
    for lam in LAMBDAS:
        pooled = np.mean([(r["net_sum"] - lam * r["captured"]) / r["years"] for r in recs])
        fracpos = np.mean([(r["net_sum"] - lam * r["captured"]) > 0 for r in recs])
        tag = "RAW (borrow unobservable in lake)" if lam == 0 else f"borrow=lambda={lam}"
        print(f"  {tag:<34} pooled net APR = {pooled:+.4f}   frac markets >0 = {fracpos:.2f}")
    print("\nCLOSURE: the apparent reverse-carry edge IS the unobservable borrow rent; at full no-arb")
    print("borrow (lambda=1.0) it is NEGATIVE. The lake cannot see borrow -> the edge cannot be claimed.")
    con.close()


if __name__ == "__main__":
    main()
