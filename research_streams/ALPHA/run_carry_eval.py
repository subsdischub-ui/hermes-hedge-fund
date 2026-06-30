"""Boucle-alpha §3 two-axis evaluation — 8-cell grid + baselines, 3 cost models.

Axis A (overfitting): plateau across grid, DSR deflated by grid size, subperiod stability,
LOO-by-coin. Axis B (structural): net by FUNDING REGIME, fraction-of-time net-positive,
drawdown in negative-funding stretches, basis-blowout tail, forward-relevance (2024/25/26).

Built on the TDD'd carry_harvest primitives. Read-only on the frozen lake.
"""
import duckdb
import numpy as np
import pandas as pd

from hermes.learning.stats import deflated_sharpe_ratio, sharpe_ratio
from hermes.research import carry_harvest as ch

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

LAKE = "/Users/lacarbonnerie/Developer/hermes-alpha-carry/data/hermes_datalake.duckdb"
con = duckdb.connect(LAKE, read_only=True)

UNIV = ['ADA','APT','ARB','ATOM','AVAX','BCH','BNB','BTC','DOGE','DOT','ETH','FIL',
        'INJ','LINK','LTC','NEAR','OP','SOL','SUI','TRX','UNI','WIF','XRP']
SLASH = [f"{c}/USDT" for c in UNIV]

# honest round-trip cost models (fractional, open+close, both legs)
COSTS = {"maker_11bps": 0.0011, "s3paper_20bps": 0.0020, "taker_35bps": 0.0035}
BPY = ch.BARS_PER_YEAR_4H  # 2190

W, M = ch.WEEKLY_BARS, ch.MONTHLY_BARS  # 42, 180

# ---- the pre-registered 8-cell grid (lookback = cadence) -------------------- #
CC_GRID = {
    "CC_t05_weekly":  ch.CarryConfig(0.05, W, W, 0.0),
    "CC_t05_monthly": ch.CarryConfig(0.05, M, M, 0.0),
    "CC_t10_weekly":  ch.CarryConfig(0.10, W, W, 0.0),
    "CC_t10_monthly": ch.CarryConfig(0.10, M, M, 0.0),
    "CC_t15_weekly":  ch.CarryConfig(0.15, W, W, 0.0),
    "CC_t15_monthly": ch.CarryConfig(0.15, M, M, 0.0),
}
XV_GRID = {
    "XV_d10_weekly": ch.CarryConfig(0.10, W, W, 0.0),
    "XV_d20_weekly": ch.CarryConfig(0.20, W, W, 0.0),
}
# references (NOT counted as trials)
BASELINES = {
    "BASE_alwayson_pos": ch.CarryConfig(0.0, W, W, 0.0),     # always-on when funding>=0
    "BASE_buyhold_all":  ch.CarryConfig(-1e9, W, W, 0.0),    # never disarm = raw buy&hold carry
}

print("Loading CC streams (23 coins)...")
cc_streams = {c: ch.cash_and_carry_stream(con, s) for c, s in zip(UNIV, SLASH)}
print("Loading XV streams (23 coins, HL klines 2024-03+)...")
xv_streams = {}
for c in UNIV:
    try:
        xv_streams[c] = ch.cross_venue_stream(con, c)
    except Exception as e:
        print(f"  XV skip {c}: {e}")


def year_of(ts_ms):
    return pd.to_datetime(ts_ms, unit="ms").year


def evaluate(streams, cfg, cost):
    c2 = ch.CarryConfig(cfg.tau_apr, cfg.cadence_bars, cfg.lookback_bars, cost)
    out = ch.build_book(streams, c2)
    ts, book, bf = out["ts"], out["book_ret"], out["book_funding"]
    yrs = np.array([year_of(t) for t in ts])
    armed_any = np.vstack([out["per_coin_armed"][c] for c in streams]).any(axis=0)
    res = {
        "total_ret": float(book.sum()),
        "ann_ret": float(book.sum() / (len(book) / BPY)),     # simple annualized
        "sharpe_ann": float(sharpe_ratio(book.tolist()) * np.sqrt(BPY)),
        "util": float(armed_any.mean()),                       # frac of bars with >=1 coin armed
        "frac_bar_pos": float((book > 0).mean()),
        "maxdd": _maxdd(book),
        "worst_bar": float(book.min()),
        "p01_bar": float(np.percentile(book, 1)),
        "by_year": {int(y): float(book[yrs == y].sum()) for y in sorted(set(yrs))},
        # Axis-B regime split on the book's own funding sign
        "regime": ch.regime_split(book, np.sign(bf)),
        "ret_when_negfund": float(book[bf < 0].sum()),         # P&L while net-paying funding
        "n_negfund_bars": int((bf < 0).sum()),
    }
    return res, out


def _maxdd(book):
    eq = np.cumsum(book)            # additive equity (returns are small, sum ~ cum)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def loo_by_coin(streams, cfg, cost):
    """Drop each coin; recompute always-on annualized net. Is the book carried by 1-2 coins?"""
    full, _ = evaluate(streams, cfg, cost)
    base = full["ann_ret"]
    deltas = {}
    for drop in streams:
        sub = {k: v for k, v in streams.items() if k != drop}
        r, _ = evaluate(sub, cfg, cost)
        deltas[drop] = base - r["ann_ret"]    # how much that coin contributed to ann_ret
    return base, deltas


# =============================== CC GRID ===================================== #
print("\n" + "=" * 100)
print("CASH-AND-CARRY (CC) — 6-cell grid + baselines, by cost model")
print("=" * 100)
for cost_name, cost in COSTS.items():
    print(f"\n----- cost = {cost_name} ({cost*1e4:.0f} bps round-trip) -----")
    rows = []
    for name, cfg in {**BASELINES, **CC_GRID}.items():
        r, _ = evaluate(cc_streams, cfg, cost)
        rows.append({
            "variant": name, "ann%": round(100*r["ann_ret"], 2), "tot%": round(100*r["total_ret"], 1),
            "Sharpe": round(r["sharpe_ann"], 2), "util%": round(100*r["util"], 1),
            "maxDD%": round(100*r["maxdd"], 1), "worstbar%": round(100*r["worst_bar"], 2),
            "negfundP&L%": round(100*r["ret_when_negfund"], 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))

# per-year net APR for the 6 CC cells + baseline, at s3paper cost (the deploy assumption)
print("\n----- CC per-year net return (%) @ s3paper_20bps (forward-relevance) -----")
yr_rows = []
for name, cfg in {**BASELINES, **CC_GRID}.items():
    r, _ = evaluate(cc_streams, cfg, COSTS["s3paper_20bps"])
    row = {"variant": name}
    row.update({str(y): round(100*v, 2) for y, v in r["by_year"].items()})
    yr_rows.append(row)
print(pd.DataFrame(yr_rows).fillna("").to_string(index=False))

# Axis-B regime split (CC) @ s3paper
print("\n----- CC funding-regime split @ s3paper_20bps (mean per-bar bps; Axis B) -----")
reg_rows = []
for name, cfg in {**BASELINES, **CC_GRID}.items():
    r, _ = evaluate(cc_streams, cfg, COSTS["s3paper_20bps"])
    reg_rows.append({
        "variant": name,
        "mean_pos_bps": round(1e4*r["regime"]["mean_pos"], 3),
        "mean_neg_bps": round(1e4*r["regime"]["mean_neg"], 3),
        "frac_pos_fund": round(r["regime"]["frac_pos_funding"], 3),
        "frac_bar_pos": round(r["frac_bar_pos"], 3),
    })
print(pd.DataFrame(reg_rows).to_string(index=False))

# Axis-A: DSR deflated by grid size (8), plateau, LOO
print("\n----- Axis-A: DSR (deflated by grid=8) + plateau, CC cells @ s3paper -----")
cell_sharpes_ann = []   # for the plateau display only
cell_sharpes_bar = []   # PER-BAR sharpes — same time-scale as the per-bar returns fed to DSR
cell_books = {}
for name, cfg in CC_GRID.items():
    r, out = evaluate(cc_streams, cfg, COSTS["s3paper_20bps"])
    cell_sharpes_ann.append(r["sharpe_ann"])
    cell_sharpes_bar.append(sharpe_ratio(out["book_ret"].tolist()))
    cell_books[name] = out["book_ret"]
# UNITS FIX: deflated_sharpe_ratio computes PSR on PER-BAR returns; its expected_max threshold
# is in the SAME units as sr_variance. Passing variance of ANNUALIZED Sharpes (~0.2) against
# per-bar returns (SR~0.10) collapses DSR to 0 trivially. Use the PER-BAR Sharpe variance.
sr_var_bar = float(np.var(cell_sharpes_bar, ddof=1))
print(f"CC cell annualized Sharpes: {[round(s,2) for s in cell_sharpes_ann]} "
      f"(per-bar: {[round(s,3) for s in cell_sharpes_bar]})")
print(f"plateau: min={min(cell_sharpes_ann):.2f} max={max(cell_sharpes_ann):.2f} "
      f"(spread={max(cell_sharpes_ann)-min(cell_sharpes_ann):.2f})")
med_name = sorted(CC_GRID, key=lambda n: evaluate(cc_streams, CC_GRID[n], COSTS["s3paper_20bps"])[0]["sharpe_ann"])[len(CC_GRID)//2]
med_book = cell_books[med_name]
dsr8 = deflated_sharpe_ratio(med_book.tolist(), sr_var_bar if sr_var_bar > 0 else 1e-6, 8)
print(f"median cell={med_name}; DSR(per-bar units, deflated by 8) = {dsr8:.3f}")
print("NB: DSR is a sanity-check only; for a carry harvest significance was never the question."
      " The annualized Sharpe ~4-5 is meaningless — smooth funding collapses denominator vol while"
      " underweighting the left tail (basis blowout / funding flip).")

print("\n----- Axis-A: LOO-by-coin (always-on @ s3paper); contribution to ann% -----")
base_ann, deltas = loo_by_coin(cc_streams, BASELINES["BASE_alwayson_pos"], COSTS["s3paper_20bps"])
ser = pd.Series({k: round(100*v, 3) for k, v in deltas.items()}).sort_values(ascending=False)
print(f"always-on ann% = {100*base_ann:.2f}; per-coin contribution to ann% (drop-delta):")
print(ser.to_string())

# =============================== XV GRID ===================================== #
print("\n" + "=" * 100)
print("CROSS-VENUE (XV) — 2-cell grid + buy&hold, HL klines 2024-03+ only")
print("=" * 100)
if xv_streams:
    for cost_name, cost in COSTS.items():
        print(f"\n----- cost = {cost_name} ({cost*1e4:.0f} bps) -----")
        rows = []
        for name, cfg in {"XV_buyhold": ch.CarryConfig(-1e9, W, W, 0.0), **XV_GRID}.items():
            r, _ = evaluate(xv_streams, cfg, cost)
            rows.append({
                "variant": name, "ann%": round(100*r["ann_ret"], 2),
                "Sharpe": round(r["sharpe_ann"], 2), "util%": round(100*r["util"], 1),
                "maxDD%": round(100*r["maxdd"], 1),
                "negfundP&L%": round(100*r["ret_when_negfund"], 1),
            })
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n----- XV per-year @ s3paper_20bps -----")
    yr_rows = []
    for name, cfg in {"XV_buyhold": ch.CarryConfig(-1e9, W, W, 0.0), **XV_GRID}.items():
        r, _ = evaluate(xv_streams, cfg, COSTS["s3paper_20bps"])
        row = {"variant": name}
        row.update({str(y): round(100*v, 2) for y, v in r["by_year"].items()})
        yr_rows.append(row)
    print(pd.DataFrame(yr_rows).fillna("").to_string(index=False))

# =============================== ROBUSTNESS EXHIBITS ========================= #
# (added after red-team: era attribution, annualized forward run-rate, oracle coin-selection
#  ceiling — to show the negative is robust in BOTH directions and not a presentation artifact)
print("\n" + "=" * 100)
print("ROBUSTNESS EXHIBITS (red-team)")
print("=" * 100)

# (E1) era attribution — how much of the headline is bull-era?
r_on, out_on = evaluate(cc_streams, BASELINES["BASE_alwayson_pos"], COSTS["maker_11bps"])
ts = out_on["ts"]; book = out_on["book_ret"]
yrs = np.array([year_of(t) for t in ts])
cum = book.sum()
era = {
    "2019-2021": float(book[yrs <= 2021].sum()),
    "2022-2023": float(book[(yrs >= 2022) & (yrs <= 2023)].sum()),
    "2024": float(book[yrs == 2024].sum()),
    "2025-2026H1 (FORWARD)": float(book[yrs >= 2025].sum()),
}
print(f"\n(E1) always-on cum return (maker) = {100*cum:.1f}%; contribution by era:")
for k, v in era.items():
    print(f"   {k:24s}: {100*v:6.1f}%  ({100*v/cum:4.0f}% of cum)")

# (E2) annualized forward run-rate (raw per-year sums under-annualize the H1-2026 stub)
m26 = yrs == 2026
m25 = yrs == 2025
yrs_frac_26 = m26.sum() / BPY
print("\n(E2) forward run-rate (annualized), always-on maker:")
print(f"   2025 net = {100*book[m25].sum():.2f}% (full year)")
print(f"   2026-H1 raw sum = {100*book[m26].sum():.2f}% over {yrs_frac_26:.2f}yr "
      f"-> annualized run-rate = {100*book[m26].sum()/yrs_frac_26:.2f}%")

# (E3) oracle coin-selection ceiling — drop the full-sample LOO-negative coins (LOOK-AHEAD,
#      not deployable) and see if the FORWARD lifts. If even the oracle doesn't help -> the
#      iter-1 coin-selection lever has ~zero forward value (bull-era cleanup only).
DROP = ["SOL", "TRX", "BNB", "BTC", "BCH", "AVAX"]   # full-sample LOO-negative/drag
keep = {c: s for c, s in cc_streams.items() if c not in DROP}
r_sel, out_sel = evaluate(keep, BASELINES["BASE_alwayson_pos"], COSTS["maker_11bps"])
ys = np.array([year_of(t) for t in out_sel["ts"]]); bs = out_sel["book_ret"]
print(f"\n(E3) oracle keep-{len(keep)} (drop {DROP}, LOOK-AHEAD) vs naive-23 (maker):")
print(f"   headline ann%: naive {100*r_on['ann_ret']:.2f}  vs  oracle {100*r_sel['ann_ret']:.2f}")
for y in (2024, 2025, 2026):
    nv = book[yrs == y].sum(); ov = bs[ys == y].sum()
    print(f"   {y}: naive {100*nv:6.2f}%  oracle {100*ov:6.2f}%  (forward lift {100*(ov-nv):+.2f}pp)")

con.close()
print("\nDONE.")
