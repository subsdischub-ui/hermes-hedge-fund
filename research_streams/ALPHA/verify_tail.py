"""Adversarial verification of the verdict-bearing claims:
  (1) basis-blowout tail — are the worst per-bar basis moves REAL events or data glitches?
  (2) forward-regime compression — independent per-coin 2025/2026 net cross-check.
  (3) the on/off rule's value-add in 2026 (dodging negative funding).
"""
import duckdb
import numpy as np
import pandas as pd

from hermes.research import carry_harvest as ch

pd.set_option("display.width", 200)
LAKE = "/Users/lacarbonnerie/Developer/hermes-alpha-carry/data/hermes_datalake.duckdb"
con = duckdb.connect(LAKE, read_only=True)
UNIV = ['ADA','APT','ARB','ATOM','AVAX','BCH','BNB','BTC','DOGE','DOT','ETH','FIL',
        'INJ','LINK','LTC','NEAR','OP','SOL','SUI','TRX','UNI','WIF','XRP']

print("=== (1) BASIS-BLOWOUT TAIL: worst single-bar (spot_ret - perp_ret) across coins ===")
worst = []
for c in UNIV:
    s = ch.cash_and_carry_stream(con, f"{c}/USDT")
    aligned = con.execute(
        "SELECT ts FROM binance_spot_klines_4h WHERE symbol = ? ORDER BY ts", [f"{c}/USDT"]
    ).df()
    basis = s["basis_ret"].to_numpy()
    ts = s["ts"].to_numpy()
    i = int(np.argmin(basis))
    worst.append({"coin": c, "worst_basis_bps": round(1e4*basis[i], 1),
                  "date": str(pd.to_datetime(ts[i], unit="ms")),
                  "p01_bps": round(1e4*np.percentile(basis, 1), 1),
                  "p99_bps": round(1e4*np.percentile(basis, 99), 1)})
wdf = pd.DataFrame(worst).sort_values("worst_basis_bps")
print(wdf.to_string(index=False))
print(f"\nbook-worst single basis bar: {wdf['worst_basis_bps'].min()} bps "
      f"({wdf.iloc[0]['coin']} {wdf.iloc[0]['date']})")

print("\n=== (2) per-coin 2026-H1 net funding (gross, from native) — is the compression real? ===")
df = con.execute("""
  SELECT symbol, ROUND(100*SUM(funding_rate),2) gross_2026_pct, COUNT(*) n
  FROM binance_funding_native
  WHERE EXTRACT(year FROM epoch_ms(ts))=2026 AND symbol IN ({})
  GROUP BY symbol ORDER BY gross_2026_pct
""".format(",".join(f"'{c}/USDT'" for c in UNIV))).df()
print(df.to_string(index=False))
print(f"\n2026-H1 book-median gross funding: {df['gross_2026_pct'].median():.2f}%  "
      f"(coins net-negative: {(df['gross_2026_pct']<0).sum()}/{len(df)})")

print("\n=== (3) on/off value-add 2026: gated-always-on vs raw buy&hold, per quarter-ish ===")
# already shown at book level (+0.45 vs -0.71); confirm the gate sits in cash when funding<0
cfg_on = ch.CarryConfig(0.0, ch.WEEKLY_BARS, ch.WEEKLY_BARS, 0.0020)
streams = {c: ch.cash_and_carry_stream(con, f"{c}/USDT") for c in UNIV}
out = ch.build_book(streams, cfg_on)
ts = out["ts"]; yrs = np.array([pd.to_datetime(t, unit="ms").year for t in ts])
m26 = yrs == 2026
armed_any = np.vstack([out["per_coin_armed"][c] for c in UNIV]).any(axis=0)
print(f"2026 bars: {m26.sum()}; armed-any util in 2026: {100*armed_any[m26].mean():.1f}% "
      f"(gate sits in cash the rest)")
print(f"2026 book net (gated always-on, 20bps): {100*out['book_ret'][m26].sum():.2f}%")

con.close()
