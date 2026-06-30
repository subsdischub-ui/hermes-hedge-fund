# H2 — PRE-REGISTRATION (frozen before run; Malachie-approved 2026-06-30)

## Cost model (single, central, conservative) — √-impact + base spread
round_trip_cost(dv_bar) = clip( taker_floor,  taker_floor + k * sqrt(Q / daily_dv),  cap )
  where daily_dv = dv_bar * BARS_PER_DAY_4H (6)
  taker_floor = 0.0035   # 35 bps round-trip = alpha's TAKER; the liquid-major anchor (never pay less than fees)
  Q           = 50_000   # $ notional per position; a SMALL book = steelmans the small-book/illiquid thesis
  k           = 0.025    # single central impact coefficient
  cap         = 0.025    # 250 bps round-trip; a micro-cap cannot diverge to infinity
Guards (Malachie): liquid major (~$100M+/day) -> ~floor (35 bps); illiquid pays more via sqrt-impact;
floor = taker fee; cap explicit; ONE k (H1 already decided the verdict, H2 is confirmatory).
Slippage is charged on the TRAILING-median dollar-volume (causal, leak-guarded), half at open + half at close.

## Grid (liquidity bands x config) — registered to BookTrialRegistry(crypto)
Universe = the 13 Binance-tail markets that HAVE a spot leg (cash-and-carry constructible).
  Bands (by median per-bar dollar-volume): illiquid | mid | upper-tail   (3 bands)
  tau_apr in {0.05, 0.10}   (arm when trailing funding-APR >= tau)        (2 thresholds)
  cadence = 180 bars (monthly), lookback = 180 bars (alpha's low-whipsaw choice)  (fixed)
  => 3 bands x 2 tau = 6 cells.  Register +6 (crypto 1900 -> 1906).
Robustness-ACROSS-bands (not best-cell). Per-DECILE net-carry reported as the exhibit.
HL-exotic tail is H1-only (no co-located spot leg in the frozen lake -> not harvestable here).
