"""New-data SHADOW sleeves (research candidates, log-only — NOT in allowed_skills).

Two confirmed slow-but-orthogonal edges found in data OUTSIDE the OHLCV+funding feed
that the prior ~11 searches exhausted. Both passed screen + adversarial verify
(Monte-Carlo nulls, look-ahead audit, cost death, orthogonality). They are SLOW
(~50-83 trades/yr) and BTC/ETH-only, so they do NOT add frequency — they are
DIVERSIFIERS (orthogonal to the live momentum/funding sleeves).

They are registered in the library so the dashboard shows their live gauge, but are
deliberately kept OUT of ``strategy.allowed_skills`` so they are computed + logged
every tick yet can NEVER be routed to execution (the selective gate rejects any
skill not in the slate — see hermes.risk.selective). This is the proven SHADOW
pattern (cf. funding_carry_harvest): accrue real out-of-sample evidence, size zero,
promote only after the edge holds forward.

Each is a causal ``fn(df) -> SkillSignal`` that reads a single pre-computed,
causally-lagged feature column surfaced by the live data layer (the heavy
timeframe-specific history + z-score lives in the data source, mirroring how
funding_rate is attached). When the column is absent (data unavailable), the sleeve
HOLDs gracefully, so the system is safe and the test suite stays offline-green.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.skills.library import SkillSignal


def _hold(skill_id: str, reason: str, feats: dict[str, float] | None = None) -> SkillSignal:
    return SkillSignal(skill_id, "HOLD", 0.0, 0.02, 0.04, reason, feats or {})


# --------------------------------------------------------------------------- #
# options_iv_dvol_reversion — Deribit DVOL fear/complacency reversion (BTC/ETH).
#   Confirmed: LONG spot when DVOL 90d z-score >= +2 (fear spike -> oversold),
#   SHORT when <= -2 (complacency), ~24h hold. rho -0.26 vs momentum / +0.21 vs
#   funding (orthogonal). The DVOL z-score is computed causally (1-bar-lagged DVOL —
#   see research_options/deribit_iv.py) in the data layer and surfaced as the
#   `dvol_z_90d` column.
#   HONEST RE-AUDIT (2026-06-24). The net bps/trade reproduce (BTC +24, ETH +50) and it
#   beats the per-asset random-entry null (BTC mc_p=0.017, ETH 0.0005) — BUT surviving the
#   null != confirmed. It FAILS the MANDATORY multiple-testing DEFLATION (gate §3.7):
#   per-trade Sharpe ~0.099 < the ~0.154 grid-max-under-null (only ~7/96 grid configs beat
#   it), so the apparent edge does not clear the deflated bar. Verdict: NOT confirmed ->
#   REAL-BUT-FRAGILE. The honest PER-TRADE OOS Sharpe is ~0.08-0.15, NOT the old ~1.04 (an
#   annualized/pooled basis — stale). BTC/ETH ~0.8 correlated (sleeve streams ~0.58) ->
#   ~1.4 effective bets, not 2 (size as ~1 correlated bet if ever promoted). Stays SHADOW,
#   size 0 (zero capital risked) — promotion requires beating the null AND deflation AND a
#   beta-controlled date-level basket null.
# --------------------------------------------------------------------------- #
_DVOL_Z = 2.0          # frozen a-priori (validated ±2.0; smooth monotone in [1.0, 2.5])
_DVOL_HOLD_BARS = 24   # ~24h hold (the validated horizon)


def options_iv_dvol_reversion(df: pd.DataFrame) -> SkillSignal:
    sid = "options_iv_dvol_reversion"
    if "dvol_z_90d" not in df.columns or len(df) < 1:
        return _hold(sid, "missing dvol_z_90d (options IV data not attached / non-BTC/ETH)")
    z = float(df["dvol_z_90d"].iloc[-1])
    if not np.isfinite(z):
        return _hold(sid, "dvol_z_90d not finite", {"dvol_z": 0.0})
    feats = {"dvol_z": z, "hold_bars": float(_DVOL_HOLD_BARS)}
    if z >= _DVOL_Z:                       # fear spike -> mean-revert UP -> long
        return SkillSignal(
            sid, "BUY", min(1.0, z / 4.0), 0.04, 0.08,
            f"DVOL z={z:.2f} >= {_DVOL_Z} (fear/oversold) -> long ~{_DVOL_HOLD_BARS}h",
            feats,
        )
    if z <= -_DVOL_Z:                      # complacency -> mean-revert DOWN -> short
        return SkillSignal(
            sid, "SELL", min(1.0, abs(z) / 4.0), 0.04, 0.08,
            f"DVOL z={z:.2f} <= -{_DVOL_Z} (complacency) -> short ~{_DVOL_HOLD_BARS}h",
            feats,
        )
    return _hold(sid, f"DVOL z={z:.2f} within +/-{_DVOL_Z} (no IV extreme)", feats)


# --------------------------------------------------------------------------- #
# onchain_stablecoin_growth — on-chain stablecoin dry-powder (market-wide).
#   Confirmed: USDT+USDC 60d circulating-supply-growth z-score; rising supply (dry
#   powder) -> LONG majors, contracting -> SHORT, ~3d hold. Net taker +46/+63bps
#   (BTC/ETH), rho ~0 vs funding. The 60d growth z is computed causally (>=1-day
#   publication lag) in the data layer and surfaced market-wide as the
#   `stbl_supply_growth_z` column (same value for every symbol).
#   HONEST RE-AUDIT (2026-06-24). Net +42/+63bps reproduce and it beats the per-asset
#   random-entry null (BTC mc_p=0.003, ETH 0.0035). The earlier 'basket p~0' was an
#   ARTIFACT of ~1972 OVERLAPPING daily crossings counted as independent (now fixed:
#   fxgate.event_basket_gate de-overlaps -> ~697 events; the de-overlapped basket null is
#   ~0.002 on daily bars, borderline/failing at the production >=3d lag under a beta-
#   controlled null). DECISIVE: it FAILS the MANDATORY multiple-testing DEFLATION (§3.7,
#   Bonferroni p=1.0 over the config grid). Verdict: NOT confirmed -> DEMOTED. The honest
#   PER-TRADE OOS Sharpe is ~0.12/0.13, NOT the old 1.07/1.21 (an annualized/pooled basis
#   — stale). BTC/ETH ~0.8 correlated (sleeve streams ~0.76) -> ~1.3 effective bets, not
#   2. Stays SHADOW, size 0 (zero capital risked).
# --------------------------------------------------------------------------- #
_STBL_Z = 1.0          # frozen a-priori (validated win60 z_thr1.0)
_STBL_HOLD_BARS = 18   # ~3 days on 4h bars (the validated horizon)


def onchain_stablecoin_growth(df: pd.DataFrame) -> SkillSignal:
    sid = "onchain_stablecoin_growth"
    if "stbl_supply_growth_z" not in df.columns or len(df) < 1:
        return _hold(sid, "missing stbl_supply_growth_z (on-chain data not attached)")
    z = float(df["stbl_supply_growth_z"].iloc[-1])
    if not np.isfinite(z):
        return _hold(sid, "stbl_supply_growth_z not finite", {"stbl_z": 0.0})
    feats = {"stbl_z": z, "hold_bars": float(_STBL_HOLD_BARS)}
    if z >= _STBL_Z:                       # supply expanding (dry powder) -> long
        return SkillSignal(
            sid, "BUY", min(1.0, z / 2.0), 0.05, 0.10,
            f"stablecoin supply growth z={z:.2f} >= {_STBL_Z} (dry powder) -> long ~3d",
            feats,
        )
    if z <= -_STBL_Z:                      # supply contracting -> short
        return SkillSignal(
            sid, "SELL", min(1.0, abs(z) / 2.0), 0.05, 0.10,
            f"stablecoin supply growth z={z:.2f} <= -{_STBL_Z} (contraction) -> short ~3d",
            feats,
        )
    return _hold(sid, f"stablecoin growth z={z:.2f} within +/-{_STBL_Z} (no extreme)", feats)
