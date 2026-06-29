"""STREAM 10 — meta-honesty / deflation + data-governance machinery.

Stream 10 is the adversarial guard: it does not *search* for alpha, it *attacks* the
alpha of every other stream and keeps the overfitting wall standing. This module makes
that mandate executable. It is research-only and side-effect-free at import; nothing here
ever trades.

It provides the five governance primitives the mandate names:

  10.1  ``PromotionBar`` — a HARDENED, per-book promotion gate. The Monte-Carlo
        random-entry null is a *floor* (a prerequisite), never the bar; the bar is the
        deflated Sharpe at the TRUE trial count (``BookTrialRegistry``) plus PBO, and it
        is made strictly stricter on the FX book than the crypto book. Acceptance is by
        leave-one-out Δ-Sharpe (book contribution), never the solo Sharpe.

        WHY per-book: FX round-trip cost (~1.6/2.1/7.1 bps) is an order of magnitude below
        crypto's (~15-20 bps), so *more* FX candidates clear the cost arithmetic — the
        overfitting wall REPLACES the cost wall on the FX book (metaprompt §3.7). The DSR
        must therefore deflate by the FX book's OWN cumulative trial count, not a global
        lump, and the FX deflation thresholds are tightened.

  10.2  ``DeadEndsLedger`` + ``AcquisitionMap`` — shared memory against re-spend. The
        ledger returns BLOCKED-BY-LEDGER when a stream proposes a falsified hypothesis
        without a genuinely NEW data axis. The acquisition map governs data-$ (CoinGlass
        acquired; Amberdata = arbitrate→NO pending Stream 4; Databento/alt-data = DO NOT
        SPEND; FRED/WGC/BIS/GLD/IAU = free/done).

  10.3  ``gp_lock_status`` — the explicit GP / feature-zoo (D1) lock. It stays LOCKED until
        (a) the deflation budget is trusted, (b) the Stream-1 book harness is trusted, and
        (c) a zoo of already-validated-orthogonal features exists. The lock cites the D1
        NO_EDGE evidence (OOS acc ≤ base rate; combined MC-null p=1.000).

All canonical state is embedded as module defaults (so the module is offline-green and
self-contained) and can be overridden from JSON artifacts under ``research_streams/S10/``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from hermes.learning.stats import deflated_sharpe_ratio

__all__ = [
    "CROSS_BAR",
    "CRYPTO_BAR",
    "D1_NO_EDGE",
    "DEFAULT_ACQUISITION_MAP",
    "DEFAULT_DEAD_ENDS",
    "DEFAULT_TRIAL_SEED",
    "FUNDING_SQUEEZE_STATUS",
    "FX_BAR",
    "AcquisitionMap",
    "Book",
    "BookTrialRegistry",
    "DataSource",
    "DataStatus",
    "DeadEnd",
    "DeadEndsLedger",
    "ForwardOOSEvidence",
    "LedgerVerdict",
    "PromotionBar",
    "PromotionThresholds",
    "PromotionVerdict",
    "SleeveEvidence",
    "Verdict",
    "bar_for_book",
    "deflated_sharpe_at_true_count",
    "funding_squeeze_status",
    "governance_registry_dir",
    "gp_lock_status",
]


# --------------------------------------------------------------------------- #
# v-next Task 1 — the SHARED, TRACKED registry location
# --------------------------------------------------------------------------- #
# The Round-1 registries were persisted under research_streams/S10/ — but that path is
# gitignored (`/research_*/`), so each worktree carried its OWN copy of the deflation count
# and the ledger. A per-worktree gitignored count cannot be the lab's authoritative state.
#
# The canonical state therefore lives in a COMMITTED, repo-root ``governance/`` directory
# (tracked → shared identically across every worktree via git). Every future trial bump goes
# through ``BookTrialRegistry()`` (no-arg → this dir) and is committed. Override with
# ``HERMES_GOVERNANCE_DIR`` for tests / alternative checkouts.
def governance_registry_dir() -> Path:
    """Repo-root ``governance/`` — the canonical, committed home of the registry state.

    Replaces the gitignored, per-worktree ``research_streams/S10/`` so the deflation count,
    the dead-ends ledger and the acquisition map are one shared, version-controlled truth.
    """
    env = os.environ.get("HERMES_GOVERNANCE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "governance"


# --------------------------------------------------------------------------- #
# enums
# --------------------------------------------------------------------------- #
class Book(StrEnum):
    CRYPTO = "crypto"
    FX_METALS = "fx_metals"
    CROSS = "cross"


class Verdict(StrEnum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"


class LedgerVerdict(StrEnum):
    BLOCKED = "BLOCKED-BY-LEDGER"
    ALLOWED_NEW_AXIS = "ALLOWED-NEW-AXIS"
    NOT_A_DEAD_END = "NOT-A-DEAD-END"


class DataStatus(StrEnum):
    ACQUIRED = "acquired"        # paid for / ingested — use freely, no new spend
    FREE = "free"               # $0 — use freely
    NOT_ACQUIRED = "not_acquired"  # spend is an open arbitrage (default recommendation: NO)
    DO_NOT_SPEND = "do_not_spend"  # explicitly forbidden — the thesis it serves is refuted


# --------------------------------------------------------------------------- #
# 10.1 — hardened per-book promotion bar
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PromotionThresholds:
    """A promotion bar for one book. ``min_dsr`` is a probability in [0,1] (the deflated
    Sharpe = P(true Sharpe > the n_trials deflation benchmark)); ``max_pbo`` is the
    Probability of Backtest Overfitting limit. The floor fields (``max_mc_null_p``,
    ``min_instruments``, ``require_beta_controlled``) are prerequisites, identical across
    books; the deflation fields (``max_pbo``, ``min_dsr``) are what we tighten on FX.
    """

    max_pbo: float
    min_dsr: float
    min_oos_sharpe: float = 0.3        # gate §3.5 (OOS Sharpe on last 40% by date, IN-SAMPLE split)
    max_abs_rho: float = 0.3           # gate §3.8 (|ρ| vs every existing sleeve)
    min_instruments: int = 2           # gate §3.5 cross-asset generalization (the killer filter)
    max_mc_null_p: float = 0.05        # gate §3.6 random-entry null — the FLOOR, not the bar
    require_beta_controlled: bool = True
    require_positive_loo: bool = True  # §7 — the unit is the BOOK (LOO-Δ-Sharpe), not solo
    # v-next Task 4 — the OOS-FORWARD precondition. The in-sample split above (last 40% by date)
    # is necessary but NOT sufficient: S0 survivorship (the losing tail of a long-only continuation
    # is absent by construction) and S2's single frozen lake establish that no in-sample result,
    # however clean, can CONFIRM an edge. Promotion requires clean forward OOS accrued in real regime.
    require_oos_forward: bool = True
    min_forward_trades: int = 20            # enough real-regime trades to be more than noise
    min_forward_window_days: int = 90        # accrued over a genuine forward window (Forward-Shadow)
    min_forward_sharpe: float = 0.0          # forward per-trade Sharpe must not be negative


# Policy (head-of-research, Stream 10.1). The floor is shared; the deflation layer is
# strictly stricter on FX (and at least as strict on cross-book) because cheap FX cost
# moves the binding constraint from cost to overfitting. These are the tunable knobs.
CRYPTO_BAR = PromotionThresholds(max_pbo=0.50, min_dsr=0.95)
FX_BAR = PromotionThresholds(max_pbo=0.40, min_dsr=0.975)
CROSS_BAR = PromotionThresholds(max_pbo=0.40, min_dsr=0.975)

_BARS: dict[Book, PromotionThresholds] = {
    Book.CRYPTO: CRYPTO_BAR,
    Book.FX_METALS: FX_BAR,
    Book.CROSS: CROSS_BAR,
}


def bar_for_book(book: Book) -> PromotionThresholds:
    """The promotion bar for ``book`` (defaults to the strictest if unknown)."""
    return _BARS.get(book, FX_BAR)


@dataclass(frozen=True)
class SleeveEvidence:
    """The adversarially-recomputed evidence for a candidate sleeve, re-gated under the
    TRUE trial count. ``dsr`` MUST already be the deflated Sharpe computed against the
    book's true trial count (see ``deflated_sharpe_at_true_count``)."""

    name: str
    book: Book
    mc_null_p: float          # §3.6 Monte-Carlo random-entry null (lower = better)
    n_instruments: int        # §3.5 distinct-channel instruments the rule generalizes on
    beta_controlled: bool     # bull/bear separated; NOT secular-bull-beta in disguise
    dsr: float                # §3.7 deflated Sharpe AT the true trial count
    pbo: float                # §3.7 probability of backtest overfitting
    max_abs_rho: float        # §3.8 max |ρ| vs every existing sleeve (dated/regime-level)
    loo_delta_sharpe: float   # §7 leave-one-out Δ-Sharpe contribution to the BOOK
    oos_sharpe: float = 0.0   # §3.5 OOS Sharpe on the last 40% by calendar date
    n_trials: int = 0         # the true trial count the dsr was deflated by (provenance)
    notes: str = ""


@dataclass(frozen=True)
class ForwardOOSEvidence:
    """Clean out-of-sample FORWARD evidence accrued in a real trading regime (the exit of
    Forward-Shadow), the hard precondition the in-sample gate cannot substitute for.

    ``regime_real`` asserts the forward window was a genuine live/paper regime (not a re-sliced
    in-sample tail). ``leak_replayed`` asserts the driver's leak-injection audit (AM-4) was
    RE-RUN on this survivor at merge — composition bugs fabricate positives, so a prior clean
    audit is not enough; it must be replayed on the exact merged artifact.
    """

    forward_sharpe: float       # per-trade Sharpe over the forward window (never annualized)
    n_trades: int               # number of real-regime forward trades
    window_days: int            # calendar span of the forward window
    regime_real: bool           # genuine forward regime, NOT a re-sliced in-sample tail
    leak_replayed: bool         # the AM-4 leak-injection audit was RE-RUN on this survivor at merge


@dataclass
class PromotionVerdict:
    sleeve: str
    book: str
    verdict: Verdict
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sleeve": self.sleeve,
            "book": self.book,
            "verdict": self.verdict.value,
            "passed": list(self.passed),
            "failed": list(self.failed),
            "reasons": dict(self.reasons),
            "thresholds": dict(self.thresholds),
        }


class PromotionBar:
    """Apply the hardened per-book gate to a candidate sleeve's adversarial evidence.

    A sleeve is PROMOTED only if EVERY criterion holds. The criteria split into:
      * the FLOOR (prerequisites) — random-entry null, cross-asset generalization,
        beta-control. Clearing the floor is necessary but NEVER sufficient.
      * the DEFLATION BAR — deflated Sharpe at the true trial count + PBO (stricter on FX).
      * orthogonality and the book test (LOO-Δ-Sharpe > 0).
    """

    def __init__(self, bars: dict[Book, PromotionThresholds] | None = None) -> None:
        self._bars = bars or _BARS

    def evaluate(self, evidence: SleeveEvidence) -> PromotionVerdict:
        bar = self._bars.get(evidence.book, FX_BAR)
        passed: list[str] = []
        failed: list[str] = []
        reasons: dict[str, str] = {}

        def record(key: str, ok: bool, reason: str) -> None:
            (passed if ok else failed).append(key)
            reasons[key] = reason

        # ---- FLOOR (prerequisites) -------------------------------------------------- #
        record(
            "mc_null_floor",
            evidence.mc_null_p < bar.max_mc_null_p,
            f"MC random-entry null p={evidence.mc_null_p:.4g} "
            f"(floor: must be < {bar.max_mc_null_p}) — a FLOOR, not the bar",
        )
        record(
            "generalization",
            evidence.n_instruments >= bar.min_instruments,
            f"generalizes on {evidence.n_instruments} instrument(s) "
            f"(need ≥ {bar.min_instruments}, distinct channels) — single-asset = artifact",
        )
        if bar.require_beta_controlled:
            record(
                "beta_control",
                evidence.beta_controlled,
                "bull/bear separated, not secular-beta"
                if evidence.beta_controlled
                else "NOT beta-controlled — possible secular-bull-beta in a costume",
            )

        # ---- DEFLATION BAR ---------------------------------------------------------- #
        record(
            "deflated_sharpe",
            evidence.dsr >= bar.min_dsr,
            f"DSR={evidence.dsr:.4g} at true n_trials={evidence.n_trials} "
            f"(need ≥ {bar.min_dsr} for the {evidence.book.value} book)",
        )
        record(
            "pbo",
            evidence.pbo <= bar.max_pbo,
            f"PBO={evidence.pbo:.4g} (need ≤ {bar.max_pbo} for the {evidence.book.value} book)",
        )

        # ---- ORTHOGONALITY + BOOK TEST ---------------------------------------------- #
        record(
            "orthogonality",
            evidence.max_abs_rho < bar.max_abs_rho,
            f"max|ρ|={evidence.max_abs_rho:.4g} vs existing sleeves (need < {bar.max_abs_rho})",
        )
        if bar.require_positive_loo:
            record(
                "loo_delta_sharpe",
                evidence.loo_delta_sharpe > 0.0,
                f"LOO-Δ-Sharpe={evidence.loo_delta_sharpe:.4g} "
                "(need > 0 — the BOOK must be better WITH the sleeve; solo Sharpe is irrelevant)",
            )
        if bar.min_oos_sharpe >= 0:  # 0.0 means "must be non-negative", NOT "skip the gate"
            record(
                "oos_sharpe",
                evidence.oos_sharpe >= bar.min_oos_sharpe,
                f"OOS Sharpe={evidence.oos_sharpe:.4g} (need ≥ {bar.min_oos_sharpe})",
            )

        verdict = Verdict.PROMOTE if not failed else Verdict.REJECT
        return PromotionVerdict(
            sleeve=evidence.name,
            book=evidence.book.value,
            verdict=verdict,
            passed=passed,
            failed=failed,
            reasons=reasons,
            thresholds={
                "max_pbo": bar.max_pbo,
                "min_dsr": bar.min_dsr,
                "max_mc_null_p": bar.max_mc_null_p,
                "min_instruments": bar.min_instruments,
                "max_abs_rho": bar.max_abs_rho,
                "min_oos_sharpe": bar.min_oos_sharpe,
            },
        )

    def evaluate_with_forward(
        self, evidence: SleeveEvidence, forward: ForwardOOSEvidence | None
    ) -> PromotionVerdict:
        """The FULL v-next promotion decision: the in-sample gate AND the OOS-forward precondition.

        This is what an actual promotion must call (``evaluate`` alone is only the in-sample
        screen). The forward precondition is strictly ADDITIVE — it can never rescue a sleeve that
        fails the in-sample deflation bar, and a sleeve that passes in-sample is still REJECTED
        until clean OOS-forward evidence has accrued in a real regime (Forward-Shadow exit) with
        the AM-4 leak audit replayed on the merged survivor.
        """
        verdict = self.evaluate(evidence)  # in-sample screen first (re-uses every gate above)
        bar = self._bars.get(evidence.book, FX_BAR)
        if not bar.require_oos_forward:
            return verdict

        def record(key: str, ok: bool, reason: str) -> None:
            (verdict.passed if ok else verdict.failed).append(key)
            verdict.reasons[key] = reason

        if forward is None:
            record(
                "oos_forward_present",
                False,
                "no clean OOS-forward accrued — the Forward-Shadow has not been exited. No in-sample "
                "result, however clean, can CONFIRM an edge (S0 survivorship; S2 single frozen lake).",
            )
        else:
            record(
                "oos_forward_present", True,
                f"forward evidence attached ({forward.n_trades} trades / {forward.window_days}d)",
            )
            record(
                "oos_forward_regime_real",
                forward.regime_real,
                "forward window is a genuine live/paper regime"
                if forward.regime_real
                else "forward window is NOT a real regime (a re-sliced in-sample tail does not count)",
            )
            record(
                "oos_forward_leak_replayed",
                forward.leak_replayed,
                "AM-4 leak-injection audit RE-RUN on this survivor at merge"
                if forward.leak_replayed
                else "AM-4 leak audit NOT replayed on the merged survivor — composition bugs "
                "fabricate positives, so the prior clean audit is insufficient",
            )
            record(
                "oos_forward_trades",
                forward.n_trades >= bar.min_forward_trades,
                f"forward trades={forward.n_trades} (need ≥ {bar.min_forward_trades})",
            )
            record(
                "oos_forward_window",
                forward.window_days >= bar.min_forward_window_days,
                f"forward window={forward.window_days}d (need ≥ {bar.min_forward_window_days}d)",
            )
            record(
                "oos_forward_sharpe",
                forward.forward_sharpe >= bar.min_forward_sharpe,
                f"forward Sharpe={forward.forward_sharpe:.4g} (need ≥ {bar.min_forward_sharpe})",
            )

        verdict.verdict = Verdict.PROMOTE if not verdict.failed else Verdict.REJECT
        verdict.thresholds.update(
            {
                "require_oos_forward": bar.require_oos_forward,
                "min_forward_trades": bar.min_forward_trades,
                "min_forward_window_days": bar.min_forward_window_days,
                "min_forward_sharpe": bar.min_forward_sharpe,
            }
        )
        return verdict


def deflated_sharpe_at_true_count(
    returns: Sequence[float], sr_variance: float, n_trials: int
) -> float:
    """The deflated Sharpe of ``returns`` deflated by the TRUE ``n_trials`` (from the
    ``BookTrialRegistry``), not the under-count a single backtest reports.

    This is the lever of 10.1: the same track record deflates harder as the true number of
    candidates tried grows, so under-counting trials fraudulently inflates the DSR. Thin
    wrapper over ``hermes.learning.stats.deflated_sharpe_ratio`` — named to make the
    governance call-site explicit.
    """
    return deflated_sharpe_ratio(returns, sr_variance, n_trials)


# --------------------------------------------------------------------------- #
# per-book TrialRegistry — the TRUE n_trials, separated by book
# --------------------------------------------------------------------------- #
# Documented point-in-time trial counts (lower bounds) per book, sourced from the lab's own
# sweeps. These SEED the per-book registry; streams bump it as they merge. The shared global
# TrialRegistry (cpcv.py) is left untouched (a single lump counter); this is the per-book
# view the FX deflation needs.
DEFAULT_TRIAL_SEED: dict[Book, list[tuple[str, int]]] = {
    Book.CRYPTO: [
        ("funding_discovery_walkforward_sweep", 648),       # 6 feats × dir × horizon × 3 assets
        ("options_surface_grid_VRP_TERM_VOV_RSKEW", 1024),  # E[max Sharpe|1024] = 0.195 deflation bar
        ("dvol_reversion_config_grid", 96),                 # 7/96 configs beat the null
        ("stablecoin_growth_config_grid", 96),
        ("positioning_flow_hunts_H1_H5", 5),
        # S2 v-next (market-neutral cross-sectional sleeves): 4 pre-registered (2A/2B/2D/2.3) +
        # 18 exploratory = 22. [VÉRIFIÉ S2/TRIAL_REGISTRY.md]. NB: 1869→1891 is THIS module's
        # arithmetic; S2 itself leaves the sum uncomputed ("1869 + 4 + 18").
        ("S2_xsec_neutral_sleeves_4prereg_plus_18exploratory", 22),
    ],
    Book.FX_METALS: [
        ("B_frontier_hunts_B1_B2_B3_B5", 24),  # the n_trials used in _book_report.json
        ("E2_wti_termstruct_configs", 18),     # MC-null failed all 18
        ("E3_fomc_tone_nlp_design_budget", 7), # lexicon×section×level×horizon×decay×threshold (count ALL)
        ("D1_ml_feature_zoo_configs", 4),      # all8/no_wti × gbm/logit, NO_EDGE
    ],
    Book.CROSS: [
        ("X1_dvol_safe_haven", 1),
        ("X2_stablecoin_risk_fx", 1),
        ("X3_conditioning", 1),
        ("crypto_fx_leadlag_both_ways", 2),
    ],
}


class BookTrialRegistry:
    """Persistent per-book count of backtested configurations (the DSR's true n_trials).

    State lives in ``<data_dir>/trials_by_book.json`` as
    ``{"counts": {book: int}, "provenance": {book: [{"source", "n"}]}, "seeded": bool}``.
    Kept SEPARATE from the legacy global ``trials.json`` (cpcv.TrialRegistry) so the
    existing contract is untouched while the FX book can deflate by its own count.
    """

    _FILENAME = "trials_by_book.json"

    def __init__(self, data_dir: Path | str | None = None) -> None:
        # v-next: the no-arg default is now the COMMITTED, shared governance dir (not the
        # per-worktree settings.data_dir), so the deflation count is one authoritative truth.
        if data_dir is None:
            data_dir = governance_registry_dir()
        self._dir = Path(data_dir)
        self._path = self._dir / self._FILENAME

    @property
    def path(self) -> Path:
        """The JSON file backing this registry (``<dir>/trials_by_book.json``)."""
        return self._path

    # -- persistence ---------------------------------------------------------------- #
    def _load(self) -> dict:
        if not self._path.exists():
            return {"counts": {}, "provenance": {}, "seeded": False}
        try:
            payload = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"counts": {}, "provenance": {}, "seeded": False}
        payload.setdefault("counts", {})
        payload.setdefault("provenance", {})
        payload.setdefault("seeded", False)
        return payload

    def _save(self, payload: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    # -- API ------------------------------------------------------------------------ #
    def count(self, book: Book) -> int:
        return int(self._load()["counts"].get(book.value, 0))

    def provenance(self, book: Book) -> list[dict]:
        return list(self._load()["provenance"].get(book.value, []))

    def bump(self, book: Book, n: int = 1, source: str = "") -> int:
        payload = self._load()
        key = book.value
        payload["counts"][key] = int(payload["counts"].get(key, 0)) + int(n)
        payload["provenance"].setdefault(key, []).append({"source": source, "n": int(n)})
        self._save(payload)
        return payload["counts"][key]

    def seed_defaults(self) -> None:
        """Idempotently seed the documented per-book trial counts. Re-running is a no-op
        (guards against double-counting across instances/sessions)."""
        payload = self._load()
        if payload.get("seeded"):
            return
        for book, entries in DEFAULT_TRIAL_SEED.items():
            key = book.value
            for source, n in entries:
                payload["counts"][key] = int(payload["counts"].get(key, 0)) + int(n)
                payload["provenance"].setdefault(key, []).append(
                    {"source": source, "n": int(n)}
                )
        payload["seeded"] = True
        self._save(payload)

    def as_dict(self) -> dict:
        return self._load()


# --------------------------------------------------------------------------- #
# 10.2 — dead-ends ledger
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeadEnd:
    name: str
    book: str
    date: str
    what_tested: str
    why_dead: str
    kill_metric: str
    revive_condition: str
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "book": self.book,
            "date": self.date,
            "what_tested": self.what_tested,
            "why_dead": self.why_dead,
            "kill_metric": self.kill_metric,
            "revive_condition": self.revive_condition,
            "tags": list(self.tags),
        }


# The canonical falsified-or-blocked families. ``tags`` are the lowercase substrings a
# proposal is matched against. Sourced from research_fx/LAB_JOURNAL.md, the stream mandates,
# and the result JSONs (dates verbatim where the source dated them).
DEFAULT_DEAD_ENDS: list[DeadEnd] = [
    DeadEnd(
        "Gold trend-following (TSMOM on gold)", "fx_metals", "inherited / re-confirmed 2026-06-24",
        "Time-series momentum / trend-following on gold, 26y daily.",
        "Secular-bull-beta in a TSMOM costume — the MC random-entry null BEATS it; 26y daily "
        "net-NEGATIVE; loses the decisive 2011-15 bear. A random long-biased entry already earns "
        "gold's drift, so trend rules add nothing.",
        "MC random-entry null beats it; 26y net-NEGATIVE; loses 2011-15 bear",
        "A non-price trend confirmation on an independent channel that de-betas gold's secular "
        "drift, or a trend test on an instrument without a 25y secular uptrend.",
        tags=("gold trend", "trend following gold", "tsmom gold", "gold tsmom", "gold momentum"),
    ),
    DeadEnd(
        "B1 COT positioning-extreme contrarian", "fx_metals", "2026-06-24 (Loop 3)",
        "156wk-percentile COT noncomm net positioning, fade extremes, 6 instruments, +1wk lag.",
        "Generalizes to NONE: only silver clears the MC-null but fails OOS-Sharpe>0.3; the short "
        "leg fights the secular bull; pooled is noise.",
        "best MC-p 0.011 (silver only); 0/6 generalize; pooled Sharpe 0.014",
        "A disaggregated/managed-money positioning panel with a true point-in-time vintage "
        "archive, or an intraday positioning proxy.",
        tags=("cot positioning", "cot contrarian", "cot extreme", "commitment of traders", "cot net"),
    ),
    DeadEnd(
        "B2 real-rate (DFII10) shock → gold", "fx_metals", "2026-06-24 (Loop 3)",
        "10y TIPS real-yield 5d z-shock → directional gold, +1bd lag, over the 2011-15 bear.",
        "Secular-bull-beta: raw +31.9bps but sharply NEGATIVE through the 2011-15 bear; AUD "
        "inverts. Vintage audit confirms data honesty → refutation is purely economic.",
        "XAU MC-null p=0.059 (FAIL); 2011-15 bear -39.5/-127.9bps XAU/XAG",
        "A regime-conditioned real-rate state that genuinely de-betas the secular gold drift AND "
        "beats both base and complement at shuffle-null p<0.05 (Stream 6 gold_regime_state).",
        tags=("dfii10", "real-rate", "real rate", "tips yield", "realrate"),
    ),
    DeadEnd(
        "B3 GVZ gold implied-vol reversion", "fx_metals", "2026-06-24 (Loop 3)",
        "Gold implied-vol (CBOE GVZ) 252d z-score, fade ±2σ (the crypto-DVOL analog on gold).",
        "Tail-driven long-gold drift, not vol-timing; silver negative → fails cross-asset; both "
        "metals negative in the 2011-15 bear.",
        "XAU MC-null p=0.144 (FAIL); winsorized collapses to +46bps; silver negative",
        "An OTC option-skew / 25Δ risk-reversal series (paywalled) rather than ATM GVZ, or a "
        "non-directional vol-sizing target.",
        tags=("gvz", "gold implied vol", "gold vrp", "gold vol reversion"),
    ),
    DeadEnd(
        "B5 GLD ETF-flow divergence (directional)", "fx_metals", "2026-06-24 (Loop 3)",
        "SPDR GLD ounces-in-trust 60d flow z-score / flow-vs-price divergence (stablecoin analog).",
        "Gold ETF flow LAGS price; no defensive value in the bear; silver negative, pooled "
        "negative. Worst sleeve in the book (LOO-Δ-Sharpe -0.014).",
        "XAU MC-null p=0.203 (FAIL); flow lags price; LOO-Δ-Sharpe -0.014",
        "A genuinely LEADING physical/flow series (COMEX registered/eligible inventory, or "
        "creation/redemption order flow that precedes price).",
        tags=("gld flow", "gld-flow", "etf flow", "gold etf flow", "ounces in trust"),
    ),
    DeadEnd(
        "B4 ForexFactory economic-surprise index", "fx_metals", "2026-06-24 (Loop 2/4)",
        "Self-built (actual − forecast) macro-surprise index from a free ForexFactory mirror.",
        "DATA-LOCKED, not falsified on merit: free mirror is current-week-only with no `actual` "
        "field; site Cloudflare-403; ALFRED has no consensus forecast.",
        "0 years of usable (actual,forecast) history",
        "A free/affordable multi-year (actual, forecast, release-timestamp) calendar panel, or a "
        "forward-accruing cron (weekly snapshot + post-release re-pull).",
        tags=("forexfactory", "forex factory", "economic surprise", "macro surprise", "actual vs forecast"),
    ),
    DeadEnd(
        "E1 COMEX physical inventory", "structural", "2026-06-24 (Loop 4)",
        "COMEX warehouse registered/eligible gold ounces as a physical-tightness signal.",
        "DATA-BLOCKED: no free scriptable multi-year COMEX feed; CME ToU bans automation; all "
        "mirrors 403/paywalled. PIT-ideal but moot.",
        "0 years of free automatable inventory history",
        "A licensed or genuinely free, automatable, multi-year COMEX/LBMA physical-inventory "
        "history (the highest-value untested physical axis if the data wall falls).",
        tags=("comex inventory", "comex warehouse", "registered eligible", "physical inventory"),
    ),
    DeadEnd(
        "E2 WTI term-structure roll-yield → CAD/AUD", "fx_metals", "2026-06-24 (Loop 5)",
        "Crude term-structure roll-yield (EIA WTI C1-C4) as a commodity-currency signal, 18 configs.",
        "Both legs net-negative; generalizes only in its failure; the MC null is not cleared on "
        "ANY of the 18 configs (roll-yield sign is inverted).",
        "MC-null fails all 18 configs, min p=0.169",
        "A distinct energy/weather/storage axis with its own economic channel (natgas/HDD→NOK), or "
        "a free multi-maturity metals curve for an actual carry test.",
        tags=("wti term", "wti roll", "crude roll", "term structure roll", "roll yield"),
    ),
    DeadEnd(
        "E3 FOMC tone-change at the print", "fx_metals", "2026-06-24 (Loop 5/6)",
        "Hawk/dove tone surprise from FOMC statements (frozen lexicon), next-session entry.",
        "Fully repriced at the 2pm print; nothing tradeable next session. Pooled MC-null p=0.707; "
        "the synchronized-event basket fix made it harder and it still failed.",
        "pooled MC-null p=0.707; basket null n=58 p=0.515; XAU p=0.440",
        "A markedly higher-frequency/intraday FOMC reaction axis, or the SLOW inter-meeting "
        "hawk/dove DRIFT of speeches entered OUTSIDE the blackout (Stream 7.1 — a different axis).",
        tags=("fomc tone", "fomc statement", "fomc print", "tone surprise at print", "hawk dove print"),
    ),
    DeadEnd(
        "X1 crypto-fear → safe-haven (one-brain crossing)", "cross", "2026-06-24 (Loop 6)",
        "Does crypto DVOL fear lead a safe-haven bid in gold/JPY/CHF? +3d lag, causal alignment.",
        "Crypto-fear gold drift is secular-bull-beta (+53bps ≈ unconditional +48bps); no "
        "incremental haven alpha. crypto↔FX lead-lag is zero; low ρ is itself beta/regime-dependent.",
        "date-level basket MC-null p=0.174; +53bps vs +48bps unconditional",
        "An intraday crypto→FX lead-lag axis (minutes/hours around crypto vol spikes vs a "
        "concurrently-trading FX venue).",
        tags=("crypto fear safe haven", "dvol safe haven", "crypto haven", "crypto to fx lead"),
    ),
    DeadEnd(
        "X2 stablecoin dry-powder → risk-on FX (one-brain crossing)", "cross", "2026-06-24 (Loop 6)",
        "Does stablecoin supply growth lead risk-on FX (AUD up / JPY down)? +3d lag.",
        "AUD/JPY are one risk-on/off factor, not two confirmations (colinear-channel rule); basket "
        "null not cleared. The leading-dry-powder mechanism is crypto-specific.",
        "date-level basket MC-null p=0.2375; AUD/JPY ≈ 1 effective bet",
        "A stablecoin/flow signal mapped onto genuinely DISTINCT FX risk channels, or an intraday "
        "alignment where dry-powder actually leads an FX session open.",
        tags=("stablecoin risk fx", "stablecoin dry powder fx", "dry powder risk on", "stablecoin to fx"),
    ),
    DeadEnd(
        "X3 conditioning / interactions (signal × regime)", "fx_metals", "2026-06-24 (Loop 6)",
        "Conditioning a base signal on a regime (real-rate→gold gated on COT-in-band; GVZ on high-vol).",
        "A non-event: conditioned ≈ base (+34.9 vs +32.0bps); the COMPLEMENT scores higher → the "
        "gate selects nothing. Conditioning cannot manufacture edge from a zero-gross/bull-beta base.",
        "shuffle-label p=0.42; complement > in-band set",
        "A regime variable with a true independent channel applied to a base that ALREADY has "
        "standalone edge (conditioning arbitrates surviving alpha; it cannot create it).",
        tags=("conditioning", "signal x regime", "interaction regime gate", "regime conditioning gold"),
    ),
    DeadEnd(
        "D1 ML feature-zoo combination (GBM/LogReg over 8 features)", "fx_metals", "2026-06-24 (Loop 6)",
        "Best leakage-free GBM/LogReg over all 8 orthogonal causal features, 6467 XAU daily bars, "
        "purged/combinatorial CV.",
        "NO_EDGE: OOS accuracy ≤ the long base rate; +15bps net is just cost-of-being-long "
        "(bull-beta); the combined MC-null is not beaten at all; PBO=0 irrelevant (no skill).",
        "OOS acc 0.518-0.543 ≤ 0.556 base; combined MC-null p=1.000; PBO=0",
        "A genuinely NEW orthogonal feature carrying directional info absent from the 8-feature zoo "
        "(intraday CME order-flow, OTC skew, COMEX inventory, a real (actual,forecast) panel). "
        "Re-running ML over the same exhausted features is forbidden.",
        tags=("feature zoo", "ml combination", "feature-zoo", "gbm logit gold", "ml feature zoo",
              "d1 feature zoo"),
    ),
    DeadEnd(
        "Combined FX book-of-4 under MC-null-first hierarchy", "fx_metals", "2026-06-24 (Loop 3)",
        "The §7 BOOK thesis on the 4 B-frontier sleeves (CPCV/PBO/DSR + LOO-Δ-Sharpe).",
        "A diversification MIRAGE: none of the 4 clears the HARD MC-null prerequisite; admitting "
        "them on positive in-sample LOO inflates PBO and dies OOS. NEVER admit an MC-null-failer on "
        "LOO grounds.",
        "book DSR 0.287, PBO 0.782, Sharpe 0.0177, n_trials 24; enforced book = empty",
        "≥2 sleeves that each independently clear the MC-null (p<0.05) on ≥2 distinct-channel "
        "instruments; only then does LOO-Δ-Sharpe arbitrate.",
        tags=("fx book of 4", "book of four", "diversification mirage", "loo on noise"),
    ),
    DeadEnd(
        "Fast/any-cadence single-asset OHLCV families", "crypto", "inherited (~11 crypto) / re-confirmed FX",
        "Fast vol-scaled momentum, Donchian/vol-expansion breakout, short-horizon continuation, "
        "intraday mean-reversion — single-asset OHLCV.",
        "Die net of honest cost = whipsaw. (Distinct from the LIVE slow-selective ts_mom_vol_scaled.)",
        "net of honest per-asset cost = whipsaw (negative)",
        "Not the cadence — a genuinely new NON-OHLCV exogenous data axis.",
        tags=("vol-scaled momentum", "breakout vol expansion", "donchian", "intraday mean reversion",
              "short-horizon continuation", "fast ohlcv", "ohlcv momentum"),
    ),
    DeadEnd(
        "Cointegration / pairs (incl. gold-silver)", "fx_metals", "inherited / re-confirmed 2026-06-24",
        "Cointegration / statistical-arbitrage pairs on crypto AND FX/metals.",
        "The killer is the half-life (structural), not cost: spread half-lives are 20-99 DAYS "
        "(gold-silver ~10d). No fast reversion to harvest in an honest holding period.",
        "spread half-life 20-99 days (gold-silver ~10d)",
        "A pair whose spread has a genuinely short (days, not weeks) mean-reversion half-life — a "
        "different driver than the slow-cointegrating macro pairs.",
        tags=("cointegration", "stat arb pairs", "pairs trade", "gold silver pair", "eth btc pair",
              "pairs_eth_btc"),
    ),
    DeadEnd(
        "Session / time-of-day on FX", "fx_metals", "inherited / re-confirmed 2026-06-24",
        "Session and time-of-day directional effects on FX (London/NY open, hour-of-day).",
        "Gross pre-cost return is zero-mean — no edge even before cost.",
        "gross pre-cost = zero-mean",
        "A session effect tied to a specific exogenous event/flow (a fix/auction with measurable "
        "order imbalance), not a calendar-clock anomaly.",
        tags=("session time of day", "time-of-day", "intraday tod", "hour of day fx", "london open",
              "intraday_tod"),
    ),
    DeadEnd(
        "Cross-sectional momentum & temporal lead-lag", "cross", "inherited / re-confirmed 2026-06-24",
        "Cross-sectional momentum and lead-lag (BTC→alts, DXY→majors, XAU→AUD).",
        "Single-asset artifacts or thin tail/news-driven; do not generalize.",
        "single-asset artifacts; thin tail/news-driven",
        "A lead-lag with a verified causal microstructure mechanism (order-flow leading, not "
        "returns-on-returns) across distinct channels.",
        tags=("cross-sectional momentum", "lead-lag", "lead lag", "btc to alts", "dxy to majors",
              "xau to aud"),
    ),
    DeadEnd(
        "Trade-flow microstructure (OFI/CVD from SPOT taker volume)", "crypto", "inherited",
        "Order-flow imbalance / CVD from taker-buy SPOT volume.",
        "Redundant with price momentum (ρ 0.77-0.96) or funding-collinear; the only orthogonal "
        "config dies on honest 2×RT cost. (CME FUTURES tape is a DIFFERENT untested axis.)",
        "ρ 0.77-0.96 vs momentum; orthogonal config dies on 2×RT cost",
        "Real intraday CME FUTURES tape / order-flow / CVD (paywalled) — a different axis from the "
        "dead SPOT taker-buy volume.",
        tags=("ofi", "cvd", "trade-flow microstructure", "taker buy volume", "order flow imbalance spot"),
    ),
    DeadEnd(
        "Maker-execution free lunch", "structural", "inherited / re-confirmed 2026-06-24",
        "Passive maker-order execution to capture spread/rebate as a free-lunch edge.",
        "MIRAGE — adverse selection: ~99% fill but you fill the losers and miss the winners; "
        "realized PnL negative.",
        "~99% fill, adverse selection (miss +59..+204bps winners)",
        "A venue/microstructure where queue position is informative without adverse selection (a "
        "verified rebate on uninformed flow).",
        tags=("maker execution", "maker free lunch", "maker rebate", "passive maker fill"),
    ),
    DeadEnd(
        "H2 funding term-structure (literal term-spread)", "crypto", "from STREAM_02",
        "Simple perp funding term-spread as a directional signal.",
        "Gross edge trivial and sign-inverted once cost applied; collapses into 'high funding "
        "continues' (already S2) or 'buy dips' (bull-beta).",
        "0.7 bps gross vs ~30 bps cost, sign inverted",
        "Cross-venue FUNDING DISPERSION + OI-acceleration at constant funding (multi-venue "
        "crowding) — NOT the simple term-spread.",
        tags=("funding term structure", "funding term-spread", "funding term spread"),
    ),
    DeadEnd(
        "H5 liquidation-cascade reversion (OHLCV proxy)", "crypto", "from STREAM_02",
        "Mean-reverting rebound after a forced-liquidation flush, using a Binance OHLCV proxy.",
        "Proxy was pure bull-beta; the edge lived only in the bull leg.",
        "joint-null p=0.907 (bull-beta) on the OHLCV proxy",
        "Real coinglass_liquidation history (≥2019, size/cluster) with STRICT bull/bear separation "
        "(the authorised re-test). If it again survives only in the bull leg → dead again.",
        tags=("liquidation cascade", "liquidation flush", "forced liquidation reversion", "liquidation rebound"),
    ),
    DeadEnd(
        "Options-surface grid (VRP/TERM/VOV/RSKEW, 1024 configs)", "crypto", "from STREAM_04",
        "1024-config grid over crypto-options features (variance-risk-premium, term, vol-of-vol, "
        "realized-skew).",
        "Grid-max did not beat the deflation null; classic overfit-grid death. RSKEW-fade is the "
        "SAME fear-reversion as the killed DVOL (~34% overlap).",
        "grid-max Sharpe 0.1934 < E[max|1024] 0.1950",
        "True 25Δ skew / Greeks history (Amberdata, or the forward-only deribit surface_snapshot once "
        "multi-year). DVOL stays a SIZING gate, not a directional signal.",
        tags=("options surface grid", "vrp grid", "rskew", "vol of vol", "realized skew", "1024 config"),
    ),
    DeadEnd(
        "Legacy dead crypto slate (breakout/pairs/intraday-tod)", "crypto", "from STREAM_01",
        "Legacy crypto legs in combination.py: breakout_vol_expansion, pairs_eth_btc, intraday_tod_btc.",
        "Dead ('morts'); to be PURGED from combination.py; the harness must NOT re-bless them. "
        "(ts_mom_vol_scaled is LIVE Strategy #1 — NOT in this dead list.)",
        "flagged dead; book rebuilt from confirmed/orthogonal sleeves only",
        "Not revivable as-is; would require a genuinely new orthogonal data axis via a fresh sleeve.",
        tags=("breakout_vol_expansion", "pairs_eth_btc", "intraday_tod_btc", "dead slate", "legacy slate"),
    ),
    DeadEnd(
        "Pre-vetted-dead frontiers (pinning, skew, straddle, WMR fix, swap-free carry, GP search)",
        "structural", "inherited (do-not-start list)",
        "NY 10am option strike-pinning; GLD/SLV ETF option-skew; release-minute straddle; month-end "
        "WMR 16:00 fix; swap-free carry spot-drift; genetic/symbolic-regression search.",
        "Each pre-vetted dead: no free PIT strike/OI history; thin/paywalled skew; ~50-pip spread "
        "eats the straddle; WMR fix crowded/decayed; thin swap-free carry; GP = overfitting machine.",
        "no free PIT data / crowded / GP overfit machine",
        "Free PIT option strike/OI history; a non-crowded fix window; GP only after the deflation "
        "budget + book harness are trusted (see gp_lock_status).",
        tags=("strike pinning", "10am pinning", "gld skew", "slv skew", "release-minute straddle",
              "wmr fix", "swap-free carry", "symbolic regression", "genetic programming"),
    ),
    # ---- dated negatives from the merged streams (S2-S8 recorded outcomes) ---------- #
    DeadEnd(
        "S2.1 top-LSR contrarian positioning", "crypto", "2026 (Stream 2 outcome)",
        "Top-trader long-short-ratio (CoinGlass) contrarian directional sleeve on BTC/ETH/SOL.",
        "REFUTED: the −0.27 ρ(top_lsr, fwd) was a ~30-day MIRAGE; on full history ρ≈0; the extreme "
        "form is directional beta whose sign flips bull↔bear; no per-instrument generalization.",
        "full-history ρ≈0; sign flips bull↔bear; no generalization",
        "A positioning axis with a stable forward sign across regimes and ≥2 distinct-channel "
        "instruments (the slow daily top-LSR contrarian axis is exhausted). NEVER re-run continuous top-LSR.",
        tags=("top-lsr", "top lsr", "long-short ratio contrarian", "lsr contrarian", "positioning contrarian"),
    ),
    DeadEnd(
        "S2.2 liquidation-cascade reversion (real CoinGlass data)", "crypto", "2026 (Stream 2 outcome)",
        "Mean-reverting rebound after real CoinGlass liquidation flushes (≥2019), bull/bear separated.",
        "An audit caught a LOOK-AHEAD in the driver glue (trade side taken un-lagged at the entry bar → "
        "leaks the day's move); corrected via lagged side array → NO edge at any lag, DSR≈0.",
        "after look-ahead fix: DSR≈0 at every lag",
        "A genuinely leading liquidation microstructure (cluster/size at sub-daily cadence on a venue "
        "with no look-ahead in the side); the daily real-liquidation reversion axis is now also dead.",
        tags=("liquidation real data", "coinglass liquidation reversion", "liquidation cascade real"),
    ),
    DeadEnd(
        "S3 funding-squeeze on 1d / universe expansion", "crypto", "2026 (Stream 3 outcome)",
        "Generalizing the validated 4h funding_squeeze edge onto 1d bars and a wider instrument universe.",
        "Does NOT generalize off 4h: 0/7 instruments pass the MC-null on 1d; AVAX is the lone Bonferroni "
        "survivor but mono-instrument → rejected. The validated edge is 4h, absent from the lake.",
        "0/7 pass MC-null on 1d; AVAX mono-instrument; mean_reversion_regime LOO −0.14",
        "A lake with 4h history for the wider universe (the validated cadence), or a new instrument that "
        "passes on its own distinct channel. Never lower z to force a pass.",
        tags=("funding_squeeze on 1d", "funding squeeze 1d", "universe expansion funding", "1d universe"),
    ),
    DeadEnd(
        "S4.3 VRP-DVOL sizing gate", "crypto", "2026 (Stream 4 outcome)",
        "Using the DVOL variance-risk-premium as a SIZING multiplier on the live funding_squeeze sleeve.",
        "Does NOT lift the held-out Sharpe: Δ+0.045 ≈ the deflation bar, shuffle-p 0.14-0.18 (fails the "
        "floor p<0.05), worse on the full sample, only 13 OOS trades → abandoned.",
        "Δ+0.045 ≈ deflation bar; shuffle-p 0.14-0.18; 13 OOS trades",
        "A non-directional vol/regime target with enough OOS trades and a shuffle-p<0.05 uplift. The DVOL "
        "VRP stays a research sizing gate, not a deployed multiplier.",
        tags=("vrp-dvol sizing", "vrp dvol sizing", "dvol sizing gate", "vrp sizing"),
    ),
    DeadEnd(
        "S5 gold-β-hedged metals relative-value book", "fx_metals", "2026 (Stream 5 outcome)",
        "Gold-beta-hedged metals RV (gold/silver/miners) combined into a neutral book; gold-neutral spreads.",
        "EMPTY book: gold-neutral spreads either re-import the OTHER metal's bull-beta or are "
        "unforecastable (gold+silver span ~2D). §3.7 overfitting wall binds, not cost. Adversarially airtight.",
        "empty book; spreads re-import bull-beta or unforecastable (2D span)",
        "A genuinely third metals factor orthogonal to the gold/silver plane, or an FX IV panel for the "
        "data-locked 5.2 VRP/skew leg. The gold-β-hedged RV axis is exhausted.",
        tags=("gold-beta-hedged", "metals relative value", "metals rv", "gold-silver-miners rv",
              "gold neutral spread"),
    ),
    DeadEnd(
        "S7 central-bank tone-drift (LLM hawk/dove)", "cross", "2026 (Stream 7 outcome)",
        "LLM-scored hawk↔dove drift of inter-meeting Fed/ECB/BoE speeches → out-of-blackout basket gate.",
        "Only the Fed price-basket lights up (MC-p 0.004) but FAILS: DSR 0.483<0.5; one-sided (dovish leg "
        "+78bps = gold/USD easing-regime BETA); rates channel dead (MC-p 0.144 → no 2nd distinct channel); "
        "the frozen E3 lexicon does NOT replicate (MC-p 0.085) → the LLM uplift is unconfirmable look-ahead.",
        "DSR 0.483<0.5; dovish leg = beta; rates MC-p 0.144; lexicon MC-p 0.085",
        "A tone signal that (a) replicates under a frozen pre-sample lexicon (no LLM look-ahead), (b) is "
        "two-sided not easing-regime beta, and (c) carries a 2nd distinct channel (rates, not just price).",
        tags=("cb tone", "central-bank tone", "hawk dove drift", "tone drift", "llm tone", "fed tone drift"),
    ),
    DeadEnd(
        "S8 regime conditioning of funding_squeeze (HMM/BOCPD)", "crypto", "2026 (Stream 8 outcome)",
        "Conditioning/sizing the live funding_squeeze sleeve on a causal HMM/BOCPD/vol-bucket regime state.",
        "No gate beats the shuffle-null: best HMM-trend conditioning shuffle-p≈0.11 (right sign but "
        "underpowered); the regime state does not provably beat trivial efficiency-ratio. Regime-aware CPCV "
        "= KILL (split perm-p 0.135 = noise → keep standard CPCV). The state ships as an S8 input only.",
        "best conditioning shuffle-p≈0.11; regime-aware CPCV perm-p 0.135 (noise)",
        "A base sleeve with standalone edge to condition (conditioning arbitrates surviving alpha, it cannot "
        "create it) and a regime gate that beats the shuffle-null p<0.05 on enough trades.",
        tags=("regime conditioning funding", "hmm conditioning", "bocpd conditioning", "regime gate funding",
              "regime-aware cpcv"),
    ),
    # ---- S2 v-next: market-neutral cross-sectional sleeve family (all NEGATIVE) ------- #
    # [VÉRIFIÉ vs ../hermes-s2-xsec/research_streams/S2/FINDINGS.md + results_s2_xsec.json]
    DeadEnd(
        "S2.2A funding cross-sectional continuation (XS)", "crypto", "2026 (Stream 2 v-next)",
        "Daily cross-sectional funding-z reshuffle, LONG high-funding / SHORT low-funding (continuation), "
        "tercile two-leg market-neutral.",
        "WRONG-WAY SIGN — cross-sectional funding MEAN-REVERTS, it does not continue (gross-only ann Sharpe "
        "-1.38). Even with the sign flipped it is COST-KILLED: the two-leg honest cost is -16.85 bps/day "
        "because a daily funding-z reshuffle turns over ~100%/day. gross(price) -8.7 · carry -1.45 · cost "
        "-16.85 bps. MC random-sign p=1.0 (sign wrong-way).",
        "gross_ann_sharpe -1.38 (wrong-way); cost -16.85 bps/day; ~100%/day rotation; MC-sign p=1.0",
        "A funding signal whose forward sign is STABLE (not mean-reverting) and that survives the ~100%/day "
        "two-leg cost — e.g. a slow, low-turnover funding state, NOT a daily XS reshuffle.",
        tags=("cross-sectional funding", "funding cross-sectional", "funding-xs", "xs funding",
              "funding cross sectional", "cross sectional funding"),
    ),
    DeadEnd(
        "S2.2D volatility-dispersion cross-sectional (XS)", "crypto", "2026 (Stream 2 v-next)",
        "Cross-sectional realized-vol dispersion sleeve (long low-vol / short high-vol style), market-neutral.",
        "NO EDGE: a faint positive whiff in the price spread (gross ann Sharpe +0.20) but it is NOT "
        "significant (MC random-sign p=0.311) and is eaten by cost (net -1.84 bps). Clean no-edge.",
        "gross ann Sharpe +0.20 (NS, MC-sign p=0.311); net -1.84 bps (cost-eaten)",
        "A vol-dispersion signal with a significant gross edge (MC-sign p<0.05) large enough to survive cost — "
        "the faint +0.20 whiff is not it.",
        tags=("vol dispersion", "volatility dispersion", "vol-dispersion", "dispersion cross-sectional",
              "vol dispersion xs"),
    ),
    DeadEnd(
        "S2.2B open-interest-growth cross-sectional (XS)", "crypto", "2026 (Stream 2 v-next)",
        "Cross-sectional open-interest growth sleeve over the only 7 CoinGlass-OI coins "
        "(ARB, AVAX, BTC, ETH, SOL, SUI, WIF), 1d.",
        "DATA-GATED / UNDER-POWERED, not a clean refutation: only 7 names (often 4-5 available), and the 7 are "
        "current SURVIVORS (survivorship). gross ann Sharpe +0.13 (NS), cost -26 bps/day.",
        "7 coins (survivorship); gross +0.13 NS; cost -26 bps/day; under-powered",
        "A deep OI panel over a wider, point-in-time (non-survivorship) universe with enough names for a "
        "powered cross-sectional test.",
        tags=("oi growth", "oi-growth", "open-interest growth", "open interest growth", "oi growth xs"),
    ),
    DeadEnd(
        "S2.3 cross-venue funding dispersion", "crypto", "2026 (Stream 2 v-next)",
        "Cross-venue funding dispersion sleeve, 5 coins × 4 venues (Binance/OKX/Bybit/Bitget), 1d.",
        "NO EDGE: ann Sharpe -0.44 (negative → wrong-sign), MC random-sign p=0.05, thin 5-coin universe. The "
        "multi-venue lake (S0) UNBLOCKED the data but the dispersion itself carries no tradeable edge.",
        "ann Sharpe -0.44 (wrong-sign); MC-sign p=0.05; thin 5-coin universe",
        "A cross-venue crowding signal with a stable forward sign on a wider venue×coin panel — the 5-coin "
        "dispersion is exhausted (data now exists; the edge does not).",
        tags=("cross-venue funding", "cross-venue dispersion", "cross venue dispersion", "cross venue funding",
              "cross-venue funding dispersion"),
    ),
]


class DeadEndsLedger:
    """Queryable shared memory of falsified/blocked hypotheses (anti re-spend).

    ``check(proposal, new_data_axis=None)`` returns one of:
      * BLOCKED-BY-LEDGER  — the proposal matches a dead end and brings no NEW data axis;
      * ALLOWED-NEW-AXIS   — it matches a dead end BUT a genuinely new data axis is named
        (the caller MUST re-gate the candidate from scratch under the hardened bar);
      * NOT-A-DEAD-END     — no match; proceed (and re-gate as normal).
    """

    def __init__(self, entries: list[DeadEnd] | None = None) -> None:
        self._entries = list(entries) if entries is not None else list(DEFAULT_DEAD_ENDS)

    @classmethod
    def load(cls, path: Path | str | None = None) -> DeadEndsLedger:
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return cls()
        entries = [
            DeadEnd(
                name=e["name"], book=e.get("book", ""), date=e.get("date", ""),
                what_tested=e.get("what_tested", ""), why_dead=e.get("why_dead", ""),
                kill_metric=e.get("kill_metric", ""), revive_condition=e.get("revive_condition", ""),
                tags=tuple(e.get("tags", ())),
            )
            for e in raw
        ]
        return cls(entries)

    def entries(self) -> list[DeadEnd]:
        return list(self._entries)

    def _match(self, proposal: str) -> DeadEnd | None:
        text = proposal.lower()
        for entry in self._entries:
            if any(tag and tag in text for tag in entry.tags):
                return entry
        return None

    def check(self, proposal: str, new_data_axis: str | None = None) -> dict:
        entry = self._match(proposal)
        if entry is None:
            return {
                "verdict": LedgerVerdict.NOT_A_DEAD_END,
                "matched": None,
                "reason": "No falsified hypothesis matches this proposal — proceed and gate normally.",
                "why_dead": "",
                "revive_condition": "",
            }
        has_new_axis = bool(new_data_axis and new_data_axis.strip())
        if not has_new_axis:
            return {
                "verdict": LedgerVerdict.BLOCKED,
                "matched": entry.name,
                "reason": (
                    f"BLOCKED-BY-LEDGER: '{entry.name}' was falsified/blocked ({entry.date}) and no "
                    f"NEW data axis was supplied. {entry.why_dead}"
                ),
                "why_dead": entry.why_dead,
                "kill_metric": entry.kill_metric,
                "revive_condition": entry.revive_condition,
            }
        return {
            "verdict": LedgerVerdict.ALLOWED_NEW_AXIS,
            "matched": entry.name,
            "reason": (
                f"ALLOWED with a new data axis ('{new_data_axis}'): you MUST re-gate the candidate "
                f"from scratch under the hardened bar. Required revival condition: {entry.revive_condition}"
            ),
            "why_dead": entry.why_dead,
            "revive_condition": entry.revive_condition,
        }

    def to_json(self) -> str:
        return json.dumps([e.as_dict() for e in self._entries], indent=2)


# --------------------------------------------------------------------------- #
# 10.2 — acquisition map (data-$ governance)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataSource:
    source: str
    status: DataStatus
    cost: str
    unlocks: str
    streams_affected: str
    notes: str

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "status": self.status.value,
            "cost": self.cost,
            "unlocks": self.unlocks,
            "streams_affected": self.streams_affected,
            "notes": self.notes,
        }


DEFAULT_ACQUISITION_MAP: list[DataSource] = [
    DataSource(
        "CoinGlass", DataStatus.ACQUIRED, "paid plan / COINGLASS_API_KEY (~30 req/min)",
        "deep crypto positioning history (top/global LSR, taker volume, OI OHLC, funding, "
        "liquidations; top-LSR BTC back to 2020-05-16). DEEP MULTI-VENUE funding/OI ingested: "
        "Binance 2020-03-30 · OKX/Bybit 2020-10-30 · Bitget 2021-08-12 (OI deeper still: "
        "OKX 2020-04-17, Bybit 2020-05-08) — coinglass_multi_{funding_rate,open_interest}",
        "Stream 0 lake; Stream 2/3 (positioning) FULLY UNBLOCKED; Stream 2.3/9 cross-venue UNBLOCKED",
        "Frozen into the point-in-time lake. CORRECTION (S0 re-freeze): the lake is NO LONGER "
        "'Binance-only' — multi-venue (OKX/Bybit→2020, Bitget→2021) funding/OI is DEEP & INGESTED, "
        "superseding the prior data-gate cited by S2.3/S9. [VÉRIFIÉ S0/COVERAGE.md].",
    ),
    DataSource(
        "Amberdata", DataStatus.NOT_ACQUIRED, "paid alt-data vendor (not quantified)",
        "true 25Δ option skew / GEX / Greeks history (backfilled, since ~2021-05) for crypto options",
        "Stream 4 (options surface 2.0)",
        "ARBITRATE. Standing recommendation: NO. Do not buy until Stream 4 demonstrates a "
        "backtestable forward edge from the forward-only deribit surface_snapshot that would only be "
        "STRENGTHENED by history — otherwise it funds a likely re-wrap of the dead RSKEW/options grid.",
    ),
    DataSource(
        "FRED (St. Louis Fed)", DataStatus.FREE, "$0 (keyless graph CSV; key at ~/.fred_api_key for ALFRED)",
        "DFII10, DGS10, T10Y2Y, BAMLH0A0HYM2 (+vintage), DTWEXBGS, GVZCLS",
        "FX/metals research; Stream 0 lake",
        "Keyless == vintage for DFII10 specifically (0.000bp revision). B2/B3 ultimately REFUTED.",
    ),
    DataSource(
        "WGC (World Gold Council / Goldhub)", DataStatus.ACQUIRED, "$0 but MANUAL download",
        "authoritative central-bank gold demand (reported+unreported), GDT balance (ETF line), "
        "monthly/annual CB changes (Turkey twins — don't sum naively)",
        "Stream 6 (CB-demand conditioner)",
        "Ingested via manual_gold.run() from manually-downloaded Goldhub files. Closed 2026-06-26.",
    ),
    DataSource(
        "BIS cbspeak", DataStatus.FREE, "$0",
        "BIS central-banker speeches corpus 1996+ (~20k) + FOMC statements (carries BoE speeches)",
        "Stream 0 lake; Stream 7 (CB tone)",
        "Filter by speaker (institution column sparse). FOMC tone-at-print (E3) itself REFUTED.",
    ),
    DataSource(
        "GLD flows (SPDR)", DataStatus.FREE, "$0 (historical-archive xlsx, no key; exchange=NYSE)",
        "GLD Total Ounces in Trust, 21.6y, published-final",
        "FX/metals research (B5); Stream 0 lake",
        "≥3d lag recommended. B5 directional signal REFUTED (flow lags price).",
    ),
    DataSource(
        "IAU flows (iShares)", DataStatus.ACQUIRED, "$0 but MANUAL download",
        "iShares IAU ETF flows 2005→2026",
        "Stream 0 lake; gold research",
        "Closed 2026-06-26 via manual_gold ingestion (durable manual parquet).",
    ),
    DataSource(
        "CFTC COT (Socrata legacy 6dca-aqww)", DataStatus.FREE, "$0 (no key)",
        "40.4y / 12,949 weekly COT positioning (gold/silver/EUR/AUD/JPY/GBP)",
        "FX/metals research (B1)",
        "Stitch on cftc_contract_market_code, not name. +1wk lag. B1 contrarian REFUTED.",
    ),
    DataSource(
        "Deribit (DVOL / surface)", DataStatus.FREE, "$0 (public endpoints)",
        "DVOL hourly BTC/ETH since 2021-03-24; 25Δ risk-reversal surface_snapshot (FORWARD-LOG only)",
        "Stream 0 lake; Stream 4 (DVOL VRP sizing gate)",
        "The 25Δ RR surface is forward-only (NOT backfillable) — the only unexplored options axis; "
        "revisit in ~6-12 months once history accrues. DVOL VRP is a SIZING gate, not direction.",
    ),
    DataSource(
        "EIA API (energy)", DataStatus.FREE, "$0 (free, registration required)",
        "WTI term-structure (C1-C4) live continuation",
        "Physical-commodities research (E2)",
        "Tier-0 hygiene. E2 roll-yield→FX REFUTED (sign inverted).",
    ),
    DataSource(
        "ForexFactory macro calendar", DataStatus.NOT_ACQUIRED, "paid calendar-history vendor",
        "multi-year (actual, forecast) macro-surprise panel",
        "Stream 7 (macro surprise, B4)",
        "SETTLED DEAD / UNBACKTESTABLE on free data. Only a paid vendor or a forward-accruing cron "
        "unlocks it. Standing recommendation: do NOT spend; prefer the forward-accruing cron.",
    ),
    DataSource(
        "Databento (CME curves / futures)", DataStatus.DO_NOT_SPEND, "$125 credit held; ~$25-80/mo paid",
        "CME term-structure / futures curves (roll-yield, carry, inventory premia)",
        "physical-commodities / FX expansion",
        "DO NOT SPEND. Every physical premium was REFUTED on free evidence and WTI ground-truth "
        "showed the roll-yield sign is inverted. Keep the $125 credit unspent.",
    ),
    DataSource(
        "Generic alt-data (satellite / AIS)", DataStatus.DO_NOT_SPEND, "$100s-$1000s/mo",
        "satellite imagery, AIS shipping, premium alt-data",
        "physical-commodities / FX expansion",
        "DEFER/NEVER. Free pre-print edge already contradicts the thesis.",
    ),
]


class AcquisitionMap:
    """Governs data-$ spend. ``can_spend(source)`` returns whether (and why) money should be
    spent; ``status(source)`` returns the raw ``DataStatus``."""

    def __init__(self, sources: list[DataSource] | None = None) -> None:
        self._sources = list(sources) if sources is not None else list(DEFAULT_ACQUISITION_MAP)
        self._by_name = {s.source.lower(): s for s in self._sources}

    @classmethod
    def load(cls, path: Path | str | None = None) -> AcquisitionMap:
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return cls()
        sources = [
            DataSource(
                source=s["source"], status=DataStatus(s["status"]), cost=s.get("cost", ""),
                unlocks=s.get("unlocks", ""), streams_affected=s.get("streams_affected", ""),
                notes=s.get("notes", ""),
            )
            for s in raw
        ]
        return cls(sources)

    def _get(self, source: str) -> DataSource | None:
        q = source.lower().strip()
        if q in self._by_name:  # exact
            return self._by_name[q]
        # canonical names carry parenthetical suffixes ("FRED (St. Louis Fed)") — match the
        # short alias by prefix, then by substring as a last resort.
        for key, src in self._by_name.items():
            if key.startswith(q):
                return src
        for key, src in self._by_name.items():
            if q in key:
                return src
        return None

    def sources(self) -> list[DataSource]:
        return list(self._sources)

    def status(self, source: str) -> DataStatus | None:
        s = self._get(source)
        return s.status if s else None

    def can_spend(self, source: str) -> dict:
        s = self._get(source)
        if s is None:
            return {
                "source": source, "status": "unknown", "allowed": False,
                "reason": "Unknown source — not in the acquisition map; do not spend without a scout verdict.",
            }
        if s.status is DataStatus.ACQUIRED:
            allowed, reason = True, f"Already acquired (paid/ingested) — use freely, no new spend. {s.notes}"
        elif s.status is DataStatus.FREE:
            allowed, reason = True, f"Free ($0) — use freely. {s.notes}"
        elif s.status is DataStatus.DO_NOT_SPEND:
            allowed, reason = False, f"DO NOT SPEND. {s.notes}"
        else:  # NOT_ACQUIRED → arbitrage; default recommendation NO
            allowed, reason = False, f"NOT ACQUIRED — arbitrate, standing recommendation NO. {s.notes}"
        return {"source": s.source, "status": s.status.value, "allowed": allowed, "reason": reason}

    def to_json(self) -> str:
        return json.dumps([s.as_dict() for s in self._sources], indent=2)


# --------------------------------------------------------------------------- #
# v-next — funding_squeeze status DOWNGRADE (convergent evidence)
# --------------------------------------------------------------------------- #
# The lone crypto sleeve that ever looked like a survivor. Two independent lines of evidence now
# converge to DOWNGRADE it from "validated survivor" to WEAKLY-SUPPORTED / not-robustly-confirmed:
#   (1) this module's own deflation re-gates it REJECT at the true book count (1869, now 1891);
#   (2) S2's 4h CLUSTER-AWARE re-test finds it suggestive-but-NS in bull (macro-event t=1.50) and
#       not-established in bear (n 35→21 macro-events, t=1.23), with the sign NOT flipping bull↔bear
#       (so it is not pure bull-beta, but it is not robustly confirmed either).
# The per-trade Sharpe (0.16 all-23 / 0.24 deep) is the real-but-weak band — a leak would read ≫1.
# [VÉRIFIÉ vs ../hermes-s2-xsec/research_streams/S2/FINDINGS.md].
FUNDING_SQUEEZE_STATUS: dict = {
    "sleeve": "funding_squeeze",
    "prior_label": "validated survivor (joint-null p=0.017)",
    "status": "WEAKLY_SUPPORTED",
    "confirmed": False,
    "per_trade_sharpe": {"all23": 0.16, "deep_btc_eth_sol": 0.24},
    "bull": {
        "excess_bps": 217,            # all-23 pool
        "iid_p": 5e-05,               # i.i.d. random-entry null (anti-conservative under clustering)
        "macro_event_t": 1.50,        # macro-event-blocked (cluster-aware) → NOT significant
        "macro_event_significant": False,
        "trade_median_bps": 25,
    },
    "bear": {
        "n_trades": 35,
        "n_macro_events": 21,         # 35 trades collapse to 21 macro-events
        "macro_event_t": 1.23,        # NOT significant
        "macro_event_significant": False,
    },
    "sign_flips_bull_bear": False,
    "evidence": [
        "deflation re-gate REJECTs at the true book count (1869 → 1891) — fails the deflated bar",
        "S2 4h cluster-aware: bull macro-event t=1.50 (NS), median +25bps only; the i.i.d. p=1e-4 is "
        "anti-conservative because squeezes co-fire across alts (cluster-aware null is mandatory)",
        "S2 bear: 35 trades → 21 macro-events, t=1.23 (NS) → NOT established in bear",
        "sign does NOT flip bull↔bear → not pure bull-beta, but NOT robustly confirmed",
        "per-trade Sharpe 0.16 (all-23) / 0.24 (deep) = real-but-weak, NOT a leak (a leak reads ≫1)",
    ],
    "note": (
        "DO NOT treat the live funding_squeeze as a confirmed edge. It is real-but-weak and "
        "unconfirmed; it is NOT a promotion candidate today and requires clean OOS-forward "
        "(Task 4) plus a cluster-aware re-confirmation to ever change status. Stays shadow size 0."
    ),
}


def funding_squeeze_status() -> dict:
    """The downgraded, convergent-evidence verdict on ``funding_squeeze`` (WEAKLY_SUPPORTED)."""
    return dict(FUNDING_SQUEEZE_STATUS)


# --------------------------------------------------------------------------- #
# 10.3 — GP / D1 lock
# --------------------------------------------------------------------------- #
# The D1 feature-zoo ML falsification result — the evidence the GP lock cites.
D1_NO_EDGE: dict = {
    "verdict": "NO_EDGE",
    "oos_accuracy": 0.5176,
    "oos_accuracy_range": [0.518, 0.543],
    "base_rate": 0.5564,
    "oos_net_bps": 15.02,
    "mc_null_p": 1.0,
    "pbo": 0.0,
    "n_features": 8,
    "n_bars": 6467,
    "source": "research_fx/_d1_results.json (2026-06-24, Loop 6)",
    "note": (
        "OOS accuracy ≤ the long base rate; +15bps net = cost-of-being-long (bull-beta, the tide); "
        "the combined MC-null is not beaten at all (p=1.000); PBO=0 is irrelevant (no skill to overfit)."
    ),
}


def gp_lock_status(
    deflation_trusted: bool = False,
    book_harness_trusted: bool = False,
    validated_orthogonal_feature_count: int = 0,
    min_validated_features: int = 3,
    forward_validated_candidate: bool = False,
    candidate_sleeves_today: list[str] | None = None,
) -> dict:
    """The explicit GP / feature-zoo (D1) lock (mandate 10.3), now also gated by OOS-forward.

    Genetic/symbolic-GP feature search is the Renaissance machinery AND an overfitting
    engine — so it runs LAST. It stays LOCKED until ALL of:
      1. the deflation budget (DSR/PBO promotion bar) exists and is TRUSTED;
      2. the Stream-1 combined-book harness is merged and TRUSTED;
      3. a zoo of ≥ ``min_validated_features`` features ALREADY validated orthogonal by
         other streams exists (GP combines validated features, it does not invent them);
      4. (v-next Task 4) at least one sleeve has actually cleared the OOS-FORWARD precondition —
         i.e. the deflation discipline has produced a real forward-validated candidate, not just
         in-sample survivors. TODAY this is UNMET: funding_squeeze is only WEAKLY supported and the
         cross-sectional family is dead, so there is no candidate sleeve at all.

    All conditions default to UNMET, so by default the lock is engaged. The lock cites the D1
    NO_EDGE evidence so no one re-litigates it without a new orthogonal feature.
    """
    conditions = [
        {
            "key": "deflation_budget_trusted",
            "description": "The DSR/PBO per-book promotion bar exists and has been trusted to gate "
            "≥1 merged stream (Stream 10.1).",
            "met": bool(deflation_trusted),
        },
        {
            "key": "book_harness_trusted",
            "description": "The Stream-1 combined-book harness (LOO-Δ-Sharpe) is merged and green.",
            "met": bool(book_harness_trusted),
        },
        {
            "key": "validated_orthogonal_feature_zoo",
            "description": f"≥ {min_validated_features} features already validated orthogonal by other "
            f"streams (currently {validated_orthogonal_feature_count}).",
            "met": validated_orthogonal_feature_count >= min_validated_features,
        },
        {
            "key": "forward_validated_candidate_exists",
            "description": "≥1 sleeve has cleared the OOS-FORWARD precondition (Task 4): clean forward "
            "OOS accrued in real regime with the AM-4 leak audit replayed. None exists today.",
            "met": bool(forward_validated_candidate),
        },
    ]
    locked = not all(c["met"] for c in conditions)
    return {
        "locked": locked,
        "conditions": conditions,
        "current_validated_feature_count": validated_orthogonal_feature_count,
        "candidate_sleeves_today": list(candidate_sleeves_today or []),
        "funding_squeeze": funding_squeeze_status(),
        "evidence": D1_NO_EDGE,
        "note": (
            "D1 remains LOCKED until every condition is met. Re-running ML/GP over the same exhausted "
            "8-feature zoo is forbidden; only a genuinely NEW orthogonal feature can reopen it. "
            "v-next: GP additionally requires a forward-VALIDATED candidate — and there is none today "
            "(funding_squeeze WEAKLY_SUPPORTED, cross-sectional family dead)."
        ),
    }
