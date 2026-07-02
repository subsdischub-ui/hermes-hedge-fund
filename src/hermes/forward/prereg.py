"""Dated pre-registrations for the two sleeves worth accruing forward OOS on.

A pre-registration freezes the EXACT cell and the FULL deflation trial count before
any forward data exists, so the eventual verdict cannot be cherry-picked. Both leads
are honestly labelled: any in-sample number is CONTAMINATED; only the forward OOS,
deflated at the true count, can confirm them.

Provenance note: ``research_streams/S2/FINDINGS.md`` and ``s2_xsec.py`` are NOT on
this branch (branched from main post-S0); the XS cell is taken verbatim from the
round-2 forward mandate, which encodes the S2 v-next empirical finding (the squeeze
direction lost; only the slow, sign-inverted spread carried a +0.58 post-hoc lead).
"""

from __future__ import annotations

from dataclasses import dataclass

from hermes.research.signals import _FUNDC_HOLD_BARS, _FUNDC_WIN, _FUNDC_Z

REGISTERED_UTC = "2026-06-29"  # frozen pre-registration date (round-2 forward session)


@dataclass(frozen=True)
class PreRegistration:
    """A frozen, dated description of a sleeve to shadow-log for forward OOS."""

    name: str
    registered_utc: str
    book: str  # "crypto" | "fx_metals" | "cross"
    sign: str
    hypothesis: str
    rule: dict
    deflation_trial_count: int
    deflation_breakdown: str
    contamination: str
    is_new_trial: bool  # True → registers +1 in the BookTrialRegistry
    trial_source: str = ""
    notes: str = ""


# ── #1 the LIVE edge — shadow it to get clean, un-contaminated bear/range OOS ──── #
FUNDING_SQUEEZE_CONTINUATION = PreRegistration(
    name="funding_squeeze_continuation",
    registered_utc=REGISTERED_UTC,
    book="crypto",
    sign="LONG (go WITH high funding-z)",
    hypothesis=(
        "The live Strategy #2: extreme-positive perp funding (|z|>=2 over 180 bars) "
        "marks aggressive leveraged demand that CONTINUES ~3-4 days; LONG, ~24x4h hold. "
        "OPEN QUESTION: it was validated over secular crypto appreciation — does it "
        "survive the current bear/range regime, or is it bull-beta in disguise?"
    ),
    rule={
        "direction": "LONG",
        "z_threshold": _FUNDC_Z,        # 2.0 — mirrors signals._FUNDC_Z
        "window_bars": _FUNDC_WIN,      # 180 — mirrors signals._FUNDC_WIN
        "hold_bars": _FUNDC_HOLD_BARS,  # 24 (~96h on 4h) — mirrors signals._FUNDC_HOLD_BARS
        "no_overlap": True,
        "universe": "live 7-asset slate (BTC/ETH/SOL/AVAX/SUI/ARB/WIF)",
        "cost": "honest crypto 20 bps round-trip (fee 7 + slippage 3, ×2)",
        "feed": "Binance USDT-M public 4h klines + funding (read-only)",
    },
    deflation_trial_count=1869,
    deflation_breakdown=(
        "Already inside the 648-config funding sweep → crypto seed 1869. Shadowing the "
        "frozen live cell is NOT a new search, so it adds NO trial."
    ),
    contamination=(
        "The backtested Sharpe is contaminated by bull-beta. The forward bear/range OOS "
        "logged here is the clean test; it accrues at size 0."
    ),
    is_new_trial=False,
    notes="Reproduced byte-identically via hermes.forward.shadow_funding_squeeze.",
)


# ── #2 the pre-registered LEAD — the inverted-slow cross-sectional reversion ───── #
XS_FUNDING_REVERSION = PreRegistration(
    name="xs_funding_reversion",
    registered_utc=REGISTERED_UTC,
    book="crypto",
    sign="INVERTED",
    hypothesis=(
        "Cross-sectional perp funding-z MEAN-REVERTS. Expressed market-neutral: every "
        "14 days rank the 23-perp universe by trailing-180-settlement funding-z, SHORT "
        "the top tercile and LONG the bottom tercile, dollar-neutral. The dollar-neutral "
        "spread removes bull-beta by construction (the killer of every directional test). "
        "The fast version lost (gross annSh -1.38); only this slow, sign-INVERTED spread "
        "carried a +0.58 post-hoc annual lead in S2 v-next."
    ),
    rule={
        "sign": "INVERTED — fade high funding-z (SHORT top tercile, LONG bottom)",
        "cadence_days": 14,
        "neutrality": "dollar-neutral (gross long = gross short)",
        "universe_size": 23,
        "universe": "binance_funding_native deep 23-perp set",
        "buckets": "terciles",
        "funding_z_window_settlements": 180,
        "cost": "honest 2-leg full turnover (~40 bps/rebalance @ fee 7 + slip 3)",
        "null": "MC random-SIGN (long-short), autocorr-preserving",
    },
    deflation_trial_count=1892,
    deflation_breakdown=(
        "1869 (crypto seed) + 22 (S2 v-next: 4 pre-reg + 18 exploratory crypto trials, "
        "to be reconciled into the registry at merge) + 1 (this pre-registration) = 1892. "
        "On THIS branch the registry holds only 1869+1 because S2 v-next is unmerged."
    ),
    contamination=(
        "The +0.58 annual @14d figure is POST-HOC and IN-SAMPLE (contaminated). The "
        "in-sample robustness check decides only WIRE vs DO-NOT-WIRE; the real verdict "
        "comes solely from the forward OOS, deflated at the full count above."
    ),
    is_new_trial=True,
    trial_source="forward_xs_funding_reversion_prereg",
    notes="Shadowed via hermes.forward.shadow_xs_reversion (size 0).",
)

ALL: tuple[PreRegistration, ...] = (FUNDING_SQUEEZE_CONTINUATION, XS_FUNDING_REVERSION)


def register_trials(data_dir=None) -> dict:
    """Register every ``is_new_trial`` pre-registration into the S10 ``BookTrialRegistry``.

    Idempotent: a pre-reg whose ``trial_source`` is already in the book's provenance is
    skipped (``BookTrialRegistry.bump`` has no guard of its own, so the guard lives
    here). This is the committed, tested mechanism — NOT a manual one-off. Returns
    ``{"counts": {...}, "registered": {name: new_count}}``.
    """
    from hermes.research import s10_governance as gov

    reg = gov.BookTrialRegistry(data_dir)
    reg.seed_defaults()  # idempotent: 1869 / 53 / 5
    book_of = {
        "crypto": gov.Book.CRYPTO,
        "fx_metals": gov.Book.FX_METALS,
        "cross": gov.Book.CROSS,
    }
    registered: dict[str, int] = {}
    for pr in ALL:
        if not pr.is_new_trial:
            continue  # shadowing an already-counted live cell adds no trial
        book = book_of[pr.book]
        sources = {e["source"] for e in reg.provenance(book)}
        if pr.trial_source in sources:
            continue  # already registered → never double-count on re-run
        reg.bump(book, n=1, source=pr.trial_source)
        registered[pr.name] = reg.count(book)
    return {"counts": reg.as_dict()["counts"], "registered": registered}


def dead_ends_verdicts() -> dict[str, dict]:
    """Informative-ONLY ``DeadEndsLedger.check()`` verdict for every pre-registration.

    Stream 10 built ``DeadEndsLedger.check()`` (BLOCKED-BY-LEDGER / ALLOWED-NEW-AXIS /
    NOT-A-DEAD-END) but nothing in production ever called it — the forward cron's own
    manifest is the natural, already-printed place for a human to see the verdict
    without a separate report. This is deliberately NON-BLOCKING: it never raises,
    never skips a collector, never touches ``run_forward``'s control flow. It is a
    log line for governance review, nothing more.

    Loads the ledger from the canonical committed ``governance/dead_ends_ledger.json``
    (``s10_governance.governance_registry_dir()``); if that file is absent or
    unreadable, ``DeadEndsLedger.load()`` already falls back to its in-module default
    entries, so this never crashes the cron over a missing/corrupt ledger file.
    """
    from hermes.research import s10_governance as gov

    ledger_path = gov.governance_registry_dir() / "dead_ends_ledger.json"
    ledger = gov.DeadEndsLedger.load(ledger_path)
    verdicts: dict[str, dict] = {}
    for pr in ALL:
        result = ledger.check(pr.hypothesis)
        verdicts[pr.name] = {
            "verdict": result["verdict"].value,
            "matched": result["matched"],
            "reason": result["reason"],
        }
    return verdicts
