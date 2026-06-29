"""STREAM 10 v-next — the JEWEL combination gate (SPEC, not executed on any real pillar).

The "jewel" is the eventual combination of surviving market-neutral pillars into one book. The
MERGE IS DEFERRED: S2 delivered no market-neutral pillar, so there is nothing to combine today.
This module is therefore a *specification in code* — fully tested on synthetic fixtures, ready for
the day a pillar survives, and it refuses to run without explicitly-named confirmed pillars.

A combination is treated as ONE pre-registered trial (it bumps the deflation count by 1) and must
clear, on the COMBINED series:
  1. a combined null — random-sign PER sleeve, autocorrelation preserved (whole-series sign flip),
     enumerated exactly over all 2^K patterns when K is small. NB: the observed book IS the all-+1
     pattern and is counted, so the best achievable p is 2/(2^K+1); against a 0.05 floor this means
     a combination needs K >= 6 sleeves to beat the null at all — by design, sign-luck dominates below;
  2. DSR at the TRUE (book_trial_count + 1) trial count, PBO, and CPCV OOS positivity;
  3. LOO-Δ-Sharpe > 0 for EVERY sleeve (the unit is the book, never a solo Sharpe);
  4. orthogonality judged on CO-ACTIVE weeks (s10_rules.coactive_correlation), not the zero grid;
  5. the Task-4 OOS-FORWARD precondition (clean forward, real regime, AM-4 leak audit replayed).

It reuses the committed validation machinery (book/cpcv/pbo/stats) rather than rebuilding it.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from hermes.learning.stats import deflated_sharpe_ratio, sharpe_ratio
from hermes.research.s10_governance import (
    Book,
    ForwardOOSEvidence,
    PromotionThresholds,
    bar_for_book,
)
from hermes.research.s10_rules import coactive_correlation
from hermes.validation.book import (  # CPCV is reached via cpcv_book_distribution (imported lazily below)
    align_sleeves,
    combine_book,
    leave_one_out_delta_sharpe,
)
from hermes.validation.pbo import probability_backtest_overfitting

__all__ = ["combined_sign_flip_pvalue", "jewel_combination_gate"]


def _as_series(sleeve_returns: dict[str, Sequence[float]]) -> dict[str, pd.Series]:
    return {k: pd.Series(np.asarray(v, dtype=float)) for k, v in sleeve_returns.items()}


def combined_sign_flip_pvalue(
    sleeve_series: dict[str, pd.Series],
    weighting: str,
    *,
    rng: np.random.Generator,
    n_sample: int = 4000,
) -> float:
    """Random-sign-PER-sleeve null for the combined book Sharpe, autocorrelation preserved.

    Each sleeve's WHOLE return series is multiplied by ±1 (preserving its autocorrelation), the book
    is recombined and its Sharpe recomputed. With K ≤ 14 sleeves all 2^K sign patterns are enumerated
    exactly (deterministic p); otherwise ``n_sample`` random patterns are drawn (with the observed
    all-+1 pattern always included). Because the observed book IS the all-+1 pattern and is counted,
    p ≥ 2/(2^K+1) always — so beating a 0.05 floor needs K ≥ 6. p = (#null ≥ observed + 1)/(N + 1).
    """
    names = list(sleeve_series)
    k = len(names)
    observed = sharpe_ratio(combine_book(sleeve_series, weighting=weighting).to_numpy())

    if k <= 14:
        patterns: Sequence[tuple[int, ...]] = list(itertools.product((-1, 1), repeat=k))
    else:
        # always include the observed all-+1 pattern so the floor 2/(N+1) holds in this branch too
        patterns = [tuple([1] * k)] + [tuple(rng.choice((-1, 1), size=k)) for _ in range(n_sample - 1)]

    count = 0
    for signs in patterns:
        flipped = {nm: sleeve_series[nm] * s for nm, s in zip(names, signs, strict=True)}
        null_sharpe = sharpe_ratio(combine_book(flipped, weighting=weighting).to_numpy())
        count += int(null_sharpe >= observed)
    return (count + 1) / (len(patterns) + 1)


def _max_coactive_rho(sleeve_series: dict[str, pd.Series], min_coactive: int) -> tuple[float, bool]:
    """Largest reliable |co-active correlation| over all sleeve pairs (R3). Returns
    (max_abs_rho, any_reliable_pair)."""
    names = list(sleeve_series)
    worst = 0.0
    any_reliable = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            res = coactive_correlation(
                sleeve_series[names[i]].to_numpy(),
                sleeve_series[names[j]].to_numpy(),
                min_coactive=min_coactive,
            )
            if res["reliable"]:
                any_reliable = True
                worst = max(worst, abs(res["corr_coactive"]))
    return worst, any_reliable


def jewel_combination_gate(
    sleeve_returns: dict[str, Sequence[float]],
    *,
    book: Book,
    book_trial_count: int,
    forward: ForwardOOSEvidence | None,
    confirmed_pillars: Sequence[str],
    weighting: str = "inv_vol",
    thresholds: PromotionThresholds | None = None,
    sr_variance: float | None = None,
    rng: np.random.Generator | None = None,
    min_coactive: int = 20,
    min_cpcv_frac_positive: float = 0.5,
) -> dict:
    """Evaluate a candidate jewel combination. Returns a verdict dict
    ``{verdict, passed, failed, reasons, metrics}``.

    SAFETY: the gate is a SPEC. It REJECTS unless ``confirmed_pillars`` lists ≥2 sleeves AND every
    sleeve in ``sleeve_returns`` is a confirmed pillar — there is no such pillar today, so on the
    real book this gate is never reached (the merge is deferred). Tests exercise it on SYNTHETIC
    fixtures only.

    ``sr_variance`` is the deflation input for the DSR: per Bailey-López de Prado it must be the
    cross-trial Sharpe VARIANCE of the ``book_trial_count`` searched candidates (sourced from the
    per-book trial sweep), NOT the dispersion of the few chosen sleeves. If left ``None`` it falls
    back to the chosen sleeves' solo-Sharpe variance, which is far too small at a large trial count
    and DEFANGS the deflation (the reason string flags this) — pass the real searched dispersion.
    """
    bar = thresholds or bar_for_book(book)
    rng = rng or np.random.default_rng(0)
    series = _as_series(sleeve_returns)

    passed: list[str] = []
    failed: list[str] = []
    reasons: dict[str, str] = {}
    metrics: dict = {}

    def record(key: str, ok: bool, reason: str) -> None:
        (passed if ok else failed).append(key)
        reasons[key] = reason

    # ---- SAFETY: confirmed pillars must exist (none today → merge deferred) -------------- #
    pillars = set(confirmed_pillars)
    pillars_ok = len(pillars) >= 2 and set(series).issubset(pillars)
    record(
        "confirmed_pillars",
        pillars_ok,
        f"{len(pillars)} confirmed market-neutral pillar(s); all sleeves are pillars"
        if pillars_ok
        else "NO confirmed market-neutral pillar exists (the merge is DEFERRED). The jewel gate is "
        "a spec: it does not run on a real book until ≥2 sleeves are confirmed pillars.",
    )

    # the combination IS a single new pre-registered trial → deflate at book_trial_count + 1
    n_trials = int(book_trial_count) + 1
    metrics["n_trials"] = n_trials

    combined = combine_book(series, weighting=weighting)
    combined_arr = combined.to_numpy()
    combined_sharpe = sharpe_ratio(combined_arr)
    metrics["combined_sharpe"] = combined_sharpe

    # ---- 1. combined random-sign null (autocorr preserved) ------------------------------ #
    sign_p = combined_sign_flip_pvalue(series, weighting, rng=rng)
    metrics["combined_sign_null_p"] = sign_p
    record(
        "combined_sign_null",
        sign_p < bar.max_mc_null_p,
        f"combined random-sign-per-sleeve null p={sign_p:.4g} (floor: must be < {bar.max_mc_null_p}; "
        f"min achievable = 1/2^K with K={len(series)} sleeves)",
    )

    # ---- 2. DSR @ true count, PBO, CPCV OOS positivity ---------------------------------- #
    solo_sharpes = [sharpe_ratio(series[s].to_numpy()) for s in series]
    if sr_variance is not None:
        sr_var = float(sr_variance)
        sr_var_note = "searched-candidate dispersion (correct deflation input)"
    else:
        sr_var = float(np.var(solo_sharpes, ddof=1)) if len(solo_sharpes) >= 2 else 0.0
        sr_var_note = (
            "FALLBACK to chosen-sleeve solo-Sharpe variance — too small at a large trial count; "
            "pass sr_variance from the per-book trial sweep to give the deflation teeth"
        )
    metrics["sr_variance"] = sr_var
    dsr = deflated_sharpe_ratio(combined_arr, sr_var, n_trials)
    metrics["dsr"] = dsr
    record(
        "deflated_sharpe",
        dsr >= bar.min_dsr,
        f"DSR={dsr:.4g} at true n_trials={n_trials} (need ≥ {bar.min_dsr} for the {book.value} book); "
        f"sr_variance={sr_var:.4g} [{sr_var_note}]",
    )

    aligned = align_sleeves(series)
    pbo = probability_backtest_overfitting(aligned.to_numpy()) if aligned.shape[1] >= 2 else 0.0
    metrics["pbo"] = pbo
    record("pbo", pbo <= bar.max_pbo, f"PBO={pbo:.4g} (need ≤ {bar.max_pbo})")

    from hermes.validation.book import cpcv_book_distribution

    cpcv = cpcv_book_distribution(combined_arr.tolist())
    metrics["cpcv"] = cpcv
    record(
        "cpcv_oos_positive",
        cpcv["frac_positive"] > min_cpcv_frac_positive,
        f"CPCV OOS frac_positive={cpcv['frac_positive']:.4g} (need > {min_cpcv_frac_positive})",
    )

    # ---- 3. LOO-Δ-Sharpe > 0 PER sleeve ------------------------------------------------- #
    loo = leave_one_out_delta_sharpe(series, weighting=weighting)
    metrics["loo_delta_sharpe"] = loo
    worst_loo = min(loo.values()) if loo else -1.0
    bad_loo = [s for s, d in loo.items() if d <= 0.0]
    record(
        "loo_per_sleeve",
        len(bad_loo) == 0,
        "every sleeve is net-accretive (LOO-Δ-Sharpe > 0)"
        if not bad_loo
        else f"sleeve(s) with LOO-Δ-Sharpe ≤ 0 (drag the book): {bad_loo} (worst {worst_loo:.4g})",
    )

    # ---- 4. orthogonality on CO-ACTIVE weeks ------------------------------------------- #
    max_rho, any_reliable = _max_coactive_rho(series, min_coactive)
    metrics["max_coactive_rho"] = max_rho
    record(
        "orthogonality_coactive",
        (not any_reliable) or (max_rho < bar.max_abs_rho),
        f"max co-active |ρ|={max_rho:.4g} over reliable pairs (need < {bar.max_abs_rho})",
    )

    # ---- 5. OOS-FORWARD precondition (Task 4) ------------------------------------------- #
    if forward is None:
        record(
            "oos_forward_present",
            False,
            "no clean OOS-forward accrued — the Forward-Shadow has not been exited; no in-sample "
            "combination, however clean, can CONFIRM the jewel.",
        )
    else:
        record("oos_forward_present", True, "forward evidence attached")
        record(
            "oos_forward_regime_real",
            forward.regime_real,
            "forward window is a genuine regime" if forward.regime_real else "forward is not a real regime",
        )
        record(
            "oos_forward_leak_replayed",
            forward.leak_replayed,
            "AM-4 leak audit replayed on the merged combination"
            if forward.leak_replayed
            else "AM-4 leak audit NOT replayed on the merged combination",
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

    verdict = "PROMOTE" if not failed else "REJECT"
    return {
        "verdict": verdict,
        "passed": passed,
        "failed": failed,
        "reasons": reasons,
        "metrics": metrics,
        "book": book.value,
    }
