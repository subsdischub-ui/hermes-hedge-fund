"""STREAM 10 v-next — hardened methodological RULES as importable, tested guards.

Stream 10 does not search for alpha; it hardens the wall. Round 1 shipped the per-book deflation
machinery (``s10_governance``). This module turns five adversarial lessons the merged streams paid
for into small, falsifiable helpers the OTHER streams import — so the lab cannot re-make the same
anti-conservative mistake twice:

  R1  ``robust_ic`` / ``ic_significance`` — IC on DELTAS/RANKS, never on levels. (S6: a level IC is
      anti-conservative — persistence inflates its t; ~16% false-positive rate at N=60.)
  R2  ``chance_corrected_agreement`` / ``state_stability_excess`` / ``excess_over_chance`` — every
      stability/agreement number must be reported as EXCESS over chance. (S8: a collapsed HMM scores
      raw agreement 1.0 = pure chance; report excess, never the raw rate.)
  R3  ``coactive_correlation`` — orthogonality is judged on CO-ACTIVE weeks + participation ratio,
      not the zero-filled grid. (S5: sparse sleeves look orthogonal only because zeros dominate.)
  R4  ``future_perturbation_leak_test`` / ``assert_no_lookahead`` / ``replay_leak_audits`` —
      leak-injection on every DRIVER before a verdict, REPLAYED on every SURVIVOR at merge.
      (AM-4: composition bugs fabricate positives; perturb the future, the past signal must not move.)
  R5  ``block_events_by_time`` / ``effective_independent_events`` / ``cluster_block_bootstrap_pvalue``
      — nulls are CLUSTER-AWARE for any signal that co-fires across instruments. (S2: a squeeze
      co-fires on the alts, so the i.i.d. null is anti-conservative; block by macro-event.)

Pure functions, no side effects at import. All Sharpes/statistics are per-period; nothing trades.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import rankdata as _rankdata
from scipy.stats import t as _student_t

__all__ = [
    "LeakDetected",
    "assert_no_lookahead",
    "block_events_by_time",
    "chance_corrected_agreement",
    "cluster_block_bootstrap_pvalue",
    "coactive_correlation",
    "effective_independent_events",
    "effective_sample_size",
    "excess_over_chance",
    "future_perturbation_leak_test",
    "ic_significance",
    "replay_leak_audits",
    "robust_ic",
    "state_stability_excess",
]


# =========================================================================== #
# R1 — IC on deltas/ranks, NEVER on levels (S6)
# =========================================================================== #
def effective_sample_size(series: Sequence[float]) -> float:
    """AR(1) effective sample size ``n·(1-ρ₁)/(1+ρ₁)``, clamped to ``[2, n]``.

    Persistence shrinks the number of effectively-independent observations: at ρ₁≈0.9 sixty
    autocorrelated points carry only a handful of independent draws. The clamp keeps a
    negatively-autocorrelated (e.g. differenced) series from reporting MORE than n.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 3:
        return float(n)
    x0, x1 = x[:-1], x[1:]
    if x0.std() <= 0 or x1.std() <= 0:
        return float(n)
    r1 = float(np.corrcoef(x0, x1)[0, 1])
    if not np.isfinite(r1):
        r1 = 0.0
    # clamp into the OPEN interval so 1+r1 stays strictly > 0 — a perfectly anti-persistent
    # (alternating) series gives r1 == -1.0 exactly, which differencing readily produces.
    r1 = float(np.clip(r1, -0.999999, 0.999999))
    factor = (1.0 - r1) / (1.0 + r1)
    return float(np.clip(n * factor, 2.0, float(n)))


def robust_ic(signal: Sequence[float], forward: Sequence[float], mode: str = "delta") -> float:
    """Information coefficient on DIFFERENCES (``mode='delta'`` → Pearson of first differences) or
    on rank-differences (``mode='rank'`` → Spearman of first differences).

    ``mode='level'`` is a hard error: an IC computed on raw levels is anti-conservative because the
    common persistence/trend in two series inflates the correlation's significance (S6: ~16% FPR at
    N=60). Differencing removes the persistence that drives the spurious correlation.
    """
    if mode == "level":
        raise ValueError(
            "IC-in-levels is BANNED (S6: anti-conservative — persistence inflates the t, ~16% FPR "
            "at N=60). Use mode='delta' (first differences) or mode='rank' (rank of differences)."
        )
    if mode not in ("delta", "rank"):
        raise ValueError(f"unknown IC mode {mode!r}; use 'delta' or 'rank'.")
    s = np.diff(np.asarray(signal, dtype=float))
    f = np.diff(np.asarray(forward, dtype=float))
    if s.size < 2:
        return 0.0
    if mode == "rank":
        s = _rankdata(s)
        f = _rankdata(f)
    if s.std() <= 0 or f.std() <= 0:
        return 0.0
    return float(np.corrcoef(s, f)[0, 1])


def ic_significance(signal: Sequence[float], forward: Sequence[float], mode: str = "delta") -> dict:
    """Robust IC + an honest two-sided p-value.

    Computes ``robust_ic`` (deltas/ranks) and a t-test whose degrees of freedom are bounded by the
    effective sample size of the differenced inputs — so a residually-autocorrelated series cannot
    borrow significance it does not have. Returns ``{ic, n_eff, t_stat, p_value, mode}``.
    """
    ic = robust_ic(signal, forward, mode=mode)
    ds = np.diff(np.asarray(signal, dtype=float))
    df = np.diff(np.asarray(forward, dtype=float))
    n = ds.size
    if n < 3:
        return {"ic": ic, "n_eff": float(n), "t_stat": 0.0, "p_value": 1.0, "mode": mode}
    n_eff = float(np.clip(min(effective_sample_size(ds), effective_sample_size(df)), 2.0, float(n)))
    if abs(ic) >= 1.0:
        return {"ic": ic, "n_eff": n_eff, "t_stat": float("inf"), "p_value": 0.0, "mode": mode}
    if n_eff <= 2.0:
        return {"ic": ic, "n_eff": n_eff, "t_stat": 0.0, "p_value": 1.0, "mode": mode}
    t_stat = ic * np.sqrt((n_eff - 2.0) / (1.0 - ic * ic))
    p_value = float(2.0 * _student_t.sf(abs(t_stat), df=n_eff - 2.0))
    return {"ic": ic, "n_eff": n_eff, "t_stat": float(t_stat), "p_value": p_value, "mode": mode}


# =========================================================================== #
# R2 — chance-correction on any stability / agreement metric (S8)
# =========================================================================== #
def excess_over_chance(observed: float, chance: float) -> float:
    """``(observed - chance)/(1 - chance)`` — the share of the achievable margin actually captured.

    0.0 when ``observed == chance`` (the S8 trap: a 0.88 raw agreement against a 0.88 chance level is
    NO information) and 0.0 when ``chance >= 1`` (a collapsed/degenerate target carries no margin).
    """
    if chance >= 1.0:
        return 0.0
    return float((observed - chance) / (1.0 - chance))


def chance_corrected_agreement(labels_a: Sequence, labels_b: Sequence) -> float:
    """Cohen's κ — agreement corrected for the agreement expected by chance from the marginals.

    A constant ("collapsed") rater scores κ≈0 even when its RAW agreement is high, which is exactly
    why a stability/accuracy number must never be reported raw (S8: the collapsed HMM that scores 1.0).
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    n = a.size
    if n == 0:
        return 0.0
    p_o = float(np.mean(a == b))
    cats = np.unique(np.concatenate([a, b]))
    p_e = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return excess_over_chance(p_o, p_e)


def state_stability_excess(states: Sequence) -> dict:
    """Persistence of a regime/state series reported as EXCESS over chance.

    ``raw_persistence`` = P(stateₜ == stateₜ₋₁); ``chance_persistence`` = Σ pₖ² (the persistence a
    memoryless draw from the same marginals would show); ``excess`` corrects the former by the latter.
    A collapsed single-state series has raw 1.0 but excess 0.0 (S8).
    """
    s = np.asarray(states)
    n = s.size
    if n < 2:
        return {"raw_persistence": 0.0, "chance_persistence": 0.0, "excess": 0.0}
    raw = float(np.mean(s[1:] == s[:-1]))
    _, counts = np.unique(s, return_counts=True)
    p = counts / n
    chance = float(np.sum(p * p))
    return {
        "raw_persistence": raw,
        "chance_persistence": chance,
        "excess": excess_over_chance(raw, chance),
    }


# =========================================================================== #
# R3 — orthogonality on co-active weeks, not the zero-filled grid (S5)
# =========================================================================== #
def coactive_correlation(
    a: Sequence[float],
    b: Sequence[float],
    *,
    active_threshold: float = 0.0,
    min_coactive: int = 20,
) -> dict:
    """Correlation of two sleeve return streams measured only on weeks where BOTH are active.

    The full zero-filled grid dilutes the correlation of sparse sleeves toward 0, making colinear
    sleeves look orthogonal (S5). The honest measure conditions on co-activity. ``reliable`` is
    ``False`` when fewer than ``min_coactive`` overlapping weeks exist — too thin to judge.
    Returns ``{corr_full, corr_coactive, n_coactive, n_total, participation_ratio, reliable}``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    corr_full = float(np.corrcoef(a, b)[0, 1]) if (n >= 2 and a.std() > 0 and b.std() > 0) else 0.0
    mask = (np.abs(a) > active_threshold) & (np.abs(b) > active_threshold)
    n_co = int(mask.sum())
    if n_co >= 2 and a[mask].std() > 0 and b[mask].std() > 0:
        corr_coactive = float(np.corrcoef(a[mask], b[mask])[0, 1])
    else:
        corr_coactive = 0.0
    return {
        "corr_full": corr_full,
        "corr_coactive": corr_coactive,
        "n_coactive": n_co,
        "n_total": n,
        "participation_ratio": (n_co / n) if n else 0.0,
        "reliable": n_co >= min_coactive,
    }


# =========================================================================== #
# R4 — leak-injection (future perturbation) + survivor replay at merge (AM-4)
# =========================================================================== #
class LeakDetected(AssertionError):
    """Raised when a signal builder's output at time t depends on data strictly after t."""


def future_perturbation_leak_test(
    build_signal: Callable[[np.ndarray], Sequence[float]],
    data: Sequence[float],
    *,
    rng: np.random.Generator,
    n_perturb: int = 12,
    tol: float = 1e-9,
) -> dict:
    """Causal-leak detector: perturb the data strictly AFTER each decision point and confirm the
    signal AT/BEFORE that point does not move.

    ``build_signal(data)`` must return an array the same length as ``data`` AND be DETERMINISTIC
    (seed any internal RNG): the test compares signal rebuilds, so a builder that varies run-to-run
    would be falsely flagged as leaking. For a sample of decision points t we rebuild on data with
    indices > t perturbed; if any of ``signal[:t+1]`` changes, the builder peeked into the future.
    This is the gold-standard look-ahead test (it catches the S2 same-day regime leak and the S8
    smoothed-posterior leak that fixed thresholds miss).
    Returns ``{leaked, n_leaked_points, checked_points, leaked_at}``.
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    baseline = np.asarray(build_signal(data), dtype=float)
    if baseline.shape[0] != n:
        raise ValueError("build_signal must return an array the same length as data")
    # self-check: a stochastic builder breaks the test's premise (its output differs independent of
    # any perturbation), so refuse LOUDLY rather than report a spurious leak.
    baseline2 = np.asarray(build_signal(data), dtype=float)
    if not np.allclose(baseline, baseline2, atol=tol, rtol=0.0):
        raise ValueError(
            "build_signal must be deterministic for the leak test to be valid (two builds on the "
            "same data differed); seed its internal RNG before calling future_perturbation_leak_test."
        )
    lo, hi = 1, n - 2  # strictly inside so a future exists to perturb
    if hi <= lo:
        return {"leaked": False, "n_leaked_points": 0, "checked_points": 0, "leaked_at": []}
    pts = np.unique(np.linspace(lo, hi, num=min(n_perturb, hi - lo + 1)).astype(int))
    leaked_at: list[int] = []
    for t0 in pts:
        d2 = data.copy()
        k = n - t0 - 1
        d2[t0 + 1 :] = d2[t0 + 1 :] + 10.0 + rng.standard_normal(k)  # perturb strictly after t0
        sig2 = np.asarray(build_signal(d2), dtype=float)
        if not np.allclose(sig2[: t0 + 1], baseline[: t0 + 1], atol=tol, rtol=0.0):
            leaked_at.append(int(t0))
    return {
        "leaked": bool(leaked_at),
        "n_leaked_points": len(leaked_at),
        "checked_points": int(pts.size),
        "leaked_at": leaked_at,
    }


def assert_no_lookahead(
    build_signal: Callable[[np.ndarray], Sequence[float]],
    data: Sequence[float],
    *,
    rng: np.random.Generator,
    **kwargs,
) -> dict:
    """Run ``future_perturbation_leak_test`` and raise :class:`LeakDetected` if it leaks."""
    res = future_perturbation_leak_test(build_signal, data, rng=rng, **kwargs)
    if res["leaked"]:
        raise LeakDetected(
            f"future-perturbation leak at {res['n_leaked_points']} decision point(s): {res['leaked_at']}"
        )
    return res


def replay_leak_audits(survivors: Sequence[dict]) -> dict:
    """Merge gate (AM-4): every surviving sleeve must have REPLAYED its leak audit on the merged
    artifact. A prior clean audit on the branch is not enough — composition bugs fabricate positives.

    Each survivor is a dict with ``name`` and ``leak_audit_replayed``. Returns
    ``{all_clean, blocked, checked}``; ``blocked`` names every survivor lacking a replayed audit.
    """
    blocked = [s["name"] for s in survivors if not s.get("leak_audit_replayed", False)]
    return {
        "all_clean": len(blocked) == 0,
        "blocked": blocked,
        "checked": [s["name"] for s in survivors],
    }


# =========================================================================== #
# R5 — cluster-aware nulls for co-firing signals (S2)
# =========================================================================== #
def block_events_by_time(timestamps: Sequence[float], window: float) -> np.ndarray:
    """Assign each event a macro-event cluster id: events within ``window`` of a cluster's START
    share its id. Events that co-fire (same shock) thus collapse to one independent observation.
    """
    ts = np.asarray(timestamps, dtype=float)
    order = np.argsort(ts, kind="stable")
    clusters = np.empty(ts.size, dtype=int)
    cid = -1
    anchor: float | None = None
    for idx in order:
        if anchor is None or ts[idx] - anchor > window:
            cid += 1
            anchor = float(ts[idx])
        clusters[idx] = cid
    return clusters


def effective_independent_events(cluster_ids: Sequence) -> int:
    """The honest sample size for a co-firing signal = the number of distinct macro-event clusters,
    NOT the inflated raw event count (S2: 35 trades collapse to 21 macro-events)."""
    return int(np.unique(np.asarray(cluster_ids)).size)


def cluster_block_bootstrap_pvalue(
    values: Sequence[float],
    cluster_ids: Sequence,
    *,
    rng: np.random.Generator,
    n_boot: int = 2000,
    alternative: str = "greater",
) -> float:
    """Cluster-aware sign-flip null for ``mean(values) ≠/>/< 0``.

    The sign is flipped per CLUSTER (whole macro-event), not per event, preserving the cross-instrument
    co-firing structure. With each event its own cluster this reduces to the i.i.d. null — which is
    anti-conservative when events co-fire (S2). p = (#null ≥ observed + 1)/(n_boot + 1).
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 1.0  # degenerate: no events → no evidence against the null
    _, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    k = int(inv.max()) + 1 if inv.size else 0
    if k == 0:
        return 1.0
    observed = float(values.mean())
    count = 0
    for _ in range(n_boot):
        signs = rng.choice(np.array([-1.0, 1.0]), size=k)
        null_mean = float((values * signs[inv]).mean())
        if alternative == "greater":
            hit = null_mean >= observed
        elif alternative == "less":
            hit = null_mean <= observed
        else:  # two-sided
            hit = abs(null_mean) >= abs(observed)
        count += int(hit)
    return (count + 1) / (n_boot + 1)
