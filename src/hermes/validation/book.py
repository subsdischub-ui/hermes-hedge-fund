"""Combined-BOOK validation harness — the §7 Renaissance thesis made measurable.

The prior arc judged every signal *in isolation*, by its solo per-trade Sharpe, and
killed anything below a hero threshold. That is the wrong unit. A sleeve with Sharpe
0.2 that is rho≈0 to the book can be strongly net-accretive; ten such weak, mutually
uncorrelated sleeves combine toward Sharpe ~1+. So the lab's true objective is to
assemble a book of many honest, weak, orthogonal sleeves and prove the *book* is great.

This module points the existing CPCV/PBO/Deflated-Sharpe machinery at a **weighted
heterogeneous BOOK return series** (not config-variants of one strategy — all the
in-repo harness has ever done) and accepts a sleeve by its **leave-one-out Δ-Sharpe
contribution to the book**, not its solo Sharpe.

Honesty by construction:
  * Sleeves trade at different cadences/instruments. We mark the book on a common grid
    (the union of every sleeve's mark periods) and treat a sleeve with no mark in a
    period as FLAT (0.0) — it simply earns nothing that period, never NaN-propagates.
  * Combination weights are CAUSAL: the weight applied to a sleeve in period t is set
    from that sleeve's realised returns strictly BEFORE t (expanding inverse-variance),
    so the book return series is one a trader could actually have earned — no
    in-sample weighting leak that would inflate the very Sharpe we then deflate.

All functions are pure. The book series produced here is the object the gate (§3) is
run against; the per-sleeve LOO-Δ-Sharpe is the accept/reject criterion (§7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.learning.stats import deflated_sharpe_ratio, sharpe_ratio
from hermes.validation.cpcv import combinatorial_purged_splits
from hermes.validation.pbo import probability_backtest_overfitting

__all__ = [
    "align_sleeves",
    "book_report",
    "causal_hrp_weights",
    "causal_inverse_vol_weights",
    "combine_book",
    "cpcv_book_distribution",
    "leave_one_out_delta_sharpe",
    "orthogonality_matrix",
    "weighting_comparison",
]


def align_sleeves(sleeve_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Align heterogeneous per-sleeve return series onto a common mark grid.

    The index is the sorted union of every sleeve's periods; a sleeve absent in a
    period is FLAT (filled 0.0), i.e. it holds no risk and earns nothing there — the
    honest treatment of sleeves that trade on different cadences. Column order follows
    insertion order of ``sleeve_returns``.
    """
    if not sleeve_returns:
        return pd.DataFrame()
    frame = pd.DataFrame(sleeve_returns)
    frame = frame.sort_index()
    return frame.fillna(0.0)


def causal_inverse_vol_weights(
    aligned: pd.DataFrame, min_periods: int = 8
) -> pd.DataFrame:
    """Expanding inverse-variance weights, set from data strictly before each period.

    The weight on sleeve *s* in period *t* is ∝ 1/Var(r_s over periods < t), normalised
    so each row sums to 1. Using only the trailing window (everything before t) makes
    the resulting book return causal: weights are knowable at the open of t. Until a
    sleeve has ``min_periods`` prior observations — or whenever a trailing variance is
    zero/undefined — the row falls back to equal weight (1/N), the robust low-data
    default (DeMiguel-Garlappi-Uppal). Returns a TxN frame aligned to ``aligned``.
    """
    n_periods, n_sleeves = aligned.shape
    if n_sleeves == 0:
        return pd.DataFrame(index=aligned.index)
    equal = np.full(n_sleeves, 1.0 / n_sleeves)
    arr = aligned.to_numpy(dtype=float)
    weights = np.empty((n_periods, n_sleeves), dtype=float)
    for t in range(n_periods):
        past = arr[:t]  # strictly before t — the causality guarantee
        if past.shape[0] < min_periods:
            weights[t] = equal
            continue
        var = past.var(axis=0, ddof=1)
        if not np.all(np.isfinite(var)) or np.any(var <= 0.0):
            weights[t] = equal
            continue
        inv = 1.0 / var
        weights[t] = inv / inv.sum()
    return pd.DataFrame(weights, index=aligned.index, columns=aligned.columns)


def _hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Static Hierarchical Risk Parity weights from a covariance matrix (AFML ch.16).

    López de Prado's recipe: correlation distance d=√((1−ρ)/2), SINGLE-linkage
    agglomerative clustering, quasi-diagonalisation by the dendrogram leaf order,
    then recursive bisection allocating inversely to *cluster* variance. Unlike
    naive inverse-variance, this accounts for correlation — redundant (correlated)
    sleeves are clustered and share one half of the split, so the book diversifies
    BETWEEN clusters instead of piling capital into a correlated bloc. Returns a
    length-N weight vector summing to 1, in the ORIGINAL column order of ``cov``.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0.0, cov / denom, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = list(leaves_list(link))  # quasi-diagonal order (original indices)

    def _cluster_var(items: list[int]) -> float:
        sub = cov[np.ix_(items, items)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    w = np.ones(n)
    clusters: list[list[int]] = [order]
    while clusters:
        # Bisect every cluster of >1 item at the MIDPOINT of its quasi-diagonal order
        # (the canonical LdP split — the ordering, not the tree, decides the cut).
        clusters = [
            part
            for cl in clusters
            for part in (cl[: len(cl) // 2], cl[len(cl) // 2 :])
            if len(cl) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_left, v_right = _cluster_var(left), _cluster_var(right)
            total = v_left + v_right
            alpha = 1.0 - v_left / total if total > 0.0 else 0.5
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w


def causal_hrp_weights(aligned: pd.DataFrame, min_periods: int = 8) -> pd.DataFrame:
    """Expanding causal HRP weights — the §1.2 allocator, zero look-ahead.

    The forward-looking sibling of ``causal_inverse_vol_weights``: the weight on
    sleeve *s* in period *t* comes from the full HRP recipe (cluster →
    quasi-diagonalise → recursive bisection) run on the trailing covariance —
    returns strictly BEFORE t — so the resulting book is one a trader could have
    earned. Until ``min_periods`` prior observations accrue, or whenever the trailing
    covariance is degenerate (non-finite or any zero variance), the row falls back
    to equal weight (1/N), the robust low-data default (DeMiguel-Garlappi-Uppal).
    """
    n_periods, n_sleeves = aligned.shape
    if n_sleeves == 0:
        return pd.DataFrame(index=aligned.index)
    equal = np.full(n_sleeves, 1.0 / n_sleeves)
    arr = aligned.to_numpy(dtype=float)
    weights = np.empty((n_periods, n_sleeves), dtype=float)
    for t in range(n_periods):
        past = arr[:t]  # strictly before t — the causality guarantee
        if past.shape[0] < min_periods:
            weights[t] = equal
            continue
        var = past.var(axis=0, ddof=1)
        if not np.all(np.isfinite(var)) or np.any(var <= 0.0):
            weights[t] = equal
            continue
        try:
            w = _hrp_weights(np.cov(past, rowvar=False, ddof=1))
        except Exception:
            w = equal
        if w.shape != (n_sleeves,) or not np.all(np.isfinite(w)) or w.sum() <= 0.0:
            w = equal
        weights[t] = w
    return pd.DataFrame(weights, index=aligned.index, columns=aligned.columns)


def combine_book(
    sleeve_returns: dict[str, pd.Series],
    weighting: str = "inv_vol",
    min_periods: int = 8,
) -> pd.Series:
    """Combine sleeves into one causal book return series.

    ``weighting``:
      * ``"inv_vol"`` (default) — causal expanding inverse-variance (the proven
        low-data allocator; weights known at the open of each period).
      * ``"hrp"`` — causal expanding Hierarchical Risk Parity (correlation-aware
        clustering; weights known at the open of each period).
      * ``"equal"`` — fixed 1/N each period (the DGU robustness baseline).

    Book return in period t = Σ_s w_{s,t} · r_{s,t}. Returns an empty Series for no
    sleeves.
    """
    aligned = align_sleeves(sleeve_returns)
    if aligned.empty:
        return pd.Series(dtype=float)
    if weighting == "equal":
        return aligned.mean(axis=1)
    if weighting == "inv_vol":
        w = causal_inverse_vol_weights(aligned, min_periods=min_periods)
        return (aligned * w).sum(axis=1)
    if weighting == "hrp":
        w = causal_hrp_weights(aligned, min_periods=min_periods)
        return (aligned * w).sum(axis=1)
    raise ValueError(
        f"unknown weighting: {weighting!r} (use 'inv_vol', 'hrp' or 'equal')"
    )


def leave_one_out_delta_sharpe(
    sleeve_returns: dict[str, pd.Series],
    weighting: str = "inv_vol",
    min_periods: int = 8,
) -> dict[str, float]:
    """Each sleeve's leave-one-out Δ-Sharpe contribution to the BOOK (§7 criterion).

    Δ_s = Sharpe(full book) - Sharpe(book rebuilt WITHOUT sleeve *s*). A positive Δ_s
    means the book is genuinely better *with* the sleeve — the honest accept test that
    replaces solo per-trade Sharpe. A weak sleeve (low solo Sharpe) that is rho≈0 to the
    rest will show Δ_s > 0; a redundant or loss-making sleeve will show Δ_s ≤ 0. A
    single-sleeve book reports Δ = full Sharpe (removing it leaves nothing → 0).
    """
    ids = list(sleeve_returns)
    full = sharpe_ratio(combine_book(sleeve_returns, weighting, min_periods).tolist())
    out: dict[str, float] = {}
    for sid in ids:
        rest = {k: v for k, v in sleeve_returns.items() if k != sid}
        if not rest:
            out[sid] = full  # book is nothing without its only sleeve
            continue
        without = sharpe_ratio(combine_book(rest, weighting, min_periods).tolist())
        out[sid] = full - without
    return out


def orthogonality_matrix(sleeve_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise Pearson rho of the aligned per-sleeve return streams.

    The whole point of a new sleeve is *new information*, not re-discovered beta, so we
    prize |rho| < 0.3 against every existing sleeve. Unit diagonal, symmetric. Periods
    where a sleeve is flat (0.0) are included — that is the honest joint distribution of
    the book's marks.
    """
    aligned = align_sleeves(sleeve_returns)
    if aligned.shape[1] == 0:
        return pd.DataFrame()
    return aligned.corr()


def cpcv_book_distribution(
    book_returns: list[float], n_groups: int = 6, k_test: int = 2, embargo: int = 0
) -> dict:
    """CPCV multi-path test-fold Sharpe distribution over the BOOK return series.

    Same engine as ``report.cpcv_sharpe_distribution`` (reused via the shared
    ``combinatorial_purged_splits`` primitive) but aimed at the book stream: a real
    book edge stays positive across most of the C(n_groups, k_test) held-out paths; an
    overfit combination does not. Reimplemented here (not imported from ``report``) so
    the book harness stays decoupled from the heavy skill-engine import chain.
    """
    rets = list(book_returns)
    zero = {
        "n_paths": 0,
        "sharpe_mean": 0.0,
        "sharpe_std": 0.0,
        "sharpe_median": 0.0,
        "frac_positive": 0.0,
    }
    if len(rets) < n_groups:
        return zero
    arr = np.asarray(rets, dtype=float)
    splits = combinatorial_purged_splits(len(rets), n_groups, k_test, embargo)
    sharpes = [
        sharpe_ratio(arr[test].tolist()) for _train, test in splits if test.size >= 2
    ]
    if not sharpes:
        return zero
    sh = np.asarray(sharpes, dtype=float)
    return {
        "n_paths": int(sh.size),
        "sharpe_mean": float(np.mean(sh)),
        "sharpe_std": float(np.std(sh, ddof=1)) if sh.size >= 2 else 0.0,
        "sharpe_median": float(np.median(sh)),
        "frac_positive": float(np.mean(sh > 0.0)),
    }


def weighting_comparison(
    sleeve_returns: dict[str, pd.Series],
    n_groups: int = 6,
    k_test: int = 2,
    embargo: int = 0,
    min_periods: int = 8,
) -> dict:
    """Compare HRP vs inverse-vol on the book's out-of-sample Sharpe (§1.2 KILL).

    Both books are CAUSAL (weights from past only); each CPCV test fold is an
    out-of-sample window, so the comparison is paired fold-by-fold over the same
    C(n_groups, k_test) paths. Per the §1.2 KILL, HRP is adopted ONLY if it beats
    plain inverse-vol on the book OOS Sharpe on **>=3 folds AND on the mean** — else
    inverse-vol is kept for parsimony (DeMiguel-Garlappi-Uppal: at low data the
    extra structure rarely pays). Returns the per-fold pairs and the verdict; the
    decision itself stays with the human merge gate.
    """
    iv = np.asarray(
        combine_book(sleeve_returns, "inv_vol", min_periods).tolist(), dtype=float
    )
    hrp = np.asarray(
        combine_book(sleeve_returns, "hrp", min_periods).tolist(), dtype=float
    )
    n = int(min(iv.size, hrp.size))
    per_fold: list[tuple[float, float]] = []
    for _train, test in combinatorial_purged_splits(n, n_groups, k_test, embargo):
        if test.size < 2:
            continue
        per_fold.append(
            (sharpe_ratio(iv[test].tolist()), sharpe_ratio(hrp[test].tolist()))
        )
    hrp_better = sum(1 for a, b in per_fold if b > a)
    iv_mean = float(np.mean([a for a, _ in per_fold])) if per_fold else 0.0
    hrp_mean = float(np.mean([b for _, b in per_fold])) if per_fold else 0.0
    return {
        "n_folds": len(per_fold),
        "hrp_better_folds": hrp_better,
        "inv_vol_mean_oos_sharpe": iv_mean,
        "hrp_mean_oos_sharpe": hrp_mean,
        "adopt_hrp": hrp_better >= 3 and hrp_mean > iv_mean,
        "per_fold": per_fold,
    }


def book_report(
    sleeve_returns: dict[str, pd.Series],
    weighting: str = "inv_vol",
    min_periods: int = 8,
    n_trials: int | None = None,
    n_blocks: int = 10,
) -> dict:
    """Honest BOOK verdict: combine the sleeves causally, then run the full gate on the
    book return series plus the per-sleeve accept criteria.

    Returns:
      * ``book``: n, sharpe, Deflated Sharpe (deflated by ``n_trials`` — the TRUE trial
        count from the TrialRegistry; falls back to the sleeve count, which over-states
        the DSR), PBO over the sleeve matrix, and the CPCV path distribution.
      * ``sleeves``: per-sleeve solo Sharpe, LOO-Δ-Sharpe contribution, and mark count.
      * ``orthogonality``: the rho matrix as a nested dict.
      * ``weighting`` / ``n_trials``: echo of the configuration.
    """
    ids = list(sleeve_returns)
    aligned = align_sleeves(sleeve_returns)
    book = combine_book(sleeve_returns, weighting, min_periods)
    book_list = book.tolist()

    # Deflate by the dispersion of solo Sharpes across sleeves (as report.py does), and
    # by the true trial count when supplied (else the sleeve count — an under-count).
    solo = {sid: sharpe_ratio(aligned[sid].tolist()) for sid in ids}
    sr_var = float(np.var(list(solo.values()), ddof=1)) if len(ids) >= 2 else 0.0
    effective_trials = int(n_trials) if n_trials else max(1, len(ids))

    loo = leave_one_out_delta_sharpe(sleeve_returns, weighting, min_periods)
    rho = orthogonality_matrix(sleeve_returns)
    pbo = (
        probability_backtest_overfitting(aligned.to_numpy().tolist(), n_splits=n_blocks)
        if len(ids) >= 2
        else 0.0
    )

    return {
        "book": {
            "n": len(book_list),
            "sharpe": sharpe_ratio(book_list),
            "dsr": deflated_sharpe_ratio(book_list, sr_var, effective_trials),
            "pbo": pbo,
            "cpcv": cpcv_book_distribution(book_list),
        },
        "sleeves": {
            sid: {
                "solo_sharpe": solo[sid],
                "loo_delta_sharpe": loo[sid],
                "n_marks": int((aligned[sid] != 0.0).sum()),
            }
            for sid in ids
        },
        "orthogonality": {
            a: {b: float(rho.loc[a, b]) for b in ids} for a in ids
        }
        if not rho.empty
        else {},
        "weighting": weighting,
        "n_trials": effective_trials,
    }
