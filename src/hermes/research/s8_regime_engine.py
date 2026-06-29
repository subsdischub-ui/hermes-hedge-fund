"""Stream 8 — a causal market-regime engine (HMM + BOCPD + vol/macro buckets).

This is a **conditioning layer**, never a directional signal: it emits *states*
and *gates* that Streams 1/3/6 consume to condition their sleeves. The trivial
in-repo router (``hermes.data.regime.detect_regime`` — Kaufman efficiency ratio
+ a single threshold) is the baseline this is meant to surpass *without* falling
into the opposite trap: an HMM that overfits the history and re-segments on every
re-estimation is **worse** than the trivial ER.

The non-negotiable invariant is **causality / zero look-ahead**: every value the
engine reports for bar ``t`` is a function of bars ``<= t`` only.

* ``CausalGaussianHMM`` — a Gaussian-emission HMM trained by Baum-Welch on a
  *warmup* window, then run as a forward *filter*. The scaled forward variable is
  the filtered posterior ``P(s_t | x_1..t)`` — the backward pass is used **only**
  inside ``fit`` (on past data), never to emit a live state. States are
  canonicalised by ascending dispersion so "stress" always means the same thing
  across refits (the OOS-stability guarantee).
* ``online_bocpd`` — Adams & MacKay (2007) Bayesian Online Change-Point Detection
  with a constant hazard and a Normal-inverse-Gamma observation model. Natively
  online: the changepoint probability at ``t`` uses only ``x_1..t``.
* causal vol/macro buckets — trailing-percentile volatility terciles and lagged
  FRED macro gates, both point-in-time.
* ``regime_engine`` — the composite, returning one causal row per bar.

Hand-rolled on numpy/scipy on purpose: ``hmmlearn``/``ruptures`` fit/segment in
one batch over the *whole* series (the smoothed posterior uses future bars) —
exactly the look-ahead this module exists to avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from hermes.learning.stats import sharpe_ratio

__all__ = [
    "CausalGaussianHMM",
    "asof_labels",
    "lag_business_days",
    "macro_regime_gate",
    "online_bocpd",
    "realized_vol",
    "regime_engine",
    "shuffle_label_null",
    "state_stability",
    "trailing_vol_bucket",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


def _student_t_pdf(
    x: float, df: np.ndarray, loc: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Student-t pdf of scalar ``x`` evaluated under vector params (one per run-length)."""
    z = (x - loc) / scale
    log_pdf = (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
        - 0.5 * (df + 1.0) * np.log1p(z * z / df)
    )
    return np.exp(log_pdf)


def online_bocpd(
    x,
    hazard_lambda: float = 250.0,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    cp_k: int = 5,
    rmax: int | None = 1000,
) -> np.ndarray:
    """Bayesian Online Change-Point Detection (Adams & MacKay 2007).

    Maintains the run-length posterior ``P(r_t | x_1..t)`` under a constant hazard
    ``H = 1/hazard_lambda`` and a Normal-inverse-Gamma observation model (Student-t
    predictive). Returns ``cp_prob[t] = P(r_t <= cp_k | x_1..t)`` — the probability
    that the current run is at most ``cp_k`` bars old, i.e. that a changepoint
    occurred recently. Online by construction: each value uses only ``x_1..t``.

    (Why ``P(r<=k)`` and not ``P(r=0)``: at a break, both the growth and changepoint
    branches scale by the *same* predictive of the existing run, so ``P(r=0)`` is
    pinned near the hazard ``H`` and never spikes. The break instead shows as the
    run-length mass *collapsing* onto small ``r`` once the fresh-run hypothesis
    out-predicts the over-confident long runs — which ``P(r<=k)`` captures and
    ``P(r=0)`` misses. ``cp_k`` is a small round a-priori constant.)

    ``rmax`` caps the run-length support (drops the longest-run tail and renormalises)
    to keep the cost ``O(T·rmax)`` instead of ``O(T^2)``; ``None`` keeps every
    run-length. The default is generous enough that short series are unaffected.
    """
    x = np.asarray(x, dtype=float).ravel()
    T = x.size
    cp = np.zeros(T, dtype=float)
    if T == 0:
        return cp
    H = 1.0 / float(hazard_lambda)

    # Hyperparameters per run-length hypothesis; start with the prior at r=0.
    mu = np.array([mu0], dtype=float)
    kappa = np.array([kappa0], dtype=float)
    alpha = np.array([alpha0], dtype=float)
    beta = np.array([beta0], dtype=float)
    R = np.array([1.0], dtype=float)  # P(r_{t-1} | x_1..t-1)

    for t in range(T):
        xt = x[t]
        df = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        pred = _student_t_pdf(xt, df, mu, scale)

        growth = R * pred * (1.0 - H)  # r -> r+1
        cp0 = float((R * pred * H).sum())  # any r -> 0 (a changepoint)
        newR = np.empty(R.size + 1, dtype=float)
        newR[0] = cp0
        newR[1:] = growth
        total = newR.sum()
        newR = newR / total if total > 0 else np.full(newR.size, 1.0 / newR.size)
        cp[t] = float(newR[: cp_k + 1].sum())  # P(run length <= cp_k) = recently changed

        # Posterior update: index 0 = fresh prior, index r+1 = old r updated with xt.
        new_mu = np.empty_like(newR)
        new_kappa = np.empty_like(newR)
        new_alpha = np.empty_like(newR)
        new_beta = np.empty_like(newR)
        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = mu0, kappa0, alpha0, beta0
        new_kappa[1:] = kappa + 1.0
        new_mu[1:] = (kappa * mu + xt) / (kappa + 1.0)
        new_alpha[1:] = alpha + 0.5
        new_beta[1:] = beta + kappa * (xt - mu) ** 2 / (2.0 * (kappa + 1.0))

        mu, kappa, alpha, beta, R = new_mu, new_kappa, new_alpha, new_beta, newR

        if rmax is not None and R.size > rmax:
            mu, kappa, alpha, beta = mu[:rmax], kappa[:rmax], alpha[:rmax], beta[:rmax]
            R = R[:rmax]
            s = R.sum()
            R = R / s if s > 0 else np.full(rmax, 1.0 / rmax)

    return cp


class CausalGaussianHMM:
    """Diagonal-covariance Gaussian HMM with a *causal* forward filter.

    Trained by Baum-Welch (``fit``) from a deterministic, quantile-seeded start so
    a regime never moves merely because of RNG. After fit, states are reordered by
    ascending total emission variance: canonical state 0 = calmest, state K-1 =
    most dispersed ("stress"). ``filter`` runs the scaled forward recursion only —
    its output at ``t`` depends on ``x_1..t`` alone.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        tol: float = 1e-5,
        min_var: float = 1e-6,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.min_var = float(min_var)
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None  # (K, F)
        self.covars_: np.ndarray | None = None  # (K, F) diagonal variances

    # -- emissions ---------------------------------------------------------- #
    def _emission_log(self, X: np.ndarray) -> np.ndarray:
        """(T, K) log N(x_t; mu_k, diag(var_k))."""
        T = X.shape[0]
        logB = np.empty((T, self.n_states), dtype=float)
        for k in range(self.n_states):
            var = self.covars_[k]
            d = X - self.means_[k]
            logB[:, k] = -0.5 * (
                (_LOG_2PI + np.log(var)).sum() + ((d * d) / var).sum(axis=1)
            )
        return logB

    def _emission_scaled(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-timestep-rescaled emission likelihoods ``B`` (max 1 per row) and the
        per-timestep log offset that was removed (so the true log-likelihood can be
        recovered). The per-t constant cancels in every posterior/EM update."""
        logB = self._emission_log(X)
        offset = logB.max(axis=1)
        B = np.exp(logB - offset[:, None])
        return B, offset

    # -- scaled forward / backward ----------------------------------------- #
    def _forward(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scaled forward recursion. Returns (alpha, c) where ``alpha[t]`` is the
        filtered posterior ``P(s_t | x_1..t)`` and ``c`` the per-t normalisers."""
        T, K = B.shape
        alpha = np.empty((T, K), dtype=float)
        c = np.empty(T, dtype=float)
        a = self.startprob_ * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0] if c[0] > 0 else np.full(K, 1.0 / K)
        for t in range(1, T):
            a = (alpha[t - 1] @ self.transmat_) * B[t]
            s = a.sum()
            c[t] = s
            alpha[t] = a / s if s > 0 else np.full(K, 1.0 / K)
        return alpha, c

    def _backward(self, B: np.ndarray, c: np.ndarray) -> np.ndarray:
        T, K = B.shape
        beta = np.empty((T, K), dtype=float)
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.transmat_ @ (B[t + 1] * beta[t + 1])) / (c[t + 1] or 1.0)
        return beta

    # -- init --------------------------------------------------------------- #
    def _init_params(self, X: np.ndarray) -> None:
        K, F = self.n_states, X.shape[1]
        qs = (np.arange(K) + 0.5) / K
        self.means_ = np.quantile(X, qs, axis=0).reshape(K, F)
        gvar = np.maximum(X.var(axis=0, ddof=0), self.min_var)
        # Mild dispersion ladder so the calm/stress split has somewhere to converge
        # to even when the regimes differ in variance, not mean. Deterministic.
        ladder = np.linspace(0.5, 1.5, K).reshape(K, 1)
        self.covars_ = np.maximum(gvar[None, :] * ladder, self.min_var)
        self.startprob_ = np.full(K, 1.0 / K)
        stay = 0.90 if K > 1 else 1.0
        off = (1.0 - stay) / (K - 1) if K > 1 else 0.0
        self.transmat_ = np.full((K, K), off) + np.eye(K) * (stay - off)

    def _canonicalise(self) -> None:
        """Reorder states by ascending total dispersion: 0 = calm, K-1 = stress."""
        order = np.argsort(self.covars_.sum(axis=1))
        self.means_ = self.means_[order]
        self.covars_ = self.covars_[order]
        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[np.ix_(order, order)]

    # -- training ----------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> CausalGaussianHMM:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._init_params(X)
        K = self.n_states
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            B, offset = self._emission_scaled(X)
            alpha, c = self._forward(B)
            beta = self._backward(B, c)

            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma = np.divide(gamma, gamma_sum, where=gamma_sum > 0)

            # Expected transition counts (scaled forward-backward; the per-t scale
            # cancels because beta is divided by the same c).
            xi_sum = np.zeros((K, K), dtype=float)
            for t in range(X.shape[0] - 1):
                num = (
                    alpha[t][:, None]
                    * self.transmat_
                    * (B[t + 1] * beta[t + 1])[None, :]
                    / (c[t + 1] or 1.0)
                )
                xi_sum += num

            # M-step
            self.startprob_ = gamma[0] / gamma[0].sum()
            row = xi_sum.sum(axis=1, keepdims=True)
            self.transmat_ = np.divide(
                xi_sum, row, out=np.full_like(xi_sum, 1.0 / K), where=row > 0
            )
            for k in range(K):
                w = gamma[:, k]
                nk = w.sum()
                if nk <= 0:
                    continue
                self.means_[k] = (w[:, None] * X).sum(axis=0) / nk
                d = X - self.means_[k]
                self.covars_[k] = np.maximum(
                    (w[:, None] * d * d).sum(axis=0) / nk, self.min_var
                )

            ll = float(np.log(c).sum() + offset.sum())
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self._canonicalise()
        return self

    # -- causal inference --------------------------------------------------- #
    def filter(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Causal filtered posterior. Returns (states, probs) where ``probs[t]`` is
        ``P(s_t | x_1..t)`` and ``states[t] = argmax``. Forward-only: no future bar
        ever influences ``t``."""
        if self.means_ is None:
            raise RuntimeError("filter() called before fit()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        B, _ = self._emission_scaled(X)
        alpha, _ = self._forward(B)
        return alpha.argmax(axis=1), alpha


# --------------------------------------------------------------------------- #
# Causal volatility features + trailing-percentile buckets
# --------------------------------------------------------------------------- #
def realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Trailing realised volatility = rolling sample std of ``returns`` over the last
    ``window`` bars. Causal: bar ``t`` uses only ``returns[t-window+1..t]``; the first
    ``window-1`` bars are NaN (insufficient history)."""
    return returns.rolling(window, min_periods=window).std(ddof=1)


def trailing_vol_bucket(
    vol: pd.Series, window: int = 252, n_buckets: int = 3
) -> pd.Series:
    """Assign each bar to a volatility bucket by its **trailing percentile**.

    Bucket of bar ``t`` = which ``n_buckets``-quantile ``vol[t]`` falls into, ranked
    against the *trailing* window ``vol[t-window+1..t]`` only (point-in-time — no
    full-sample quantile, which would leak the future). Bars without a full window of
    valid observations are NaN. 0 = calmest tercile, ``n_buckets-1`` = most volatile.
    """
    v = vol.to_numpy(dtype=float)
    n = v.size
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        cur = v[t]
        if np.isnan(cur):
            continue
        lo = max(0, t - window + 1)
        w = v[lo : t + 1]
        w = w[~np.isnan(w)]
        if w.size < window:
            continue
        # Mid-rank percentile of the current value within its trailing window.
        pct = float((w < cur).mean() + 0.5 * (w == cur).mean())
        out[t] = float(min(n_buckets - 1, int(pct * n_buckets)))
    return pd.Series(out, index=vol.index)


# --------------------------------------------------------------------------- #
# regime_engine — the composite causal conditioning layer
# --------------------------------------------------------------------------- #
_LABELS_3 = {0: "range", 1: "trend", 2: "stress"}


def regime_engine(
    df: pd.DataFrame,
    price_col: str = "close",
    warmup: int = 750,
    n_states: int = 3,
    vol_window: int = 20,
    bucket_window: int = 252,
    hazard_lambda: float = 250.0,
    hmm_iter: int = 100,
) -> pd.DataFrame:
    """Causal regime features for one instrument: HMM state + probs, BOCPD changepoint
    probability, realised vol and its trailing-percentile bucket.

    Every output value for bar ``t`` depends only on bars ``<= t``:

    * HMM — a :class:`CausalGaussianHMM` is fit ONCE on the first ``warmup`` valid
      feature rows (features = [log-return, log realised-vol]), then run as a forward
      filter. States are emitted only from the ``warmup``-th valid bar onward (before
      that the training window is not strictly in the past — burn-in, left NaN).
    * BOCPD on the vol-standardised return ``r/rv`` (unit-scale, online).
    * ``rv`` trailing std; ``vol_bucket`` its trailing-percentile tercile.

    Returns a frame aligned to ``df.index`` with columns ``ts`` (if present), ``state``,
    ``regime_label`` (heuristic dispersion-tier alias for ``n_states==3``), ``p0..pK-1``,
    ``cp_prob``, ``rv``, ``vol_bucket``. NO directional column — this is a conditioner.
    """
    n = len(df)
    idx = df.index
    out = pd.DataFrame(index=idx)
    if "ts" in df.columns:
        out["ts"] = df["ts"].to_numpy()

    close = df[price_col].to_numpy(dtype=float)
    r = np.full(n, np.nan, dtype=float)
    if n >= 2:
        r[1:] = np.log(close[1:] / close[:-1])
    r_series = pd.Series(r, index=idx)
    rv = realized_vol(r_series, vol_window)
    rv_arr = rv.to_numpy()

    out["rv"] = rv_arr
    out["vol_bucket"] = trailing_vol_bucket(rv, bucket_window, n_buckets=3).to_numpy()

    # -- BOCPD on the vol-standardised return (causal: rv is trailing) ---------- #
    cp = np.full(n, np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = r / rv_arr
    z_valid = np.isfinite(z)
    if z_valid.any():
        start = int(np.argmax(z_valid))  # first finite standardised return
        cp[start:] = online_bocpd(z[start:], hazard_lambda=hazard_lambda)
    out["cp_prob"] = cp

    # -- HMM filtered state (fit on warmup, emit from warmup onward) ------------ #
    state = np.full(n, np.nan, dtype=float)
    probs = np.full((n, n_states), np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        logrv = np.log(rv_arr)
    feat = np.column_stack([r, logrv])
    valid = np.isfinite(feat).all(axis=1)
    valid_pos = np.flatnonzero(valid)
    if valid_pos.size >= warmup + 1:
        hmm = CausalGaussianHMM(n_states=n_states, n_iter=hmm_iter)
        hmm.fit(feat[valid_pos[:warmup]])
        filt_states, filt_probs = hmm.filter(feat[valid_pos])
        armed = valid_pos[warmup:]  # strictly-past training window for these bars
        state[armed] = filt_states[warmup:].astype(float)
        probs[armed] = filt_probs[warmup:]

    out["state"] = state
    if n_states == 3:
        out["regime_label"] = pd.Series(state, index=idx).map(
            lambda s: _LABELS_3.get(int(s)) if np.isfinite(s) else None
        )
    for k in range(n_states):
        out[f"p{k}"] = probs[:, k]

    return out


# --------------------------------------------------------------------------- #
# Conditioning validation — causal attach, shuffle-label null, OOS stability
# --------------------------------------------------------------------------- #
def asof_labels(events, label_ts, label_values) -> np.ndarray:
    """Causal as-of join: for each event time, the most recent label whose timestamp
    is ``<= event`` (NaN if the event precedes all labels). Used to stamp each trade
    with the regime/gate that was *knowable at the entry bar* — never a future label.
    ``events`` and ``label_ts`` must share units."""
    events = np.asarray(events, dtype=float)
    lt = np.asarray(label_ts, dtype=float)
    lv = np.asarray(label_values, dtype=float)
    order = np.argsort(lt, kind="stable")
    lt, lv = lt[order], lv[order]
    pos = np.searchsorted(lt, events, side="right") - 1
    out = np.full(events.shape, np.nan, dtype=float)
    ok = pos >= 0
    out[ok] = lv[pos[ok]]
    return out


def _metric_fn(metric):
    if callable(metric):
        return metric
    if metric == "mean":
        return lambda r: float(np.mean(r)) if len(r) else 0.0
    if metric == "sharpe":
        return lambda r: sharpe_ratio(list(r))
    raise ValueError(f"unknown metric {metric!r} (use 'mean', 'sharpe' or a callable)")


def _keep_mask(labels: np.ndarray, keep) -> np.ndarray:
    if callable(keep):
        return np.array([bool(keep(x)) for x in labels], dtype=bool)
    keepset = set(keep) if hasattr(keep, "__iter__") else {keep}
    return np.array([x in keepset for x in labels], dtype=bool)


def _gate_contrast(rets: np.ndarray, mask: np.ndarray, mfn) -> float:
    """Interaction effect: metric(gated trades) minus metric(the rest). > 0 means the gate
    selects better trades than it rejects."""
    g, rest = rets[mask], rets[~mask]
    if g.size == 0 or rest.size == 0:
        return 0.0
    return float(mfn(g) - mfn(rest))


def shuffle_label_null(
    rets,
    labels,
    keep,
    metric: str = "mean",
    n_perm: int = 2000,
    seed: int = 0,
    alternative: str = "greater",
) -> dict:
    """Shuffle-label null for a regime/macro **conditioner** (Stream 8, §8.1/§8.2 KILL).

    A regime is only useful if it *improves a confirmed sleeve in interaction*. The
    statistic is the gate contrast ``metric(trades where label in keep) minus metric(the
    rest)``. The null permutes the labels across trades ``n_perm`` times (preserving
    the gated-subset size) and recomputes the contrast — i.e. "is this gate better
    than a random subset of the same size?". ``p_value`` uses add-one smoothing so it
    is never exactly 0.

    A conditioner that does **not** beat this null carries no conditioning information
    and must be killed (this is exactly how the lab's prior conditioning attempts —
    real-rate→gold gated on COT, etc. — were honestly refuted at p≈0.42).
    """
    rets = np.asarray(rets, dtype=float)
    labels = np.asarray(labels)
    mfn = _metric_fn(metric)
    mask = _keep_mask(labels, keep)
    observed = _gate_contrast(rets, mask, mfn)

    rng = np.random.default_rng(seed)
    m = rets.size
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = _gate_contrast(rets, mask[rng.permutation(m)], mfn)

    if alternative == "greater":
        hits = int(np.sum(null >= observed))
    elif alternative == "less":
        hits = int(np.sum(null <= observed))
    else:
        hits = int(np.sum(np.abs(null) >= abs(observed)))
    p = (1 + hits) / (n_perm + 1)

    return {
        "observed": float(observed),
        "p_value": float(p),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if n_perm > 1 else 0.0,
        "n_kept": int(mask.sum()),
        "n": int(m),
    }


def state_stability(
    X,
    frac_a: float = 0.5,
    frac_b: float = 0.7,
    n_states: int = 3,
    hmm_iter: int = 100,
    n_null: int = 200,
    seed: int = 0,
) -> dict:
    """OOS state stability (§8.1 KILL): do the filtered states agree when the HMM is
    re-estimated on a longer window?

    Fit on ``X[:a]`` and on ``X[:b]`` (a<b), then compare their filtered states on the
    held-out tail ``X[b:]`` (out-of-sample for *both* fits). The raw best-match
    ``agreement`` fraction is NOT a trustworthy headline: a degenerate fit that
    collapses to one state scores 1.0, and on structureless noise raw agreement is
    high purely from marginal imbalance. So the headline is ``agreement_excess`` =
    raw agreement minus a **shuffle-pairing chance baseline** (permute one tail's
    labels, destroying temporal alignment but keeping marginals, and re-best-match).
    Excess ~0 means the apparent stability is chance (re-segmentation or collapse);
    a large positive excess means the same bars genuinely keep the same regime across
    re-estimation. ``p_value`` is one-sided vs that null.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(X)
    a, b = int(n * frac_a), int(n * frac_b)
    zero = {"agreement": 0.0, "agreement_excess": 0.0, "chance_baseline": 0.0,
            "marginal_floor": 0.0, "p_value": 1.0, "n_oos": 0, "split_a": a, "split_b": b}
    if a < n_states + 1 or b <= a or b >= n:
        return zero
    sa, _ = CausalGaussianHMM(n_states=n_states, n_iter=hmm_iter).fit(X[:a]).filter(X)
    sb, _ = CausalGaussianHMM(n_states=n_states, n_iter=hmm_iter).fit(X[:b]).filter(X)
    sa_t, sb_t = sa[b:], sb[b:]
    raw = _best_match_agreement(sa_t, sb_t, n_states)

    rng = np.random.default_rng(seed)
    null = np.array(
        [_best_match_agreement(sa_t, rng.permutation(sb_t), n_states) for _ in range(n_null)]
    )
    chance = float(null.mean())
    counts = np.bincount(sa_t, minlength=n_states)
    floor = float(counts.max() / counts.sum()) if counts.sum() else 0.0
    p = (1 + int(np.sum(null >= raw))) / (n_null + 1)
    return {
        "agreement": raw,
        "agreement_excess": float(raw - chance),
        "chance_baseline": chance,
        "marginal_floor": floor,
        "p_value": float(p),
        "n_oos": int(n - b),
        "split_a": a,
        "split_b": b,
    }


def _best_match_agreement(sa: np.ndarray, sb: np.ndarray, k: int) -> float:
    """Agreement between two label sequences maximised over all relabelings of ``sb``.

    Two fits that produce the SAME segmentation but with swapped state indices are
    perfectly stable (no re-segmentation) — only a genuine re-grouping of bars should
    lower the score. For small ``k`` (regimes are few) every permutation is cheap.
    """
    from itertools import permutations

    if sa.size == 0:
        return 0.0
    best = 0.0
    for perm in permutations(range(k)):
        mapped = np.array(perm, dtype=int)[sb]
        best = max(best, float((sa == mapped).mean()))
    return best


# --------------------------------------------------------------------------- #
# Macro regime gates (8.2) — point-in-time (business-day lag) conditioners
# --------------------------------------------------------------------------- #
def lag_business_days(s: pd.Series, n: int = 1) -> pd.Series:
    """Shift a date-indexed series forward by ``n`` business days: a value DATED ``T``
    is only AVAILABLE at ``T+n`` business days (FRED's ~+1 business-day publication
    latency — the point-in-time discipline from the gate §3.1)."""
    return pd.Series(s.to_numpy(), index=s.index + pd.tseries.offsets.BusinessDay(n))


def macro_regime_gate(
    value_by_date: pd.Series,
    lag_busdays: int = 1,
    kind: str = "trailing_bucket",
    window: int = 252,
    n_buckets: int = 3,
) -> pd.Series:
    """A macro **conditioner** (NOT a signal) from a FRED series (§8.2).

    * ``kind="trailing_bucket"`` — trailing-percentile tercile of the (lagged) level,
      e.g. HY-spread wide/tight (risk-off/on). Causal: trailing window only.
    * ``kind="sign"`` — 1 where the (lagged) level is negative, e.g. 2s10s inverted
      (the carry-vs-trend switch).

    Always business-day lagged first so the gate is point-in-time. Returns a date-
    indexed Series of gate states (NaN during warmup). Tested only in interaction via
    :func:`shuffle_label_null`, never standalone (standalone = re-discovering macro beta).
    """
    s = lag_business_days(value_by_date, lag_busdays) if lag_busdays else value_by_date
    s = s.sort_index()
    if kind == "trailing_bucket":
        return trailing_vol_bucket(s, window=window, n_buckets=n_buckets)
    if kind == "sign":
        out = (s < 0).astype(float)
        out[s.isna()] = np.nan
        return out
    raise ValueError(f"unknown kind {kind!r} (use 'trailing_bucket' or 'sign')")
