"""Meta-labeled fractional-Kelly sizing layer (Stream 1, vector 1.3).

The lab's confirmed sleeves resolve DIRECTION (e.g. S2 funding-squeeze is long-only,
armed at funding z>=2); their SIZE/TIMING has never been optimised. This layer adds
a *secondary* model — a meta-label (López de Prado) that predicts P(win) from each
event's decision-time features — and turns that probability into a bet fraction with
**fractional Kelly** (full Kelly is fragile; we take 0.20 of it). One feature is a
``news_blackout`` flag, so the model can LEARN to down-size around news (the lab's
news-discipline, from the $12k print loss) rather than enforcing a brittle hard veto.

The honesty discipline (the §1.3 KILL): meta-labeling cannot manufacture edge from a
zero-gross signal — it can only lift the **deflated** Sharpe of a real-but-marginal
edge by sizing high-conviction events up and low-conviction ones down. So it is
validated OUT-OF-SAMPLE, leave-one-out across instruments: train the meta-label on
the other instruments, size the HELD-OUT one, and require the sized stream's deflated
Sharpe (DSR — never raw) to beat the *un-sized* baseline on >=2/3 held-outs while
degrading none. An in-sample-only gain is size overfit and is rejected.

All functions are pure (the MetaLabeler fit is deterministic given its inputs); they
reuse the committed primitives ``risk.meta_labeling.MetaLabeler`` and
``learning.stats`` (DSR) — nothing here is reimplemented.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hermes.learning.stats import deflated_sharpe_ratio, sharpe_ratio
from hermes.risk.meta_labeling import MetaLabeler

__all__ = [
    "estimate_win_loss_ratio",
    "kelly_fraction_from_pwin",
    "loo_sizing_gate",
    "meta_sized_returns",
]


def kelly_fraction_from_pwin(
    p_win: float, win_loss_ratio: float, fraction: float = 0.20
) -> float:
    """Fractional-Kelly bet fraction from P(win) and the win/loss payoff ratio b.

    Full Kelly is ``f* = (p·b - q)/b`` with q = 1-p; we return ``fraction · max(0,
    f*)`` so a non-positive edge bets nothing and a real edge is sized at a fraction
    of full Kelly (drawdown control — full Kelly is famously fragile to estimation
    error in p and b).
    """
    b = float(win_loss_ratio)
    if b <= 0.0:
        return 0.0
    p = min(max(float(p_win), 0.0), 1.0)
    f_star = (p * b - (1.0 - p)) / b
    return float(fraction * max(0.0, f_star))


def estimate_win_loss_ratio(
    gross: Sequence[float], outcomes: Sequence[int]
) -> float:
    """Payoff ratio b = mean(win return) / |mean(loss return)| from realised events.

    Falls back to 1.0 (neutral payoff) when no losses are observed, so b is always a
    usable positive number for the Kelly formula.
    """
    g = np.asarray(list(gross), dtype=float)
    o = np.asarray(list(outcomes))
    wins = g[o == 1]
    losses = g[o == 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = abs(float(losses.mean())) if losses.size else 0.0
    if avg_loss <= 0.0:
        return 1.0
    return avg_win / avg_loss


def meta_sized_returns(
    train_features: Sequence[dict],
    train_outcomes: Sequence[int],
    train_gross: Sequence[float],
    test_features: Sequence[dict],
    test_gross: Sequence[float],
    fraction: float = 0.20,
    min_samples: int = 50,
) -> dict:
    """Size a held-out event stream by a meta-label fitted on the training events.

    Fits ``MetaLabeler`` on ``(train_features, train_outcomes)``; if it cannot fit
    (too few samples or a single class) the layer is a NO-OP — the sized stream is the
    flat un-sized baseline, an honest "sizing adds nothing yet". Otherwise each test
    event is sized by ``fractional-Kelly(P(win))`` using the payoff ratio estimated
    from the *training* events (never the test events — no leakage).

    Returns ``{fitted, baseline, sized, p_win, bet_fraction}``. Sharpe is scale-free,
    so what matters is how the bet fractions *re-weight* events relative to baseline.
    """
    labeler = MetaLabeler(min_samples=min_samples).fit(
        list(train_features), list(train_outcomes)
    )
    baseline = [float(x) for x in test_gross]
    if not labeler.is_fitted:
        return {
            "fitted": False,
            "baseline": baseline,
            "sized": list(baseline),
            "p_win": [None] * len(baseline),
            "bet_fraction": [1.0] * len(baseline),
        }

    b = estimate_win_loss_ratio(train_gross, train_outcomes)
    p_win = [labeler.predict_proba(f) for f in test_features]
    bet = [
        kelly_fraction_from_pwin(p if p is not None else 0.5, b, fraction)
        for p in p_win
    ]
    sized = [g * w for g, w in zip(baseline, bet)]
    return {
        "fitted": True,
        "baseline": baseline,
        "sized": sized,
        "p_win": p_win,
        "bet_fraction": bet,
    }


def loo_sizing_gate(
    instruments: dict[str, dict],
    fraction: float = 0.20,
    min_samples: int = 50,
    n_trials: int = 3,
    degradation_margin: float = 0.01,
) -> dict:
    """Leave-one-out held-out sizing gate (§1.3 KILL).

    ``instruments`` maps each name to ``{"features": [...], "outcomes": [...],
    "gross": [...]}``. For each held-out instrument the meta-label is trained on the
    OTHER instruments and used to size the held-out stream; the sized stream's
    DEFLATED Sharpe is compared to the un-sized baseline's. Deflation uses the
    cross-instrument solo-Sharpe dispersion as ``sr_variance`` (how much Sharpe varies
    by instrument luck) and ``n_trials`` as the documented model-evaluation count.

    Verdict (``passed``): the sizing layer is accepted iff every fold fitted, it
    IMPROVED the deflated Sharpe on >=2/3 held-outs, and it degraded NONE by more than
    ``degradation_margin``. Reports every fold (no cherry-pick). The decision itself
    stays with the human merge gate.
    """
    names = list(instruments)
    solo = [sharpe_ratio(instruments[k]["gross"]) for k in names]
    sr_var = float(np.var(solo, ddof=1)) if len(solo) >= 2 else 0.0

    per_fold: dict[str, dict] = {}
    for held in names:
        train = [k for k in names if k != held]
        tf = [f for k in train for f in instruments[k]["features"]]
        to = [o for k in train for o in instruments[k]["outcomes"]]
        tg = [g for k in train for g in instruments[k]["gross"]]
        h = instruments[held]
        res = meta_sized_returns(
            tf, to, tg, h["features"], h["gross"], fraction, min_samples
        )
        base_dsr = deflated_sharpe_ratio(res["baseline"], sr_var, n_trials)
        sized_dsr = deflated_sharpe_ratio(res["sized"], sr_var, n_trials)
        per_fold[held] = {
            "fitted": res["fitted"],
            "n_events": len(h["gross"]),
            "baseline_sharpe": sharpe_ratio(res["baseline"]),
            "sized_sharpe": sharpe_ratio(res["sized"]),
            "baseline_dsr": base_dsr,
            "sized_dsr": sized_dsr,
            "improved": bool(res["fitted"] and sized_dsr > base_dsr),
            "degraded": bool(
                res["fitted"] and sized_dsr < base_dsr - degradation_margin
            ),
        }

    improved = sum(1 for f in per_fold.values() if f["improved"])
    degraded = sum(1 for f in per_fold.values() if f["degraded"])
    all_fitted = all(f["fitted"] for f in per_fold.values())
    return {
        "n_folds": len(per_fold),
        "improved_folds": improved,
        "degraded_folds": degraded,
        "all_fitted": all_fitted,
        "passed": bool(all_fitted and improved >= 2 and degraded == 0),
        "sr_variance": sr_var,
        "n_trials": n_trials,
        "per_fold": per_fold,
    }
