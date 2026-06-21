"""Probability of Backtest Overfitting (PBO) via Combinatorial Symmetric
Cross-Validation — Bailey, Borwein, López de Prado & Zhu (2017).

Given a T x N matrix of per-period returns (rows = time, cols = strategy configs),
split time into S blocks, and over every way of choosing S/2 blocks as in-sample:
pick the IS-best config, measure its out-of-sample rank. PBO is the fraction of
splits where that IS-best config lands in the bottom half OOS — i.e. how often
"the best backtest" is actually a fluke.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np


def probability_backtest_overfitting(
    returns_matrix: Sequence[Sequence[float]], n_splits: int = 10
) -> float:
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2:
        return 0.0
    t, n = m.shape
    if n < 2 or t < n_splits or n_splits < 2:
        return 0.0
    if n_splits % 2:
        n_splits -= 1  # need an even number of blocks for a symmetric split

    blocks = np.array_split(np.arange(t), n_splits)
    half = n_splits // 2
    overfit = 0
    total = 0
    for is_blocks in itertools.combinations(range(n_splits), half):
        is_set = set(is_blocks)
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_set])
        is_perf = m[is_rows].mean(axis=0)
        oos_perf = m[oos_rows].mean(axis=0)
        best = int(np.argmax(is_perf))
        rank = int(np.sum(oos_perf <= oos_perf[best]))  # 1 (worst) .. n (best)
        omega = min(max(rank / (n + 1), 1e-9), 1 - 1e-9)
        if np.log(omega / (1 - omega)) <= 0:  # OOS rank in the bottom half
            overfit += 1
        total += 1
    return float(overfit / total) if total else 0.0
