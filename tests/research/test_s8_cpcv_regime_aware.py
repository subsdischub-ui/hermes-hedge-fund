"""Stream 8 — regime-aware CPCV extension (8.3).

The standard CPCV (``combinatorial_purged_splits``) holds out contiguous *time*
blocks. A sleeve whose edge lives only in one regime can pass it by averaging the
good regime over the bad. The regime-aware extension instead runs CPCV *within
each regime's index subset*, so a regime-specific edge becomes visible as a
per-regime Sharpe distribution. The existing API must be untouched (an optional
add, not an edit) — hence a dedicated function and a guard test for the original.
"""

from __future__ import annotations

from math import comb

import numpy as np

from hermes.validation.cpcv import (
    combinatorial_purged_splits,
    regime_aware_purged_splits,
)


def test_original_cpcv_api_is_unchanged():
    # The regime-aware add must not alter the existing primitive.
    splits = combinatorial_purged_splits(n_obs=60, n_groups=6, k_test=2)
    assert len(splits) == comb(6, 2) == 15
    train, test = splits[0]
    assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)


def test_regime_aware_returns_one_split_set_per_regime():
    labels = np.array([0, 1] * 60)  # 60 of each, interleaved
    out = regime_aware_purged_splits(120, labels, n_groups=6, k_test=2)
    assert set(out.keys()) == {0, 1}
    for _regime, splits in out.items():
        assert len(splits) == comb(6, 2) == 15


def test_regime_aware_test_folds_are_pure_within_regime():
    labels = np.array([0] * 60 + [1] * 60)
    out = regime_aware_purged_splits(120, labels, n_groups=6, k_test=2)
    for regime, splits in out.items():
        regime_idx = set(np.flatnonzero(labels == regime).tolist())
        for _train, test in splits:
            assert set(test.tolist()).issubset(regime_idx)


def test_regime_aware_train_test_disjoint():
    labels = np.array([0, 1, 2] * 40)
    out = regime_aware_purged_splits(120, labels, n_groups=6, k_test=2)
    for splits in out.values():
        for train, test in splits:
            assert set(train.tolist()).isdisjoint(test.tolist())


def test_regime_aware_embargo_excludes_global_neighbors():
    labels = np.array([0, 1] * 60)
    out = regime_aware_purged_splits(120, labels, n_groups=6, k_test=2, embargo=2)
    for splits in out.values():
        for train, test in splits:
            if test.size == 0 or train.size == 0:
                continue
            # No train index within `embargo` of any test index on the GLOBAL axis.
            dist = np.abs(train[:, None] - test[None, :]).min(axis=1)
            assert dist.min() > 2


def test_regime_aware_test_folds_cover_each_regime_exactly_once():
    labels = np.array([0, 1] * 60)
    out = regime_aware_purged_splits(120, labels, n_groups=6, k_test=2)
    for regime, splits in out.items():
        regime_idx = set(np.flatnonzero(labels == regime).tolist())
        union = set()
        for _train, test in splits:
            union |= set(test.tolist())
        assert union == regime_idx


def test_regime_aware_skips_regimes_too_small_to_split():
    # Regime 2 has only 3 members < n_groups -> it yields no splits, the rest do.
    labels = np.array([0] * 50 + [1] * 50 + [2] * 3 + [0] * 17)
    out = regime_aware_purged_splits(len(labels), labels, n_groups=6, k_test=2)
    assert out.get(2, []) == []
    assert len(out[0]) == comb(6, 2)


def test_regime_aware_ignores_nan_labels():
    labels = np.array([0, 1, np.nan, 0, 1, np.nan] * 20, dtype=float)
    out = regime_aware_purged_splits(len(labels), labels, n_groups=6, k_test=2)
    # NaN-labelled bars never appear in any test fold.
    nan_idx = set(np.flatnonzero(~np.isfinite(labels)).tolist())
    for splits in out.values():
        for _train, test in splits:
            assert nan_idx.isdisjoint(test.tolist())
