"""Combinatorial Purged CV (CPCV) splits + a persistent trial registry.

CPCV (López de Prado, *Advances in Financial Machine Learning*, ch. 7 & 12) is
the arbiter of the engine: partition time into ``n_groups`` contiguous blocks,
put every combination of ``k_test`` blocks in the test set, the remainder in
train, then *purge* train indices that overlap the test window and *embargo* a
few indices right after each test block. This yields C(n_groups, k_test) paths
whose Sharpe distribution is far harder to overfit than a single split.

The TrialRegistry persists the count of backtested configurations so the
Deflated Sharpe uses the *true* number of trials (anti-overfit trap #1).

All expectations below are hand-derived.
"""

from __future__ import annotations

import json
from math import comb

import numpy as np

from hermes.validation.cpcv import TrialRegistry, combinatorial_purged_splits


def _as_sets(splits):
    return [(set(map(int, tr)), set(map(int, te))) for tr, te in splits]


def test_number_of_paths_equals_n_choose_k():
    splits = combinatorial_purged_splits(n_obs=60, n_groups=6, k_test=2)
    assert len(splits) == comb(6, 2) == 15


def test_number_of_paths_other_combination():
    splits = combinatorial_purged_splits(n_obs=50, n_groups=5, k_test=2)
    assert len(splits) == comb(5, 2) == 10


def test_train_and_test_are_disjoint_and_test_blocks_contiguous():
    # 12 obs, 6 groups -> blocks of 2: [0,1][2,3][4,5][6,7][8,9][10,11].
    splits = _as_sets(
        combinatorial_purged_splits(n_obs=12, n_groups=6, k_test=2, embargo=0)
    )
    for train, test in splits:
        assert train.isdisjoint(test)  # purge guarantees no overlap


def test_purge_removes_test_indices_from_train():
    # No embargo: train = all blocks not chosen as test. With contiguous blocks
    # and no label overlap modelled here, purge simply means test ∉ train and
    # every non-test index is in train.
    n_obs, n_groups = 12, 6
    splits = _as_sets(
        combinatorial_purged_splits(n_obs=n_obs, n_groups=n_groups, k_test=2, embargo=0)
    )
    # First path: test blocks {0,1} -> test indices {0,1,2,3}.
    train, test = splits[0]
    assert test == {0, 1, 2, 3}
    assert train == set(range(4, 12))  # everything after, nothing purged extra
    assert test.isdisjoint(train)


def test_embargo_excludes_k_indices_after_each_test_block():
    # 12 obs, 6 groups of 2. embargo=1 removes the single index right after each
    # test block from train.
    n_obs, n_groups = 12, 6
    splits = _as_sets(
        combinatorial_purged_splits(n_obs=n_obs, n_groups=n_groups, k_test=2, embargo=1)
    )
    # Path with test blocks {1,3} -> test indices {2,3} and {6,7}.
    # array_split(12,6) = [0,1][2,3][4,5][6,7][8,9][10,11].
    # embargo=1 removes index 4 (after block1 ends at 3) and 8 (after block3 ends at 7).
    target = None
    for train, test in splits:
        if test == {2, 3, 6, 7}:
            target = (train, test)
            break
    assert target is not None, "expected the {1,3} test-block path"
    train, test = target
    assert 4 not in train  # embargoed after first test block
    assert 8 not in train  # embargoed after second test block
    # Remaining train = all - test - embargoed = {0,1,5,9,10,11}.
    assert train == {0, 1, 5, 9, 10, 11}


def test_embargo_excludes_two_indices_when_embargo_two():
    n_obs, n_groups = 12, 6
    splits = _as_sets(
        combinatorial_purged_splits(n_obs=n_obs, n_groups=n_groups, k_test=2, embargo=2)
    )
    # test blocks {0,1} -> test {0,1,2,3}; embargo=2 removes indices 4,5 after it.
    train, test = splits[0]
    assert test == {0, 1, 2, 3}
    assert 4 not in train and 5 not in train
    assert train == set(range(6, 12))


def test_embargo_does_not_run_past_end_of_series():
    # Last block as part of test -> embargo would point past n_obs; must clamp.
    n_obs, n_groups = 12, 6
    splits = _as_sets(
        combinatorial_purged_splits(n_obs=n_obs, n_groups=n_groups, k_test=2, embargo=2)
    )
    # test blocks {4,5} -> test indices {8,9,10,11}; nothing after 11 to embargo.
    target = next((s for s in splits if s[1] == {8, 9, 10, 11}), None)
    assert target is not None
    train, _test = target
    assert train == {0, 1, 2, 3, 4, 5, 6, 7}  # blocks 0..3, no out-of-range error


def test_indices_are_numpy_integer_arrays():
    splits = combinatorial_purged_splits(n_obs=12, n_groups=6, k_test=2)
    train, test = splits[0]
    assert isinstance(train, np.ndarray)
    assert isinstance(test, np.ndarray)
    assert np.issubdtype(train.dtype, np.integer)
    assert np.issubdtype(test.dtype, np.integer)


# --- TrialRegistry ---------------------------------------------------------


def test_trial_registry_starts_at_zero(tmp_path):
    reg = TrialRegistry(data_dir=tmp_path)
    assert reg.count() == 0


def test_trial_registry_bump_increments_and_persists(tmp_path):
    reg = TrialRegistry(data_dir=tmp_path)
    assert reg.bump() == 1
    assert reg.bump(3) == 4
    assert reg.count() == 4

    # A fresh instance reading the same dir sees the persisted count.
    reg2 = TrialRegistry(data_dir=tmp_path)
    assert reg2.count() == 4

    # The on-disk JSON is human-readable and holds the count.
    payload = json.loads((tmp_path / "trials.json").read_text())
    assert payload["count"] == 4


def test_trial_registry_defaults_to_settings_data_dir(tmp_path, monkeypatch):
    # When no data_dir is passed, it falls back to settings.data_dir.
    import hermes.validation.cpcv as cpcv_mod

    class _FakeSettings:
        data_dir = tmp_path

    monkeypatch.setattr(cpcv_mod, "get_settings", lambda: _FakeSettings())
    reg = TrialRegistry()
    reg.bump(2)
    assert reg.count() == 2
    assert (tmp_path / "trials.json").exists()
