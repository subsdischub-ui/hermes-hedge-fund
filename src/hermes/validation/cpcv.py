"""Combinatorial Purged Cross-Validation (CPCV) + a persistent trial registry.

Thesis: a single train/test split is trivial to overfit on a few years of
crypto bars. CPCV (López de Prado, *Advances in Financial Machine Learning*,
ch. 7 & 12) instead partitions time into ``n_groups`` contiguous blocks and runs
*every* combination of ``k_test`` blocks as the test set — C(n_groups, k_test)
paths — purging train indices that overlap each test window and embargoing a few
indices right after it. The resulting Sharpe *distribution* (not a point) is the
engine's arbiter and is far harder to cherry-pick.

The TrialRegistry persists how many configurations have been backtested so the
Deflated Sharpe deflates by the *true* trial count (anti-overfit trap #1: under-
counting trials inflates the DSR).

Anti-look-ahead: splits are pure index bookkeeping — purge guarantees no test
index leaks into train, and the embargo removes the indices immediately *after*
each test block (the only ones whose labels could overlap forward into test).
No future bar is ever placed in a training fold. Parameters (n_groups=6,
k_test=2) are round, standard a-priori choices — no tuning.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from hermes.settings import get_settings


def combinatorial_purged_splits(
    n_obs: int,
    n_groups: int = 6,
    k_test: int = 2,
    embargo: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return all CPCV ``(train_idx, test_idx)`` paths over ``range(n_obs)``.

    Partition ``[0, n_obs)`` into ``n_groups`` contiguous blocks. For each
    combination of ``k_test`` blocks used as the test set, the remaining indices
    form the train set, then:

    * **purge** — drop from train any index that falls inside the test window
      (here, contiguous test blocks; disjointness is guaranteed);
    * **embargo** — additionally drop ``embargo`` indices immediately following
      each contiguous run of test blocks (clamped at the end of the series).

    Returns C(n_groups, k_test) paths. Indices are int64 numpy arrays.
    """
    if n_obs <= 0 or n_groups < 2 or k_test < 1 or k_test >= n_groups:
        return []

    all_idx = np.arange(n_obs, dtype=np.int64)
    blocks = np.array_split(all_idx, n_groups)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for combo in itertools.combinations(range(n_groups), k_test):
        test_block_set = set(combo)
        test_idx = np.concatenate([blocks[b] for b in combo])
        test_set = set(test_idx.tolist())

        # Purge: every non-test index starts as a train candidate.
        drop = set(test_set)

        # Embargo: remove `embargo` indices right after the END of each maximal
        # contiguous run of selected test blocks. Using the run boundary (rather
        # than every block) avoids embargoing inside the test window itself.
        if embargo > 0:
            for block_i in combo:
                # Only embargo after a block whose successor is NOT also a test
                # block — i.e. the trailing edge of a contiguous run.
                if (block_i + 1) not in test_block_set:
                    last = int(blocks[block_i][-1])
                    for off in range(1, embargo + 1):
                        nxt = last + off
                        if nxt < n_obs:
                            drop.add(nxt)

        train_idx = all_idx[~np.isin(all_idx, list(drop))]
        splits.append((train_idx, test_idx))

    return splits


def regime_aware_purged_splits(
    n_obs: int,
    regime_labels,
    n_groups: int = 6,
    k_test: int = 2,
    embargo: int = 0,
) -> dict:
    """Regime-aware CPCV (Stream 8, §8.3) — an OPTIONAL add that does **not** touch
    ``combinatorial_purged_splits``.

    Standard CPCV holds out contiguous *time* blocks, so a sleeve whose edge lives
    only in one regime can pass by averaging the good regime over the bad. Here we
    instead run CPCV *within each regime's index subset*: partition each regime's
    bars (in time order) into ``n_groups`` blocks and hold out every ``k_test``-combo,
    yielding a separate ``C(n_groups, k_test)`` Sharpe distribution per regime — so a
    regime-specific edge is *seen* as such rather than blended away.

    ``regime_labels`` is an array aligned to ``range(n_obs)``; non-finite labels
    (e.g. the engine's warmup burn-in) are excluded from every fold. As in
    ``combinatorial_purged_splits``, "purge" here means test indices never appear in
    train; the **label-horizon** protection is the caller's ``embargo`` (this function
    does not infer label spans), which drops train indices within ``embargo`` of any
    test index **on the global time axis** (the right generalisation when a regime's
    indices are non-contiguous in global time). For non-overlapping trade-level
    returns, ``embargo=0`` is already leak-free; pass the sleeve's hold horizon when
    marking overlapping bar-level labels.

    Returns ``{regime_label: [(train_idx, test_idx), ...]}``. A regime with fewer than
    ``n_groups`` bars cannot be split and maps to ``[]``.
    """
    if n_obs <= 0 or n_groups < 2 or k_test < 1 or k_test >= n_groups:
        return {}

    labels = np.asarray(regime_labels)
    out: dict = {}
    finite = np.isfinite(labels) if labels.dtype.kind == "f" else np.ones(n_obs, bool)
    distinct = sorted({labels[i].item() for i in range(n_obs) if finite[i]})

    for label in distinct:
        regime_idx = np.array(
            [i for i in range(n_obs) if finite[i] and labels[i].item() == label],
            dtype=np.int64,
        )
        if regime_idx.size < n_groups:
            out[label] = []
            continue

        blocks = np.array_split(regime_idx, n_groups)  # contiguous within the regime
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for combo in itertools.combinations(range(n_groups), k_test):
            test_idx = np.concatenate([blocks[b] for b in combo])
            test_set = set(test_idx.tolist())
            train_idx = np.array(
                [i for i in regime_idx.tolist() if i not in test_set], dtype=np.int64
            )
            if embargo > 0 and train_idx.size and test_idx.size:
                dist = np.abs(train_idx[:, None] - test_idx[None, :]).min(axis=1)
                train_idx = train_idx[dist > embargo]
            splits.append((np.sort(train_idx), np.sort(test_idx)))
        out[label] = splits

    return out


class TrialRegistry:
    """Persistent counter of backtested configurations (the DSR's true n_trials).

    State lives in ``<data_dir>/trials.json`` as ``{"count": <int>}``. Every
    distinct config evaluated in a backtest should ``bump()`` the registry so the
    Deflated Sharpe knows how many trials were really run.
    """

    _FILENAME = "trials.json"

    def __init__(self, data_dir: Path | str | None = None) -> None:
        if data_dir is None:
            data_dir = get_settings().data_dir
        self._dir = Path(data_dir)
        self._path = self._dir / self._FILENAME

    def count(self) -> int:
        """Current trial count (0 if no registry file exists yet)."""
        if not self._path.exists():
            return 0
        try:
            payload = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return 0
        return int(payload.get("count", 0))

    def bump(self, n: int = 1) -> int:
        """Add ``n`` to the count, persist, and return the new total."""
        new_count = self.count() + int(n)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"count": new_count}, indent=2))
        return new_count
