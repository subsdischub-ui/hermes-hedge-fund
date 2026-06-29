"""FORWARD — in-sample robustness aggregators for the XS reversion gate.

These compute the IN-SAMPLE (contaminated) diagnostics that decide WIRE vs
DO-NOT-WIRE: annualised Sharpe of the period ledger, chronological subperiod
stability, and (in the runnable analysis) parameter-neighbourhood + LOO sweeps.
Only the metric primitives are unit-tested here; they must exclude open positions
and split strictly by time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.forward import xs_reversion_sanity as sanity

_DAY_MS = 86_400_000


def _book(rets, *, closed=True, start=1_700_000_000_000):
    n = len(rets)
    return pd.DataFrame(
        {
            "rebalance_ts": [start + i * 14 * _DAY_MS for i in range(n)],
            "ret_net": list(rets),
            "closed": [closed] * n,
        }
    )


def test_annualized_sharpe_equals_definition():
    rets = [0.03, -0.01, 0.02, -0.005, 0.015, 0.01]
    book = _book(rets)
    ppy = 365.25 / 14
    arr = np.array(rets)
    expected = arr.mean() / arr.std(ddof=1) * np.sqrt(ppy)
    assert np.isclose(sanity.annualized_sharpe(book, periods_per_year=ppy), expected)


def test_open_positions_excluded_from_sharpe():
    closed = _book([0.02, 0.02, 0.02, 0.02])
    opened = _book([99.0], closed=False)
    book = pd.concat([closed, opened], ignore_index=True)
    # the wild open MTM must not enter the Sharpe (constant closed rets → std 0 → 0.0)
    assert sanity.annualized_sharpe(book) == 0.0


def test_subperiod_sharpes_split_chronologically():
    rets = [0.05, 0.04, 0.03, -0.05, -0.04, -0.03, 0.01, 0.02, 0.03]
    book = _book(rets)
    parts = sanity.subperiod_sharpes(book, n_splits=3)
    assert len(parts) == 3
    # first third is all-positive → positive Sharpe; middle third all-negative → negative
    assert parts[0] > 0
    assert parts[1] < 0


def test_subperiod_handles_too_few_trades():
    book = _book([0.01, 0.02])
    parts = sanity.subperiod_sharpes(book, n_splits=3)
    assert len(parts) == 3  # padded with 0.0 where a split is too thin
