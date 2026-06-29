"""Shared append-only PIT log writer for every forward flux + shadow log.

A *forward log* is a parquet that accrues one snapshot per (UTC-day, key). It is the
opposite of the frozen lake: it can only grow forward in time and is never
backfilled. Two invariants make it trustworthy as out-of-sample evidence:

* **point-in-time** — a row is stamped with the ``ts`` at which it was collected and
  is never rewritten with hindsight (the caller must pass only data available at
  collection time);
* **idempotent** — re-running the cron on the same UTC day overwrites that day's row
  per key instead of appending a duplicate, so a cron that double-fires is harmless.

Idempotency is delegated to ``hermes.research.s4_options.dedup_forward_log`` (already
proven + tested for the S4 skew log) so all forward logs share one dedup semantics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hermes.research.s4_options import dedup_forward_log
from hermes.settings import get_settings

_LOG_FILENAME = "log.parquet"


def _base_dir(data_dir: Path | str | None) -> Path:
    """Resolve the data root: explicit arg wins, else ``settings.data_dir``."""
    if data_dir is not None:
        return Path(data_dir)
    return Path(get_settings().data_dir)


def forward_dir(name: str, *, data_dir: Path | str | None = None) -> Path:
    """Return (creating) ``<data_dir>/forward/<name>`` for a flux/sleeve ``name``."""
    d = _base_dir(data_dir) / "forward" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log_path(name: str, *, data_dir: Path | str | None = None) -> Path:
    return forward_dir(name, data_dir=data_dir) / _LOG_FILENAME


def read_log(name: str, *, data_dir: Path | str | None = None) -> pd.DataFrame:
    """Read the accrued forward log for ``name`` — empty frame if never written."""
    path = _log_path(name, data_dir=data_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def append_snapshot(
    name: str,
    rows: pd.DataFrame,
    *,
    key_cols: tuple[str, ...],
    ts_col: str = "ts",
    data_dir: Path | str | None = None,
) -> Path:
    """Idempotently append ``rows`` to the forward log for ``name``.

    At most one row survives per (UTC-day, ``key_cols``); a same-day re-run keeps the
    latest. ``rows`` must carry an epoch-ms ``ts_col``. Returns the log path.

    Hardening (per the adversarial audit of ``dedup_forward_log``): ``key_cols`` must
    be non-empty (else a whole day collapses to one row), ``ts_col`` must be numeric
    epoch-ms (a datetime column would silently mis-floor), and the new batch is sorted
    ascending by ts so the latest-ts row deterministically wins the per-(day,key) dedup
    even if a collector emits same-key rows out of order.
    """
    if not key_cols:
        raise ValueError("append_snapshot: key_cols must be non-empty (else a day collapses to one row)")
    if rows is None or len(rows) == 0:
        return _log_path(name, data_dir=data_dir)
    if not pd.api.types.is_numeric_dtype(rows[ts_col]):
        raise TypeError(
            f"append_snapshot: {ts_col!r} must be numeric epoch-ms, got {rows[ts_col].dtype}"
        )
    rows = rows.sort_values(ts_col)  # latest ts is last → wins keep='last' deterministically
    existing = read_log(name, data_dir=data_dir)
    merged = dedup_forward_log(
        existing if len(existing) else None, rows, ts_col=ts_col, key_cols=key_cols
    )
    path = _log_path(name, data_dir=data_dir)
    merged.to_parquet(path, index=False)
    return path
