"""Source-agnostic data-lake infrastructure (Stream 0).

Five load-bearing guarantees, each a small, unit-tested unit:

* ``RateLimiter`` — strict min-interval throttle (CoinGlass Hobbyist = 30 req/min).
  ``monotonic``/``sleep`` are injected so the spacing is testable without waiting.
* ``covers_range`` — the point-in-time idempotency predicate ("is this pull
  already complete?"), so a re-run after a crash resumes instead of re-pulling.
* ``detect_gaps`` — holes in a timestamp grid, recorded in the manifest.
* ``Manifest`` / ``summarize_manifest`` / ``render_*_md`` — provenance trail and
  the human-readable DATA_MANIFEST / BLOCKERS reports.
* ``cached_pull`` — ties them together: skip-if-complete, fetch, fail-soft,
  write Parquet, log provenance.

Plus thin Parquet helpers and a DuckDB view builder (``build_views``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------- #
# rate limiting
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Strict min-interval spacing, one per source.

    ``max_per_min`` requests/minute -> ``min_interval = 60/max_per_min`` seconds
    enforced between consecutive ``wait()`` calls. The first ``wait()`` never
    sleeps; later calls sleep only for the shortfall since the previous wake.

    ``monotonic`` and ``sleep`` are injectable so tests drive a fake clock.
    """

    def __init__(
        self,
        max_per_min: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_min <= 0:
            raise ValueError("max_per_min must be > 0")
        self.min_interval = 60.0 / max_per_min
        self._monotonic = monotonic
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._monotonic()
        if self._last is not None:
            shortfall = self.min_interval - (now - self._last)
            if shortfall > 0:
                self._sleep(shortfall)
        # Re-read the clock after any sleep so spacing is measured from the
        # actual wake time, not the pre-sleep instant.
        self._last = self._monotonic()


# --------------------------------------------------------------------------- #
# idempotency + gap detection
# --------------------------------------------------------------------------- #
def covers_range(have_min: Any, have_max: Any, want_start: Any, want_end: Any) -> bool:
    """True iff existing [have_min, have_max] already contains [want_start, want_end].

    ``want_start``/``want_end`` may be None (unconstrained on that side). With no
    data (``have_min``/``have_max`` is None) nothing is covered.
    """
    if have_min is None or have_max is None:
        return False
    if want_start is not None and have_min > want_start:
        return False
    return not (want_end is not None and have_max < want_end)


def detect_gaps(
    timestamps: Sequence[Any] | Iterable[Any],
    interval: float,
    *,
    tolerance: float = 1.5,
) -> list[tuple[Any, Any]]:
    """Return (prev, next) pairs where consecutive timestamps are more than
    ``tolerance * interval`` apart (a missing bar). ``timestamps`` and
    ``interval`` must share units (e.g. epoch seconds, or ms)."""
    ts = sorted(timestamps)
    threshold = tolerance * interval
    gaps: list[tuple[Any, Any]] = []
    for prev, nxt in zip(ts, ts[1:]):
        if nxt - prev > threshold:
            gaps.append((prev, nxt))
    return gaps


# --------------------------------------------------------------------------- #
# Parquet helpers
# --------------------------------------------------------------------------- #
def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``df`` to Parquet, creating parent directories. No index column."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def frame_ts_bounds(df: pd.DataFrame, ts_col: str) -> tuple[Any, Any]:
    """(min, max) of ``ts_col``, or (None, None) when the frame is empty / lacks
    the column."""
    if df is None or ts_col not in df.columns or len(df) == 0:
        return (None, None)
    return (df[ts_col].min(), df[ts_col].max())


# --------------------------------------------------------------------------- #
# provenance / manifest
# --------------------------------------------------------------------------- #
_VALID_STATUS = ("ok", "skip", "partial", "blocked", "empty")


class Manifest:
    """Accumulates provenance entries + blockers for one frozen run."""

    def __init__(self, run_as_of: str) -> None:
        self.run_as_of = run_as_of
        self.entries: list[dict[str, Any]] = []
        self.blockers: list[dict[str, Any]] = []

    def log(
        self,
        *,
        source: str,
        endpoint: str,
        params: Any,
        status: str,
        rows: int | None = None,
        min_ts: Any = None,
        max_ts: Any = None,
        gaps: int | None = None,
        path: str | None = None,
        error: str | None = None,
        note: str | None = None,
    ) -> None:
        self.entries.append(
            {
                "source": source,
                "endpoint": endpoint,
                "params": params,
                "status": status,
                "rows": rows,
                "min_ts": min_ts,
                "max_ts": max_ts,
                "gaps": gaps,
                "path": path,
                "error": error,
                "note": note,
            }
        )

    def blocker(self, *, source: str, endpoint: str, params: Any, error: str, action: str) -> None:
        self.blockers.append(
            {"source": source, "endpoint": endpoint, "params": params, "error": error, "action": action}
        )
        self.log(source=source, endpoint=endpoint, params=params, status="blocked", error=error)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_as_of": self.run_as_of,
            "summary": summarize_manifest(self.entries),
            "entries": self.entries,
            "blockers": self.blockers,
        }


def summarize_manifest(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Counts per status + total rows across all entries."""
    out = {s: 0 for s in _VALID_STATUS}
    total_rows = 0
    for e in entries:
        status = e.get("status", "")
        if status in out:
            out[status] += 1
        total_rows += int(e.get("rows") or 0)
    out["total_rows"] = total_rows
    return out


_STATUS_EMOJI = {"ok": "✅", "skip": "⏭️", "partial": "⚠️", "blocked": "❌", "empty": "⚪"}


def render_manifest_md(manifest: Manifest) -> str:
    s = summarize_manifest(manifest.entries)
    lines = [
        "# DATA_MANIFEST — Hermès data lake (Stream 0)",
        "",
        f"- run_as_of: `{manifest.run_as_of}`",
        f"- datasets: {len(manifest.entries)}  "
        f"(✅ ok {s['ok']} · ⏭️ skip {s['skip']} · ⚠️ partial {s['partial']} · ❌ blocked {s['blocked']})",
        f"- total rows pulled: {s['total_rows']:,}",
        "",
        "| status | source | endpoint | params | rows | min_ts | max_ts | gaps | path |",
        "|---|---|---|---|---:|---|---|---:|---|",
    ]
    for e in manifest.entries:
        emoji = _STATUS_EMOJI.get(e.get("status", ""), e.get("status", ""))
        lines.append(
            f"| {emoji} | {e.get('source','')} | `{e.get('endpoint','')}` | "
            f"{_fmt_params(e.get('params'))} | {_fmt(e.get('rows'))} | {_fmt(e.get('min_ts'))} | "
            f"{_fmt(e.get('max_ts'))} | {_fmt(e.get('gaps'))} | {e.get('path') or ''} |"
        )
    return "\n".join(lines) + "\n"


def render_blockers_md(blockers: list[dict[str, Any]]) -> str:
    if not blockers:
        return "# BLOCKERS — Hermès data lake\n\n_None. All sources pulled or cleanly skipped._\n"
    lines = ["# BLOCKERS — Hermès data lake", "", f"{len(blockers)} item(s) need Malachie's action.", ""]
    for b in blockers:
        lines.append(f"- **{b.get('source','')}** / `{b.get('endpoint','')}` params={_fmt_params(b.get('params'))}")
        lines.append(f"  - erreur : `{b.get('error','')}`")
        lines.append(f"  - ACTION REQUISE DE MALACHIE : {b.get('action','')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(v: Any) -> str:
    return "" if v is None else str(v)


def _fmt_params(p: Any) -> str:
    if p is None:
        return ""
    if isinstance(p, dict):
        return ", ".join(f"{k}={v}" for k, v in p.items())
    return str(p)


# --------------------------------------------------------------------------- #
# the idempotent, fail-soft pull
# --------------------------------------------------------------------------- #
def cached_pull(
    target: str | Path,
    fetch_fn: Callable[[], pd.DataFrame | None],
    *,
    source: str,
    endpoint: str,
    params: Any,
    manifest: Manifest,
    expected_range: tuple[Any, Any] | None = None,
    ts_col: str | None = None,
    rate: RateLimiter | None = None,
) -> pd.DataFrame | None:
    """Pull ``target`` idempotently.

    * If ``target`` exists and (no ``expected_range`` given, or its ``ts_col``
      bounds already ``covers_range`` the wanted span) -> log *skip*, return it,
      never call ``fetch_fn``.
    * Otherwise throttle (``rate.wait()``), call ``fetch_fn``. Any exception or
      empty result -> record a blocker and return ``None`` (fail-soft, the rest
      of the run continues). A non-empty frame is written to Parquet and logged
      *ok* with row count + ts bounds + gap count.
    """
    target = Path(target)

    if target.exists():
        existing = read_parquet(target)
        covered = expected_range is None
        if not covered and ts_col is not None:
            bmin, bmax = frame_ts_bounds(existing, ts_col)
            covered = covers_range(bmin, bmax, expected_range[0], expected_range[1])
        if covered:
            manifest.log(
                source=source, endpoint=endpoint, params=params, status="skip",
                rows=len(existing), path=str(target),
            )
            return existing

    if rate is not None:
        rate.wait()

    try:
        df = fetch_fn()
    except Exception as e:  # fail-soft: one source's failure never aborts the run
        manifest.blocker(
            source=source, endpoint=endpoint, params=params, error=repr(e),
            action=f"investigate {source}/{endpoint} — request raised; see error above",
        )
        return None

    if df is None or len(df) == 0:
        manifest.blocker(
            source=source, endpoint=endpoint, params=params, error="empty response",
            action=f"confirm data availability for {source}/{endpoint} params={params} "
            f"(may be a not-listed pair / unsupported interval — research note, not always fixable)",
        )
        return None

    try:
        write_parquet(df, target)
    except Exception as e:  # fail-soft: a frame that won't serialize is a per-unit blocker
        try:
            if target.exists():
                target.unlink()  # never leave a corrupt partial Parquet behind
        except OSError:
            pass
        manifest.blocker(
            source=source, endpoint=endpoint, params=params, error=repr(e),
            action=f"{source}/{endpoint}: data fetched but failed to persist to Parquet "
            f"(likely a column dtype issue) — inspect the frame schema",
        )
        return None

    mn, mx = frame_ts_bounds(df, ts_col) if ts_col else (None, None)
    manifest.log(
        source=source, endpoint=endpoint, params=params, status="ok",
        rows=len(df), min_ts=mn, max_ts=mx, path=str(target),
    )
    return df


# --------------------------------------------------------------------------- #
# DuckDB view layer
# --------------------------------------------------------------------------- #
def discover_views(raw_root: str | Path) -> dict[str, str]:
    """Map the frozen Parquet tree to {view_name: glob}: every directory that
    directly contains ``*.parquet`` becomes one logical table, named by its path
    relative to ``raw_root`` with separators flattened to ``_``.

    e.g. raw/coinglass/lsr_top/*.parquet -> view ``coinglass_lsr_top``;
    raw/fred/{DFII10,DGS10}.parquet      -> view ``fred``.
    """
    root = Path(raw_root)
    views: dict[str, str] = {}
    for parent in sorted({p.parent for p in root.rglob("*.parquet")}):
        rel = parent.relative_to(root)
        name = "_".join(rel.parts) if rel.parts else "root"
        views[name] = str(parent / "*.parquet")
    return views


def build_views(duckdb_path: str | Path, mapping: dict[str, str]) -> None:
    """Create ``data/hermes_datalake.duckdb`` with one view per logical table.

    ``mapping`` is {view_name: parquet_glob}. Each view is
    ``SELECT * FROM read_parquet('<glob>', union_by_name=true)`` so a research
    stream can run plain SQL over the frozen Parquet tree.
    """
    import duckdb  # lazy: duckdb is a dev/lab-only dependency

    con = duckdb.connect(str(duckdb_path))
    try:
        for view, glob in mapping.items():
            # DuckDB cannot bind a prepared parameter inside CREATE VIEW; inline the
            # glob literal with single-quote escaping (paths are orchestrator-controlled).
            safe = str(glob).replace("'", "''")
            con.execute(
                f'CREATE OR REPLACE VIEW "{view}" AS '
                f"SELECT * FROM read_parquet('{safe}', union_by_name=true, filename=true)"
            )
    finally:
        con.close()


def write_manifest(manifest: Manifest, manifest_dir: str | Path) -> None:
    """Persist data_manifest.json + DATA_MANIFEST.md + BLOCKERS.md."""
    d = Path(manifest_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "data_manifest.json").write_text(json.dumps(manifest.to_json(), indent=2, default=str))
    (d / "DATA_MANIFEST.md").write_text(render_manifest_md(manifest))
    (d / "BLOCKERS.md").write_text(render_blockers_md(manifest.blockers))
