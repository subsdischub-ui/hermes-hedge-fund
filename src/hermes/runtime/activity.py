"""Persisted agent-activity feed.

The dashboard previously showed only CLOSED trades, so a selective agent that
correctly decides HOLD for hours looked dead. This bounded ring buffer records
EVERY tick/pulse decision (with its reason) to ``<data_dir>/activity.json`` so the
operator can see the agent is alive and *why* it did or didn't trade.

Plain JSON + atomic replace; tolerant of a corrupt/missing file (returns empty).
Entries are stamped with a UTC timestamp if the caller didn't provide one.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ActivityLog:
    def __init__(self, path: str | os.PathLike[str], cap: int = 200) -> None:
        self._path = Path(path)
        self._cap = int(cap)

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    def record(self, entry: dict[str, Any]) -> None:
        """Append an activity entry (stamping ``ts`` if absent) and keep the last
        ``cap`` entries. Newest is stored last."""
        e = dict(entry)
        e.setdefault("ts", datetime.now(UTC).isoformat())
        items = self._load()
        items.append(e)
        items = items[-self._cap :]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self._path.name + ".", dir=str(self._path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(items, f, default=str)
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            finally:
                raise

    def recent(self, n: int = 50) -> list[dict]:
        """Return the last ``n`` entries, NEWEST FIRST (for display)."""
        return list(reversed(self._load()[-int(n) :]))
