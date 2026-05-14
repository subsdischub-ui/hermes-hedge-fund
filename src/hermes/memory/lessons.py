"""Lessons store — append-only knowledge distilled from trade reflections.

Write order: vector store FIRST, then SQL. If the vector write fails we
abort before committing the SQL row, so a SQL lesson never lacks a vector
counterpart. The reverse failure (SQL ok, vector down) would leave an
unsearchable lesson invisible to retrieval.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import desc, select

from hermes.logging_setup import get_logger
from hermes.memory.store import LessonRow, get_db
from hermes.memory.vector import get_vector_store

log = get_logger(__name__)


class LessonsStore:
    async def add(
        self,
        text: str,
        tags: list[str],
        source_trade_ids: list[int],
        score: float = 0.0,
    ) -> int | None:
        if not text.strip():
            return None
        db = get_db()
        async with db.session() as s:
            row = LessonRow(
                text=text,
                tags=",".join(tags),
                source_trade_ids=",".join(str(i) for i in source_trade_ids),
                score=score,
            )
            s.add(row)
            await s.flush()                   # populate row.id without committing
            lesson_id = row.id
            try:
                vs = get_vector_store()
                vs.add(
                    ids=[f"lesson:{lesson_id}"],
                    docs=[text],
                    metas=[
                        {
                            "tags": ",".join(tags),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                )
            except Exception as e:
                # Abort the transaction so we don't leave an orphan SQL row.
                await s.rollback()
                log.warning("lesson_vector_write_failed", error=str(e))
                return None
            await s.commit()
        return lesson_id

    async def search(self, query: str, k: int = 6) -> list[dict]:
        return get_vector_store().query(query, k=k)

    async def recent(self, n: int = 30) -> list[LessonRow]:
        db = get_db()
        async with db.session() as s:
            return (
                await s.execute(select(LessonRow).order_by(desc(LessonRow.created_at)).limit(n))
            ).scalars().all()
