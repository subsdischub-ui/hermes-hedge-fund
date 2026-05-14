"""Lessons store — append-only knowledge distilled from trade reflections."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import desc, select

from hermes.memory.store import LessonRow, get_db
from hermes.memory.vector import get_vector_store


class LessonsStore:
    async def add(self, text: str, tags: list[str], source_trade_ids: list[int], score: float = 0.0) -> int:
        db = get_db()
        async with db.session() as s:
            row = LessonRow(
                text=text,
                tags=",".join(tags),
                source_trade_ids=",".join(str(i) for i in source_trade_ids),
                score=score,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
        vs = get_vector_store()
        vs.add(
            ids=[f"lesson:{row.id}"],
            docs=[text],
            metas=[{"tags": ",".join(tags), "created_at": datetime.now(timezone.utc).isoformat()}],
        )
        return row.id

    async def search(self, query: str, k: int = 6) -> list[dict]:
        return get_vector_store().query(query, k=k)

    async def recent(self, n: int = 30) -> list[LessonRow]:
        db = get_db()
        async with db.session() as s:
            return (
                await s.execute(select(LessonRow).order_by(desc(LessonRow.created_at)).limit(n))
            ).scalars().all()
