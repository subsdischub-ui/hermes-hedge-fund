"""SQLite (via aiosqlite + SQLAlchemy 2.0 async)."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from hermes.settings import get_settings


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JournalRow(Base):
    __tablename__ = "trade_journal"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    skill_id: Mapped[str] = mapped_column(String(64), index=True, default="")
    thesis: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    qty: Mapped[float] = mapped_column(Float)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    fee_usd: Mapped[float] = mapped_column(Float, default=0.0)
    exit_reason: Mapped[str] = mapped_column(String(32), default="")
    market_state: Mapped[dict] = mapped_column(JSON, default=dict)
    reflection: Mapped[str] = mapped_column(Text, default="")
    stage: Mapped[str] = mapped_column(String(32), default="")


class LessonRow(Base):
    __tablename__ = "lessons"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    text: Mapped[str] = mapped_column(Text)
    tags: Mapped[str] = mapped_column(String(256), default="")
    source_trade_ids: Mapped[str] = mapped_column(String(512), default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)


class SkillStatRow(Base):
    __tablename__ = "skill_stats"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    week_starting: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    skill_id: Mapped[str] = mapped_column(String(64), index=True)
    trades: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    dsr: Mapped[float] = mapped_column(Float, default=0.0)  # Deflated Sharpe (ranking/archival basis)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    archived: Mapped[bool] = mapped_column(default=False)


class StageEventRow(Base):
    __tablename__ = "stage_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    from_stage: Mapped[str] = mapped_column(String(32))
    to_stage: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    note: Mapped[str] = mapped_column(Text, default="")


class Database:
    def __init__(self, url: str):
        self.engine = create_async_engine(url, pool_pre_ping=True)
        self.session: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Lightweight additive migration: create_all never ALTERs existing
            # tables, so a DB created before a column was added (e.g. skill_stats.dsr
            # on an already-deployed volume) would miss it. Add any missing columns.
            existing = {
                row[1]
                for row in (
                    await conn.exec_driver_sql("PRAGMA table_info(skill_stats)")
                ).all()
            }
            if existing and "dsr" not in existing:
                await conn.exec_driver_sql(
                    "ALTER TABLE skill_stats ADD COLUMN dsr FLOAT DEFAULT 0.0"
                )

    async def dispose(self) -> None:
        await self.engine.dispose()


@lru_cache(maxsize=1)
def get_db() -> Database:
    return Database(get_settings().database_url)
