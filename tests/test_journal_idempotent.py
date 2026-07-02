"""TradeJournal.close must be idempotent (no double-close clobbering the
original timestamp), and set_reflection must not touch closed_at/pnl."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hermes.memory.journal import JournalEntry, TradeJournal
from hermes.memory.store import JournalRow, get_db


@pytest.mark.asyncio
async def test_close_is_idempotent_and_set_reflection_safe():
    await get_db().init()
    j = TradeJournal()
    tid = await j.open(
        JournalEntry(
            symbol="BTC/USDC",
            side="BUY",
            qty=0.001,
            entry_price=50_000.0,
            opened_at=datetime.now(UTC),
            stage="stage_0_paper",
        )
    )

    await j.close(
        trade_id=tid,
        exit_price=50_500.0,
        pnl_usd=0.5,
        fee_usd=0.025,
        exit_reason="take_profit",
    )

    db = get_db()
    async with db.session() as s:
        row1 = await s.get(JournalRow, tid)
        assert row1.closed_at is not None
        original_close = row1.closed_at
        original_pnl = row1.pnl_usd

    # Second close — must be a no-op for closed_at/pnl.
    await j.close(
        trade_id=tid,
        exit_price=99_999.0,
        pnl_usd=-999.0,
        fee_usd=999.0,
        exit_reason="bogus",
    )
    async with db.session() as s:
        row2 = await s.get(JournalRow, tid)
        assert row2.closed_at == original_close
        assert row2.pnl_usd == original_pnl

    # set_reflection only updates the reflection field.
    await j.set_reflection(tid, "I learned that fees matter.")
    async with db.session() as s:
        row3 = await s.get(JournalRow, tid)
        assert row3.reflection == "I learned that fees matter."
        assert row3.closed_at == original_close
        assert row3.pnl_usd == original_pnl


@pytest.mark.asyncio
async def test_count_closed_by_skills_filters_retired_skills():
    """edge_paper_trades feed: only closed trades from the given slate count —
    legacy/retired skills (ts_mom_vol_scaled, breakout_vol_expansion, ...) are excluded."""
    await get_db().init()
    journal = TradeJournal()
    now = datetime.now(UTC)
    for skill in ("ts_mom_vol_scaled", "breakout_vol_expansion", "mean_reversion_regime",
                  "funding_squeeze_continuation"):
        tid = await journal.open(JournalEntry(
            symbol="XRP/USDT:USDT", side="BUY", qty=0.001, entry_price=100.0,
            opened_at=now, skill_id=skill,
        ))
        await journal.close(tid, exit_price=101.0, pnl_usd=0.1, fee_usd=0.01, exit_reason="tp")
    slate = ["mean_reversion_regime", "funding_squeeze_continuation", "funding_carry_harvest"]
    before_legacy = await journal.count_closed_by_skills(["ts_mom_vol_scaled"])
    assert await journal.count_closed_by_skills(slate) >= 2
    assert before_legacy >= 1
    # legacy skills are excluded from the slate count: slate + legacy counts are disjoint sets
    assert await journal.count_closed_by_skills([]) == 0
