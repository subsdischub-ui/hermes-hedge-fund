"""Forward OOS-accrual infra: unified PIT collection cron + size-0 shadow harness.

All feeds are read-only; all writes go to ``<data_dir>/forward/``. This package NEVER
touches the live trading runtime (market_analyst/pulse/config/allowed_skills) — that
wiring is the human merge gate.
"""
