"""Stream-0 data lake: a frozen, point-in-time Parquet lake (data/raw/<source>/)
queried through DuckDB views, plus the throttle / idempotency / provenance
machinery that makes the build resumable and reproducible.

The heavy lifting lives in `core` (source-agnostic infra) and `sources/`
(one client per feed). Acquisition only — no research, no backtests.
"""
