"""Per-feed lake clients. Each module exposes ``run(raw_root, manifest, *, run_as_of)``
performing that source's idempotent pulls into ``data/raw/<source>/`` Parquet, plus
pure transform helpers (unit-tested offline). Acquisition only.
"""
