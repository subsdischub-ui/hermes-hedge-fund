# `governance/` — the lab's canonical, SHARED governance state

This directory is the **single authoritative home** of Stream 10's mutable registry state. It is
**tracked in git** (committed on `round2/s10-gov`), so every worktree sees the *same* state through
version control — unlike the Round-1 location `research_streams/S10/`, which is gitignored
(`/research_*/`) and therefore held a *separate, divergent copy per worktree*. A per-worktree,
gitignored deflation count cannot be the lab's authoritative number; this dir fixes that.

## Files

| File | What it is | Source of truth |
|------|------------|-----------------|
| `trials_by_book.json` | per-book **deflation trial count** (the DSR's true `n_trials`) | **this file** — streams bump it and commit |
| `dead_ends_ledger.json` | falsified/blocked hypotheses (anti re-spend) | generated from `s10_governance.DEFAULT_DEAD_ENDS` |
| `acquisition_map.json` | data-$ governance (acquired / free / do-not-spend) | generated from `s10_governance.DEFAULT_ACQUISITION_MAP` |

The current canonical counts: **crypto 1891** (1869 + 22 from S2 v-next), **fx_metals 53**, **cross 5**.

## How to update (the only sanctioned path)

```python
from hermes.research.s10_governance import BookTrialRegistry, Book
reg = BookTrialRegistry()                 # no-arg → THIS dir (governance_registry_dir())
reg.bump(Book.CRYPTO, n=<configs_tried>, source="<stream>_<sweep>")
# then: git add governance/trials_by_book.json && commit
```

- The dead-ends ledger and acquisition map are **code-as-truth**: their canonical content lives in
  the embedded `DEFAULT_DEAD_ENDS` / `DEFAULT_ACQUISITION_MAP` constants (tracked in
  `src/hermes/research/s10_governance.py`). The `.json` here is the committed serialized snapshot,
  regenerated from those constants. `test_governance_snapshots_are_synced_with_embedded_defaults`
  fails if they ever drift — so a new dead end is added by editing the constant **and** regenerating:

  ```python
  from hermes.research.s10_governance import DeadEndsLedger, AcquisitionMap, governance_registry_dir
  d = governance_registry_dir()
  (d / "dead_ends_ledger.json").write_text(DeadEndsLedger().to_json())
  (d / "acquisition_map.json").write_text(AcquisitionMap().to_json())
  ```

## Override

Set `HERMES_GOVERNANCE_DIR` to point the registries elsewhere (used by tests via `tmp_path`). Tests
that *mutate* the registry MUST use a temp dir — never bump the committed file from a test.
