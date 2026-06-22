# `.githooks/` — auto-documentation hook

Git hooks that maintain a concise, bounded narrative ("Histoire du code") of
this repo's evolution, appended to `ETAT_DES_LIEUX.md` after every commit.

## Activation (once per checkout — including the VPS)

Git does **not** auto-use a versioned hooks dir (for safety). Enable it with:

```bash
git config core.hooksPath .githooks
```

Add that line to your deploy/bootstrap so fresh clones pick it up.

## What it does

- `post-commit` — thin launcher; skips doc-only commits (loop guard), then runs
  the worker **detached** so a commit never blocks on the LLM call.
- `histoire-worker.sh` — summarises commits since the last recorded point into a
  ≤70-line French récit and injects it into a marker-delimited section of
  `ETAT_DES_LIEUX.md`. Old entries are merged into terse thematic summaries so
  the doc stays short but never loses the arc.
- `histoire-llm.py` — bridges the prompt through the project's own LLM router
  (`hermes.llm.router`), via `.venv`, so it reuses Hermès's credentials/config.

## Requirements where commits happen

- The project `.venv` present (or a `python3` that can import `hermes`).
- The router's LLM target reachable (local llama.cpp on the Mac tailnet, or a
  DeepSeek key). If unreachable, the hook logs to `.git/histoire.log` and skips
  the update — it never breaks the commit.

Runtime state (`histoire.log`, `histoire.lock`, `histoire.lastref`,
`hermes-histoire-du-code.md`) lives in `.git/` and is intentionally not tracked.
