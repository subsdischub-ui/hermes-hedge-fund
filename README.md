# Hermès — autonomous AI crypto hedge-fund agent

> Local-first. Multi-agent (LangGraph-style). LLM brain + TRM verifier.
> Starts on $100, paper-only, escalates risk only as it proves itself.

## What it is

Hermès is a Python service that runs continuously and tries to compound a small
crypto bankroll (default $100) on Hyperliquid. It is **paper-first** — Stage 0
forbids touching real funds. Each stage advancement is gated on measurable
performance (Sharpe / Sortino / drawdown / live-trade count) and is **manual**:
the operator flips `HERMES_STAGE` after reading the gate report.

## Architecture

```
                        ┌──────────────┐
                        │ Tailscale    │  Mac M2 Max (Ollama)
                        │   tunnel     │   - qwen3-coder-next (primary brain)
                        │              │   - qwen3.6:27b      (secondary)
                        └──────┬───────┘
                               │ OpenAI-compatible HTTP (private tailnet)
                               ▼
Docker (Dokploy) ─── hermes-agent ─── FastAPI :8000
                       │
                       ├── graph.HermesOrchestrator
                       │     ├── analysts (market, news, sentiment)        (secondary LLM)
                       │     ├── researchers (bull / bear, parallel)       (primary LLM)
                       │     ├── trader (tool-calling, proposes a trade)   (primary LLM)
                       │     ├── risk_manager
                       │     │     ├── TRM verifier (refines/reject)
                       │     │     └── deterministic guardrails (Kelly+ATR, stage caps,
                       │     │         daily kill-switch, max-positions, min-notional)
                       │     └── portfolio_manager (executes if approved)
                       │
                       ├── execution: paper | hyperliquid | bybit (CCXT)
                       ├── memory: SQLite (journal, lessons, skill stats) + ChromaDB (semantic)
                       ├── skills: Voyager-style strategy library (mean-reversion, breakout, trend)
                       ├── learning:
                       │     - per-trade Reflexion             (on close)
                       │     - nightly lesson distillation     (cron)
                       │     - weekly skill re-rank + archive  (cron)
                       │     - monthly DSPy/GEPA train-set     (cron — exports JSONL for offline compile)
                       └── scheduler (APScheduler)
```

## Models (May 2026)

| Role | Default | Why |
|---|---|---|
| Primary brain | `qwen3-coder-next` (80B-A3B, 4-bit MLX via Ollama) | Format-invariant tool calling, 256K context, ~3B active = fast on M2 Max |
| Secondary | `qwen3.6:27b` dense | Apache 2.0, 262K context, strong classifier work |
| Escalation (opt) | `deepseek-v4-pro` API | Hard reasoning, cannot run locally on 96 GB |
| Verifier | Samsung-SAIL TRM (7M) | Recursive refinement of (state, action) → cheap second opinion that downgrades over-confident LLM proposals |

Swap any of the above via `LLM_PRIMARY_MODEL` / `LLM_SECONDARY_MODEL` env vars.

## Project layout

```
src/hermes/
├── settings.py        # pydantic-settings, loads config/default.yaml
├── llm/               # OpenAI-compatible router (primary/secondary/escalation) + tool schemas
├── trm/               # MarketStateEncoder, ActionCodec, TRMVerifier (+ Null fallback), train.py
├── data/              # CCXT market data, CryptoPanic news, alternative.me sentiment
├── execution/         # paper / hyperliquid / bybit + Exchange protocol
├── risk/              # PositionSizer (Kelly+ATR), StageManager, RiskEngine guardrails
├── memory/            # SQLite (journal, lessons, skill stats) + Chroma vector store
├── skills/            # mean-reversion, breakout, trend-follow (deterministic strategy templates)
├── learning/          # reflexion, nightly, weekly, monthly
├── graph/             # HermesOrchestrator, shared state
├── agents/            # market/news/sentiment analysts, bull/bear, trader, risk, portfolio_manager
├── runtime/           # stop_watcher
├── scheduler.py       # APScheduler wiring all cadences
├── api.py             # FastAPI surface
└── cli.py             # `hermes status | tick | stage | learn | serve`
```

## Quickstart (local dev, paper)

```bash
uv venv && uv sync --extra dev
cp .env.example .env                  # leave HERMES_STAGE=stage_0_paper
uv run hermes serve                   # -> http://localhost:8000
# or single tick:
uv run hermes tick BTC/USDC
```

## Risk model (the short version)

```
risk_pct = clamp(
    fractional_kelly * kelly * confidence,
    floor   = stage.floor_pct,        # 0.25%
    cap     = stage.per_trade_risk_cap_pct
)
position_size_usd = (equity * risk_pct) / (k * ATR(14) / price)
```

- `fractional_kelly = 0.20`, never full Kelly.
- Kelly only kicks in after ≥ 100 closed trades; cold-start uses a confidence-scaled floor.
- The TRM verifier's halt-confidence multiplicatively lowers the proposal's confidence.
- Hard kill-switch when realized PnL ≤ −`daily_loss_kill_pct` of equity.

## Maturity ladder

| Stage | Venue | Mode | Per-trade cap | Max open | Daily kill |
|---|---|---|---|---|---|
| 0 — paper | paper | spot | n/a | 3 | n/a |
| 1 — micro | Hyperliquid | spot | 0.5% | 3 | 2% |
| 2 — normal | Hyperliquid | spot | 2.0% | 5 | 4% |
| 3 — perps | Hyperliquid | perps 1x→2x | 2.0% | 5 | 5% |

Gates (see `config/default.yaml`) must be met before the operator may advance.

## Self-improvement loop

| Cadence | What it does |
|---|---|
| Per-trade close | Reflexion: 3-5 sentence reflection + tags, persisted on the journal row |
| Nightly 23:00 | LLM distills the day's reflections into 1-3 generalizable lessons → SQLite + Chroma |
| Weekly Sun 22:00 | Score each skill by 30-day Sharpe; archive negative-Sharpe skills |
| Monthly 1st 21:00 | Export `trainset_YYYYMMDD.jsonl` for offline DSPy/GEPA prompt compilation or LoRA |

## Deployment

See **DEPLOYMENT.md** for the full operator checklist.

## License

MIT (this repo). TRM source vendored from Samsung-SAIL is also MIT.
