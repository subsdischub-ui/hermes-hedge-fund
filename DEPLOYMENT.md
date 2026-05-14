# Hermès — Deployment Checklist

Step-by-step. **Do them in order.** Each box is something you (the operator) must do.

---

## 0. Pre-flight (one-time)

- [ ] Create a Tailscale account, install Tailscale on **both** the MacBook M2 Max and the VPS that runs Dokploy.
- [ ] In Tailscale admin → Access Controls, apply `docker/tailscale-acl.example.json` (tags `tag:mac-llm`, `tag:hermes`; only `tag:hermes` can reach `tag:mac-llm:11434`).
- [ ] In Tailscale admin → **Settings → Keys**, generate an **ephemeral, reusable, tagged (`tag:hermes`) auth key**. Save it as `TS_AUTHKEY`.
- [ ] Install Dokploy on the VPS: `curl -sSL https://dokploy.com/install.sh | sh`.

## 1. MacBook M2 Max — local LLM (one-time, then `launchd` it)

- [ ] `brew install ollama tailscale`
- [ ] `ollama pull qwen3-coder-next`   ← primary brain (~40 GB at 4-bit)
- [ ] `ollama pull qwen3.6:27b`         ← secondary
- [ ] `chmod +x docker/start-mac-llm.sh && TS_HOSTNAME=hermes-mac ./docker/start-mac-llm.sh`
- [ ] Verify from another tailnet device: `curl http://hermes-mac:11434/v1/models`
- [ ] Persist with a launchd plist (optional but recommended) so it survives reboots. See `docker/start-mac-llm.sh` for the env vars.

## 2. TRM checkpoint (optional but recommended)

- [ ] On a GPU box (or M2 Max with MLX equivalent), run `python -m hermes.trm.train --in-dir <csv_dir>` to emit `/data/trm/dataset`.
- [ ] Train upstream: `cd /opt/trm && OMP_NUM_THREADS=8 python pretrain.py data_path=/data/trm/dataset arch=trm` (≈ a few hours on a 24 GB GPU).
- [ ] Copy the resulting `step_*` file to the Hermès `/data/trm/latest.pt` volume.
- [ ] If you skip this, `TRMVerifier` falls back to `NullTRMVerifier` automatically — the agent runs, just without TRM refinement.

## 3. GitHub repo + secrets

- [ ] Push this repo to GitHub (the included `.github/workflows/deploy.yml` will build + push the image to `ghcr.io`).
- [ ] In repo → Settings → Secrets and variables → Actions, add:
  - [ ] `DOKPLOY_URL`         (e.g. `https://dokploy.your-vps.com`)
  - [ ] `DOKPLOY_TOKEN`       (Dokploy → API → generate)
  - [ ] `DOKPLOY_COMPOSE_ID`  (after step 4, copy from Dokploy's UI URL)

## 4. Dokploy app

- [ ] In Dokploy, create a new **Compose** application from this repo's `docker-compose.yml`.
- [ ] In **Environment**, paste the contents of `.env.example` and fill in:
  - [ ] `HERMES_STAGE=stage_0_paper`  ← **do not change yet**
  - [ ] `HERMES_DOMAIN_API=api.hermes.example.com` (whatever DNS you want)
  - [ ] `TS_AUTHKEY=…`
  - [ ] `LLM_PRIMARY_BASE_URL=http://hermes-mac:11434/v1`
  - [ ] `LLM_PRIMARY_MODEL=qwen3-coder-next`
  - [ ] `LLM_SECONDARY_*` similarly
  - [ ] `LLM_ESCALATION_*` only if using DeepSeek API
  - [ ] `HL_*` — leave blank until stage 1+
  - [ ] `CRYPTOPANIC_TOKEN` — free signup at cryptopanic.com
- [ ] Add the domain in the UI; Traefik labels in `docker-compose.yml` handle the rest.
- [ ] Enable **Auto Deploy** (toggle in the General tab) and add the webhook URL to GitHub → Settings → Webhooks (push events on `main`).
- [ ] Hit **Deploy**. Wait for healthcheck = green. `curl https://api.hermes.example.com/health` must return `{"status":"ok"}`.

## 5. Smoke test (paper)

- [ ] `curl -XPOST https://api.hermes.example.com/tick/btc`   (manual tick — should return a `proposal` and either `REJECTED` or `APPROVED` with paper fill)
- [ ] `curl https://api.hermes.example.com/status` — verify equity = $100, no real funds touched.
- [ ] Watch logs for 24 h. The scheduler runs `market_tick` every 5 min for each symbol in `HERMES_UNIVERSE`. Nightly reflection fires at 23:00 in your `HERMES_TZ`.
- [ ] After ~1 week, hit `/learn/weekly` once to seed skill stats. The cron will run it automatically thereafter.

## 6. Stage advancement (manual, gated)

- [ ] After ~200 paper trades (≈ 4–6 weeks on 1h timeframe × 3 symbols), run `hermes stage`. The report shows pass/fail per gate.
- [ ] If **all gates pass**, fund the Hyperliquid wallet with $100 USDC (Arbitrum bridge), put `HL_ACCOUNT_ADDRESS` + `HL_SECRET_KEY` in Dokploy env, and set `HERMES_STAGE=stage_1_micro`. Redeploy.
- [ ] **Never bypass a failed gate.** The 4-stage ladder exists because tiny bankrolls are unforgiving and LLM agents over-trade.

---

## What's left for you to do — quick checklist

Everything below is ON YOU; the code is ready.

### Mandatory (to run paper)
1. **Tailscale**: account, install on Mac + VPS, ACL, ephemeral `TS_AUTHKEY`.
2. **MacBook**: `brew install ollama`; `ollama pull qwen3-coder-next`; `ollama pull qwen3.6:27b`; run `docker/start-mac-llm.sh`.
3. **Dokploy**: install on VPS; create Compose app from this repo; fill env vars (start with `HERMES_STAGE=stage_0_paper`); deploy.
4. **DNS**: point `api.hermes.example.com` (or your chosen host) at the VPS.
5. **CryptoPanic free token** (optional but recommended): paste into `CRYPTOPANIC_TOKEN`.

### Mandatory before stage 1 (real money)
6. **Hyperliquid wallet**: create one with a fresh seed phrase, fund with $100 USDC bridged to Arbitrum, paste address + secret in Dokploy env. Use a wallet that you treat as cold money — never reuse another wallet.
7. **Manual stage advancement** only after `hermes stage` shows ALL gates green.

### Optional (improves performance — can do later)
8. **TRM checkpoint**: run `hermes.trm.train` against historical OHLCV CSVs, train upstream TRM (`pretrain.py`), drop checkpoint into `/data/trm/latest.pt`.
9. **DeepSeek V4-Pro API key**: paste into `LLM_ESCALATION_API_KEY` if you want a remote brain for hard reasoning.
10. **GitHub Actions secrets** (`DOKPLOY_URL`, `DOKPLOY_TOKEN`, `DOKPLOY_COMPOSE_ID`): enables CI auto-deploy. Otherwise you can rely on Dokploy's built-in GitHub webhook.

### Things you should NOT do
- Do **NOT** set `HERMES_STAGE` higher than `stage_0_paper` before the gates pass. The agent will refuse, but please don't tempt it.
- Do **NOT** put a wallet with > $100 in `HL_SECRET_KEY`. Hot keys leak; treat this as gambling money.
- Do **NOT** bind Ollama to `0.0.0.0` on the Mac. The included script binds to the Tailscale IP only.
- Do **NOT** push secrets to GitHub. Use Dokploy env vars or a Vault sidecar.

That's it. Once steps 1–5 are done, Hermès is live in paper mode and will start learning.
