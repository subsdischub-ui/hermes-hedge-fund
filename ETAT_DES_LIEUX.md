# Hermès — État des lieux & feuille de route (2026-06-20)

> Audit multi-agents (code, déploiement, dépendances, vérification des affirmations externes) +
> recherche d'optimisation quant/IT. Tous les points marqués **VÉRIFIÉ** ont été reproduits
> (commande, fichier:ligne, ou requête API live) et passés au crible adversarial.

---

## 0. Verdict en une page

**Ce qu'est Hermès** : un agent de hedge-fund crypto autonome, bien conçu et étonnamment mûr
(~3 800 lignes Python, architecture multi-agents LangGraph + vérifieur TRM, paper-first, échelle de
risque en 4 stages). Le code **s'importe et tourne**, 16 tests/17 passent, les dépendances résolvent
vers les **dernières versions de mi-2026** (numpy 2.2.6, torch 2.12.1, langgraph 1.2.6, fastapi 0.137,
ccxt 4.5.59). L'hygiène de base est bonne (user non-root, healthcheck, journal idempotent, secrets
non hardcodés, retries tenacity).

**Peut-on déployer aujourd'hui sur Dokploy ?** → **Pas tout à fait, mais on en est très proche.**
Il y a **1 vrai bloquant de déploiement** (les labels Traefik sont sur le mauvais conteneur → API
injoignable par le domaine) corrigeable en déplaçant 7 lignes, plus une poignée de pré-requis
**humains** qui ne dépendent pas de moi (Tailscale, le LLM local sur ton Mac, Dokploy sur le VPS, DNS).

**Pour le paper trading (stage 0)** : déployable dès que (a) le bloquant Traefik est corrigé et
(b) les pré-requis infra sont en place. Les bugs « argent réel » ci-dessous ne mordent pas en paper.

**Avant de passer en argent réel (stage 1+)** : il y a des bloquants de sécurité/fiabilité à corriger
(pas de stops natifs sur l'exchange, API sans authentification, pas d'idempotence d'ordre) **et** une
erreur de conception à trancher (Hyperliquid n'a **pas** de spot BTC/ETH/SOL — voir §2).

---

## 1. Réalité du build & des tests (VÉRIFIÉ localement)

| Élément | Résultat |
|---|---|
| `import hermes.api` | ✅ OK — l'app charge, routes `/health /status /tick/{symbol} /learn/*` |
| `uv sync --extra dev` | ✅ OK — résout vers les versions mi-2026 (uv.lock généré : 169 paquets) |
| `pytest` | ⚠️ **16 passent / 1 échoue** (`greenlet` manquant — voir 3.A) |
| `ruff check` | ❌ **64 erreurs** (43 auto-corrigeables : 15 tris d'imports, 7 `;`, 4 noms ambigus, UP017…) |
| Le piège `pandas-ta` + numpy 2 | ✅ **non déclenché** — le lock prend pandas-ta 0.4.71b0 (numpy2-safe) |

> **Conséquence CI** : le job `test` de `.github/workflows/deploy.yml` lance `ruff check` **et**
> `pytest`. En l'état, **le CI échoue** (ruff + le test greenlet sur runner) → `build-deploy` ne
> se déclenche jamais → l'auto-deploy GitHub→Dokploy est cassé tant que ces 2 points ne sont pas
> réglés. Corrections triviales (`ruff check --fix` + ajouter greenlet).

---

## 2. Bloquants (à corriger selon l'objectif)

### Pour DÉPLOYER (même en paper)

**B1 — Traefik ne peut pas router vers l'API. [VÉRIFIÉ — bloquant]**
`docker-compose.yml:40` met `hermes-agent` en `network_mode: "service:hermes-ts"` (il partage la
pile réseau du sidecar Tailscale et **n'a aucune IP propre** sur `dokploy-network`), MAIS les labels
Traefik et le port 8000 sont déclarés sur `hermes-agent` (lignes 54-60). Traefik cherche l'IP du
conteneur **étiqueté** sur le réseau indiqué et ne la trouve pas → backend `http://:8000` → 502/404.
Le healthcheck interne passe (d'où le piège « conteneur vert mais domaine mort »). Bug Traefik
toujours ouvert mi-2026 (#11645).
**Fix (correctif minimal)** : déplacer **tous** les labels `traefik.*` de `hermes-agent` vers
`hermes-ts`. Ne **pas** sortir l'agent du `network_mode: service:` (ça casserait la résolution
MagicDNS `hermes-mac:11434` via Tailscale, qui est la raison d'être du design).

### Pour passer en ARGENT RÉEL (stage 1+) — pas avant

**B2 — Hyperliquid n'a PAS de spot BTC/ETH/SOL. [VÉRIFIÉ contre l'API live — bloquant conceptuel]**
Requête `POST /info {"type":"spotMeta"}` (2026-06-19) : sur 307 paires spot, **aucune** BTC/USDC,
ETH/USDC, SOL/USDC. Sur Hyperliquid, BTC/ETH/SOL n'existent qu'en **perpétuels** ; le spot se limite
aux tokens HIP-1 (HYPE, PURR) et aux actifs pontés HyperUnit (UBTC/UETH/USOL). Or les stages 1-2
sont configurés `venue: hyperliquid, instruments: spot` sur BTC/ETH/SOL.
**Pire** : `execution/hyperliquid.py:110` fait `coin = symbol.split("/")[0]` → `"BTC"` → le SDK
route **silencieusement vers le PERP**. Donc en stage 1, on ouvrirait des perps (funding, short
possible) en croyant faire du spot.
**Fix** : trancher — (a) perps 1x sur BTC/ETH/SOL (marché natif, déjà ce que le code exécute de
facto → aligner doc/config), (b) spot via HyperUnit (UBTC/USDC…), ou (c) garder spot mais exécuter
sur Kraken/Coinbase. Et faire en sorte que l'adapter honore `instruments` au lieu de router en
silence.

**B3 — Positions live sans stop natif. [VÉRIFIÉ — critique live]** `check_stops` retourne `[]` hors
paper et les ordres HL partent avec `stop_price=None`/`take_profit=None`. Si le conteneur, le Mac/
Ollama ou Tailscale tombe, une position live reste **sans aucun stop**. → attacher des ordres trigger
`reduce_only` (`bulk_orders(grouping="normalTpsl")`).

**B4 — API sans authentification. [VÉRIFIÉ — critique live]** `api.py` n'a aucun `Depends/Security` ;
`POST /tick/{symbol}`, `/learn/nightly`, `/learn/weekly` sont exposés publiquement par Traefik.
N'importe qui connaissant le domaine peut déclencher un trade. → `APIKeyHeader('x-api-key')` global
sauf `/health` + wallet HL « agent » (trading-only, pas de retrait) au lieu de la clé maître.

**B5 — Pas d'idempotence d'ordre (cloid) ni de réconciliation. [VÉRIFIÉ — critique live]**
`Order.client_id` n'est jamais passé en `cloid` ; `place_order` puis `journal.open` (fenêtre de crash
→ position orpheline) ; aucune fonction de réconciliation exchange↔journal. Un retry réseau peut
doubler une position. → dériver un `Cloid`, `query_order_by_cloid` avant retry, job de réconciliation
au démarrage + périodique, gérer les fills partiels.

---

## 3. Bugs & dettes (hors bloquants) — par sévérité

### Élevé

- **A. `greenlet` manquant. [VÉRIFIÉ]** SQLAlchemy 2.x n'auto-installe pas greenlet sur Mac `arm64`
  (marqueur PyPI sans `arm64`). → la DB async plante **sur ton Mac** (dev/test). **Le VPS Linux
  x86_64 n'est PAS affecté.** Fix : `sqlalchemy[asyncio]>=2.0` (ou `greenlet>=3.1`) dans pyproject.
- **C. Equity des shorts surestimée en paper. [VÉRIFIÉ, reproduit]** `paper.py:100` calcule
  `cash + qty*(px-entry)` mais le cash est déjà crédité du notional à l'ouverture → surestimation
  constante de `|qty|*entry` (SELL 1@100 → equity 1100 au lieu de 1000). Gonfle le sizing et retarde
  le kill-switch. Fix : `total += qty*px`. Latent en spot-only, actif dès qu'un short existe.
- **D. Variables d'env de risque ignorées. [VÉRIFIÉ, reproduit]** `guardrails.py:126,134` lit
  `daily_loss_kill_pct`/`max_open_positions` **depuis le YAML**, jamais depuis `Settings`.
  `HERMES_DAILY_LOSS_KILL_PCT=2.0` est sans effet (le kill-switch stage0 reste à 5.0). Faux sentiment
  de sécurité aux stages live. Fix : merger les overrides env dans `stage_config`, ou retirer ces
  variables + corriger le commentaire trompeur `default.yaml:1`.
- **E. uv.lock non utilisé ET non commité → builds non reproductibles. [VÉRIFIÉ]** Le Dockerfile fait
  `uv pip install -e .` (pas `uv sync --frozen`) et ne copie pas le lock ; de plus `uv.lock` n'est pas
  dans git. Chaque build re-résout les ranges → l'image prod peut diverger du CI. Fix : `git add
  uv.lock` ; `COPY pyproject.toml uv.lock` ; `uv sync --frozen --no-dev` ; **et** bumper l'image
  `astral-sh/uv:0.5` (trop vieille pour `revision=3` du lock) vers `uv:0.11`.

### Moyen

- **env_file Dokploy** : `env_file: .env` est le bon pattern, mais des bugs Dokploy par version (#2777
  & co.) placent parfois le `.env` au mauvais endroit. À **vérifier après déploiement** (`docker exec
  hermes-agent env | grep HERMES_STAGE`). Atténué : le repo déploie en mode Git, pas Raw.
- **TRM_COMMIT=main** (`Dockerfile:21`) contredit le commentaire « pin a specific commit » : clone
  réseau au build, non figé. Atténué : le repo upstream est **archivé** (read-only) depuis 2026-04-01.
  Fix : pinner un SHA, ou vendoriser le sous-ensemble TRM.
- **Auth API Dokploy (CI)** : le workflow envoie `Authorization: Bearer`, mais la doc Dokploy attend
  `x-api-key`. Selon la version → 401. Fix : `-H "x-api-key: ${DOKPLOY_TOKEN}"`.
- **CryptoPanic** : `news.py:27` code en dur le plan `/api/developer/v2/` alors que la doc dit
  « free signup ». Avec un token gratuit, c'est probablement `/api/free/v2/` et le filtre `currencies`
  est réservé aux plans payants. Fallback Reddit présent. À tester.
- **Dépendances mortes lourdes** : `sentence-transformers` (jamais importé — Chroma utilise son ONNX
  par défaut), `redis` (provisionné mais aucun `import redis`), `pandas-ta` (jamais importé, ATR
  calculé à la main). Les retirer allège l'image de plusieurs centaines de Mo.
- **Floors périmés** (le lock est sain, mais les `>=` traînent) : `chromadb>=0.5` (autorise un downgrade
  pré-rewrite-Rust 1.x, migration irréversible), `openai>=1.55`, `langgraph>=0.2.50`, `pandas-ta>=0.3.14b`
  (autorise la 0.3.14b cassée sous numpy 2). Resserrer : `chromadb>=1.5,<2`, `openai>=2,<3`, etc.

### Faible / info

- Orchestrateur : `trader/risk/portfolio_manager` hors du wrapper `_safe` → une exception remonte en
  500 non structurée (`orchestrator.py:38-40`).
- `paper.py` lit `_state` sans `self._lock` (risque `dictionary changed size` : stop-watcher 30s vs
  tick 5min).
- `torch.load(weights_only=False)` (`verifier.py:104`) : mauvaise pratique, mais risque réel faible
  (torch locké 2.12.1 patché, checkpoint produit hors-bande par l'opérateur). Fix : `weights_only=True`.
- Affirmation « Binance ne liste que USDT » : **fausse** (sans impact, le paper price est sur Kraken).

### Affirmations externes — VÉRIFIÉES réelles
`qwen3-coder-next` (Ollama, 80B-A3B, ~52 GB en q4_K_M — pas 40 GB), `qwen3.6:27b`, `deepseek-v4-pro`
(API DeepSeek), repo TRM `SamsungSAILMontreal/TinyRecursiveModels` (MIT, archivé), `all-MiniLM-L6-v2`,
alternative.me F&G, Dokploy install + `/api/compose.deploy`, ccxt/hyperliquid-sdk : **tous réels**.
Réserve : `MLX via Ollama` (README) est imprécis (Ollama = GGUF/llama.cpp).

---

## 4. Feuille de route d'optimisation (math + IT)

**Principe directeur (non négociable)** : à 100 $ en 24/7, l'ordre est **(1) ne pas se ruiner / casser
en live → (2) ne pas sur-trader / mesurer faux → (3) ne pas sur-apprendre → (4) seulement ensuite,
chercher l'alpha.** ~40 pistes fusionnées en 8 niveaux.

### Niveau 0 — Sécurité/fiabilité live (PROUVÉ) — = B3, B4, B5 ci-dessus + :
- **0.4 min-notional 5 → 10 $** (`guardrails.py:173`, 1 ligne) : HL rejette < 10 $, donc un ordre
  « approved » entre 5 et 10 $ est rejeté à l'exécution.
- **0.5 dead-man's switch / heartbeat + circuit breaker** sur les dépendances (Ollama/Tailscale/exchange)
  avec mode « no new positions ».

### Niveau 1 — Ne pas se ruiner par les coûts (PROUVÉ, surtout quick wins)
- **1.1 (le plus important) frais paper réalistes** : `paper.py:38` `FEE_BPS=5` ; le vrai taker spot HL
  est **7 bps** (perp 4.5). Le paper sous-estime de ~40 % → **Sharpe gonflé qui débloque les gates à
  tort**. Mettre une section `fees:` par instrument et re-mesurer les gates.
- **1.2 porte « edge-net-of-fees » + cadence** : tick 5 min sur bougie 1h = jusqu'à 12 décisions
  redondantes/barre. À 100 $, viser **1h/4h** et rejeter si TP attendu < k·(2·frais+spread+slippage).
- **1.3** le plancher 0.25 %/trade est **mort** (= 0.25 $ de risque, impossible avec min-notional 10 $) :
  raisonner en notional.
- **1.4 ordres maker post-only** : divise le coût par ~2 (4 bps au lieu de 7). Arrondir qty/prix aux
  `szDecimals`/tick lus dans `meta()`.
- À **écarter** (anti-sur-ingénierie) : Almgren-Chriss / TWAP / VWAP — impact de marché nul à 100 $.

### Niveau 2 — Ne pas mesurer faux (anti-overfit amont, PROUVÉ)
- **2.1 Sharpe → PSR / Deflated Sharpe Ratio** dans le re-rank hebdo (`weekly.py`) : aujourd'hui
  annualisation fausse + i.i.d. supposé + aucune pénalité multi-tests + archivage sur `sharpe<0` sur
  ~10 trades = cherry-picking. Classer par PSR, exiger MinTRL et DSR > 0.95 (~40 lignes numpy).
- **2.2 gates de stage : exiger la borne basse de l'IC**, pas le point-estimate (`stage.py`). C'est là
  que se décide l'escalade de risque réel.
- **2.3 (structurant) `walk_forward_sharpe` est un placeholder** : aucun backtest/CV n'existe en code.
  Module `validation/` avec **Combinatorial Purged CV + PBO** (skfolio), frais HL inclus.
- **2.4 calibrer la confiance LLM (Platt/isotonic) AVANT le sizing Kelly** : la confiance sert
  non-calibrée 2× (gate `<0.55` + multiplicateur de Kelly). Les LLM sont sur-confiants → 0.8 réel ~0.55
  sur-dimensionne de ~45 %. Logger `(conf, win/loss)` dès maintenant ; fitter `IsotonicRegression` à
  ~50-100 trades. Complète (ne remplace pas) le TRM.

### Niveau 3 — Risque portefeuille corrélation-aware (PROUVÉ, impact élevé)
- **3.1 haircut de corrélation + budget de risque joint** : BTC/ETH/SOL corrélés 0.85-0.93 = un
  quasi-mono-facteur « BTC beta ». 3 longs = ~3× le beta BTC sous une apparence de « 0.5 %/trade ».
  Passer les positions ouvertes au sizer, `risk_pct/√(1+(n-1)ρ)`, cap dur sur `√(w'Σw)` (Σ Ledoit-Wolf).
  **Ne PAS** mettre un optimiseur HRP/NCO à 3 actifs (instable — réserver à ≥8-10 actifs décorrélés).
- **3.2 kill-switch sur drawdown** (equity totale, incl. unrealized) en plus du PnL réalisé.

### Niveau 4 — Meilleur signal de risque/sizing (PROUVÉ, quick wins)
- **4.1 prévision de vol HAR-RV** pour le vol-targeting (remplace l'ATR brut). Baseline dure à battre,
  triviale à coder.
- **4.2 stops chandelier / ATR-trailing** au lieu d'un stop statique.
- **4.3 Kelly bayésien/shrinkage continu** au lieu du saut binaire à 100 trades. **Garder 0.20, ne pas
  l'augmenter.**

### Niveau 5 — Régime + agrégation de signaux (régime = PROUVÉ, impact élevé)
- **5.1 détecteur de régime HMM/BOCPD** (1h/4h) → conditionne quel skill activer **en temps réel**
  (le re-rank hebdo réagit en 1-4 semaines, trop lent).
- **5.2 ensembling déterministe des signaux** avant le LLM (auditable, réduit la dépendance au débat).
- **5.3 funding rate comme FEATURE** (pas comme stratégie de carry — le carry est devenu **négatif**
  en 2025).

### Niveau 6 — Alpha plus structuré (ÉMERGENT, en shadow d'abord)
- **6.1 méta-labeling triple-barrier** (LightGBM) pour filtrer les trades du LLM → p(win). Brancher en
  **shadow** (loggé, non bloquant) avant de gater.
- **6.2 détection de decay d'alpha** (BOCPD par skill) au lieu d'archiver a posteriori sur Sharpe<0.

### Niveau 7 — Plomberie IT/ML (utile, hors chemin critique alpha)
- **7.1 décodage contraint par grammaire** (XGrammar/llguidance) : Ollama **ignore** le `json_schema`
  strict → la fiabilité réelle = un fallback regex. Forcer `tool_choice` sur `propose_trade` + valider
  via Pydantic. Fiabilise directement les décisions de trade.
- **7.2 backend MLX (+ décodage spéculatif)** : 1.2-3× plus rapide que llama.cpp sur Apple Silicon →
  réduit la latence par tick (5-7 appels primaires × 3 symboles). Zéro changement côté agent (garder
  l'abstraction OpenAI-compatible de `router.py`).
- **7.3 bug embedding** : `vector.py:21` n'applique **jamais** l'`embedding_model` configuré (Chroma
  utilise son MiniLM par défaut → `self.embedding_model` est du code mort). Corriger d'abord, upgrade
  ensuite (EmbeddingGemma/bge-m3) + ré-indexation.
- **7.4 jobstore APScheduler persistant** (Redis déjà provisionné, ou SQLAlchemyJobStore) : sinon tout
  redémarrage perd les crons nightly/weekly/monthly.
- **7.5 observabilité** : `/metrics` Prometheus (equity, drawdown, PnL, latence, reject/approve) +
  alerting (drawdown, 0 tick depuis X min). Aujourd'hui : structlog seul.
- **7.6 rate-limits HL** : mutualiser `all_mids()` une fois par tick global, cacher `meta()`.

### Niveau 8 — Backlog R&D (ÉMERGENT/EXPÉRIMENTAL, ne pas prioriser)
GEPA réel (DSPy) ; TSFM (TimesFM 2.5/Chronos-2) **pour la vol seulement** (« forecast collapse » sur
la direction) ; débat bull/bear (la self-consistency l'égale souvent à budget égal) ; GRPO/ART (RL) ;
mémoire agentique Letta/Mem0 (surdimensionné à 100 $) ; stat-arb/order-flow/on-chain (non rentables
après frais à cette échelle).

### Les 5 façons dont « optimiser » NUIRAIT
1. **Sur-apprentissage par auto-modification non validée** — rien d'auto-modifiant en bloquant tant
   que le Niveau 2 (CPCV/PBO/DSR) n'est pas là. Toujours shadow d'abord.
2. **Latence qui casse le budget du tick** — empiler régime+HAR+TSFM+méta-labeler+débat+grammaire peut
   dépasser la fenêtre. Cacher hors chemin critique, MLX, réduire le débat avant d'ajouter.
3. **Faux sentiment de diversification** — tant que 3.1 n'est pas fait, le sizing par position ment.
   Un optimiseur HRP à 3 actifs **aggraverait**.
4. **Complexité opérationnelle = nouveaux points de panne** — pour un agent 24/7 solo, préférer les
   correctifs ciblés à la grande réécriture (LangGraph seulement si l'orchestration maison cale).
5. **Optimiser le sizing/alpha avant de corriger frais/min-notional/stops** = peaufiner la peinture
   d'une voiture sans freins. L'ordre des niveaux n'est pas négociable.

### Plan d'attaque (sprints)
1. **Sécurité live** : 0.1 stops natifs, 0.2 auth+wallet restreinte, 0.4 min-notional, 0.5 dead-man,
   0.3 cloid+réconciliation → débloque le stage 1.
2. **Anti-ruine coûts** : 1.1 frais réels, 1.2 porte edge-net+cadence, 1.3 floor, 7.4 jobstore.
3. **Anti-faux** : 2.1 PSR/DSR, 2.4 calibration (logging immédiat), 2.2 gates IC.
4. **Risque portefeuille + signal** : 3.1 budget joint, 3.2 DD kill-switch, 4.1 HAR-RV, 4.2 trailing,
   5.1 régime, 5.3 funding feature.
5. **Rigueur lourde + alpha shadow** : 2.3 CPCV/PBO, 5.2 ensemble, 6.1 méta-labeling (shadow),
   7.1 grammaire, 7.2 MLX, 7.3 embedding fix.

---

## 5. Déploiement Dokploy — go/no-go & répartition des rôles

### Ce que je PEUX faire (en pilotant ton navigateur sur l'UI Dokploy)
- Créer l'application **Compose** depuis ce repo, coller/renseigner les variables d'environnement,
  ajouter le domaine, lancer le déploiement, lire les logs/healthcheck, itérer.
- Au préalable, corriger dans le repo : **B1 (labels Traefik)**, ruff `--fix`, greenlet, lockfile —
  pour que le déploiement ait une chance de réussir du premier coup.

### Ce que TOI seul peux faire (pré-requis, hors de mon contrôle)
1. **Tailscale** : créer le compte, l'installer sur le Mac **et** le VPS, appliquer l'ACL, générer la
   clé `TS_AUTHKEY` (éphémère, taggée).
2. **Le cerveau LLM sur ton Mac** : `ollama pull qwen3-coder-next` (~52 GB) + `qwen3.6:27b`, lancer
   `docker/start-mac-llm.sh`. **Sans ce LLM joignable, l'agent ne fait rien — même en paper.**
3. **VPS** : installer Dokploy (`curl -sSL https://dokploy.com/install.sh | sh`), avoir ~10 GB libres.
4. **DNS** : pointer ton domaine d'API vers le VPS.
5. Vérifier que les **tags de modèles** existent réellement (`ollama pull …` réussit) avant de lancer.

### Recommandation
- **Pour démarrer en paper** : on corrige B1 + les 3 quick wins CI (ruff/greenlet/lock), tu mets en
  place les pré-requis infra, puis je pilote Dokploy pour déployer. Les bloquants « argent réel »
  (B2-B5) ne sont pas requis pour le paper.
- **Avant tout passage stage 1** : traiter le Sprint 1 (sécurité live) + trancher B2 (spot vs perps).
