# Dossier Hermès — fonctionnement, recherches & incident

> **Date** : 2026-06-25 · **Capital** : $100 (paper, stage_0) · **Stade** : paper-first, aucune capital réel engagé
> **Déploiement** : conteneur unique sur VPS/Dokploy, `https://api.betterlifewithmk.com`
> **Cerveau** : DeepSeek V4 (primaire, direct) · **Fallback** : OpenRouter -> `inclusionai/ring-2.6-1t`

Hermès est un **fonds algorithmique crypto autonome 24/7** : un système multi-agents (Python/FastAPI,
graphe d'agents LLM + vérifieur TRM) qui décide, exécute (en paper), apprend et se surveille seul, sous
une discipline imposée par le code. Sa contrainte fondatrice est **l'honnêteté absolue** : aucune
stratégie n'est mise en production sans avoir survécu à une machine de validation adverse (null de
Monte-Carlo, OOS par date, déflation), et tout résultat est restitué sans embellissement.

Ce dossier documente **trois choses** :
- **Partie I — Le fonctionnement** : comment l'entreprise tourne, du tick de marché à l'exécution, en
  passant par les stratégies, le risque, le cerveau LLM et la machine d'honnêteté.
- **Partie II — Les recherches du labo** : tout ce qui a été tenté pour *étendre* Hermès (matières
  premières/FX, nouvelles données crypto, re-audit du book live) — l'essentiel **réfuté**, et pourquoi
  c'est un résultat précieux.
- **Partie III — L'incident du 25 juin & les réajustements** : la panne qui a arrêté l'agent « tout
  seul », son diagnostic forensique, et la nouvelle configuration (mono-conteneur, sans Tailscale) qui
  en découle.

> ⚠️ *Note de fidélité* : les Parties I & II sont cartographiées directement depuis le code et le corpus
> de recherche (citations `fichier:ligne`). Quelques chiffres de performance « validés » qui y figurent
> (ex. l'OOS Sharpe « +1.1 » de la Stratégie #2) sont les valeurs **annualisées historiques** ; la
> Partie II et le re-audit du 25 juin les **recalibrent honnêtement** (OOS-40 ~0.3). En cas de
> divergence, le re-audit (Partie II, §re-audit) fait foi.

---

# PARTIE I — Le fonctionnement de l'entreprise

## 1. Le moteur — la boucle de trading

### 1.1 Contexte global et configuration

**Bankroll et univers** : `/src/hermes/settings.py:32, 33`
- Capital initial : `$100 USD` (paper, stage_0) configurable via `HERMES_BANKROLL_USD`
- Univers : `7 perps Binance USDT-M` : `BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `AVAX/USDT:USDT`, `SUI/USDT:USDT`, `ARB/USDT:USDT`, `WIF/USDT:USDT` (défini dans `config/default.yaml:11`, code-driven)
- Venue data : Binance USDT-M futures (`binanceusdm`) — lieu où les stratégies ont été validées en backtest
- Venue paper : Même que la data venue (cohérence des prix et OHLCV)

**Stage (maturity ladder)** : `/src/hermes/settings.py:28` et `config/default.yaml:16-73`
- Actuel : `stage_0_paper` (l'agent ne traduit QUE en paper money jusqu'à 200 trades + Sharpe >= 1.0)
- Progression : `stage_0_paper` → `stage_1_micro` (live Hyperliquid, $X risque/trade) → `stage_2_normal` → `stage_3_perps` (leverage)
- Chaque stage exige : `max_open_positions` limité (stage_0 = 6, stage_1-2 = 3-5) et `daily_loss_kill_pct` (2-5% — stop-loss quotidien)

**Kill-switch (stop-loss quotidien)** : `/src/hermes/risk/guardrails.py:142-150`
- Déclenché si `realized_pnl_today_usd / equity <= -daily_loss_kill_pct` (stage_0 = -5%, stage_1 = -2%)
- Calculé via `/src/hermes/memory/store.py` → toutes les clôtures depuis minuit UTC
- Évalué AVANT chaque risque check dans `RiskEngine.evaluate()`

---

### 1.2 La cadence : scheduler APScheduler

**Fichier** : `/src/hermes/scheduler.py:24-124`

**Tâches programmées** (en parallèle, APScheduler-driven) :

| Job | Cadence | Fonction |
|-----|---------|----------|
| `market_tick` | Tous les 5 min (par défaut `*/5 * * * *`, configurable `loops.market_tick` dans YAML) | Décision BUY/SELL/HOLD + exécution |
| `stop_watcher` | Toutes les 30s (`IntervalTrigger(30s)`) | Fermeture des stops/TP, Reflexion post-close |
| `pulse` | Chaque heure (`0 * * * *`) | Snapshot des signaux actuels (log-only, pas d'ordre) |
| `status_heartbeat` | Chaque heure (`0 * * * *`) | Log du PnL/equity/positions |
| `nightly_reflection` | 23:00 UTC (`0 23 * * *`) | Memory/lessons extraction |
| `weekly_rerank` | Dimanche 22:00 UTC (`0 22 * * 0`) | Archival des skills sous-performants |
| `monthly_compile` | 1er du mois 21:00 UTC (`0 21 1 * *`) | Compilation mensuelle des métriques |

**Startup** : `/src/hermes/api.py:49-70`
- Lifespan FastAPI démarre APScheduler
- Rehydrate la lib des skills depuis la DB (archival persiste)
- Logs de démarrage : `hermes_starting` avec stage/bankroll/univers

---

### 1.3 Un tick complet (5 min) — orchestration multi-agents

**Point d'entrée** : `/src/hermes/scheduler.py:34-55` → appel `orch.tick(symbol, timeframe=decision_tf)` pour CHAQUE symbol en parallèle

**Graphe d'orchestration linéaire** : `/src/hermes/graph/orchestrator.py:28-55`

```
Parallèle (concurrence max) :
  market_analyst   ─┐
  news_analyst     ─┼─► fetch_lessons  ► bull ║ bear  ► trader  ► risk_manager  ► portfolio_manager
  sentiment        ─┘
```

**État partagé** : `/src/hermes/graph/state.py` — un `HermesState` alimente la chaîne

---

#### Étape 1 : Market Analyst (parallèle, 3 agents lancés simultanément)

**Agent** : `/src/hermes/agents/market_analyst.py:24-100`

**Inputs** :
- `symbol` (BTC/USDT:USDT, etc.)
- `timeframe` = "4h" (décision_timeframe depuis YAML)

**Étapes** :
1. **Fetch OHLCV** : Via CCXT (async) depuis Binance `ohlcv(symbol, timeframe="4h", limit=300)` → 300 barres de 4h (soit ~50j historique)
2. **Attach funding** : `md.funding_history(symbol)` — récupère l'historique de taux de financement perp live via Binance REST public (`/fapi/v1/fundingRate`, 200 derniers)
3. **Attach shadow features** : Options IV (DVOL z-score) et on-chain stablecoin supply growth z (best-effort, fail-soft)
4. **Calcul du régime** : `detect_regime(df)` → hysteresis-routed TREND/RANGE/NEUTRAL (Efficiency Ratio 0..1)
5. **Skill evaluation** : `get_skill_library().evaluate_for_symbol(symbol, df)` → exécute les 4 skills actifs (ts_mom_vol_scaled, mean_reversion_regime, funding_squeeze_continuation, funding_carry_harvest)
   - **Signal #1 (Momentum)** : ts_mom_vol_scaled : |z-score momentum| >= 1.0 → (side=BUY/SELL, conf=z, stop=2%, tp=4%)
   - **Signal #2 (Funding squeeze)** : funding_squeeze_continuation : funding_z >= 2.0 → (side=LONG, conf~funding_z, tp=96h)
   - **Signal #3 (Carry delta-neutral)** : funding_carry_harvest : funding_bps >= 2.0 → (side=LONG perp + SHORT spot pair, ~8j)
   - **Shadow #4 (DVOL)** : options_iv_dvol_reversion : |dvol_z| >= 2.0 → (log-only, never executes)
6. **Regime-weight signals** : Down-weight off-regime signals (mean-reversion en tendance, momentum en range) → `confidence *= regime_weight`
7. **Snapshot market** : Stocke price, ATR, vol avg/last, high/low 20-bar, régime → `state.snapshot`

**Outputs** :
- `state.ohlcv` : DataFrame complet 300 barres
- `state.skill_signals` : liste de signaux structurés (side, confidence, stop_pct, tp_pct)
- `state.regime` : {regime, efficiency_ratio}
- `state.snapshot` : {price, atr, ret_1, vol_avg_20, high_20, low_20, ...}

**Timeout/Error handling** : Fail-soft — si funding/shadow indisponible, skill HOLDs automatiquement

---

#### Étape 2 : News & Sentiment Analysts (parallèle avec market)

**Agents** : `/src/hermes/agents/news_analyst.py` et `/src/hermes/agents/sentiment_analyst.py`

**Inputs** : symbol, snapshot

**Outputs** :
- `state.news` : liste headlines + sentiment
- `state.sentiment` : score agrégé

*(En pratique, ces agents se connectent à des APIs externes — l'architecture est prête mais les clés ne sont souvent pas configurées)*

---

#### Étape 3 : Fetch Lessons (séquentiel après les 3 analysts)

**Fonction** : `/src/hermes/agents/researcher.py:28-32`

**Inputs** : symbol, timeframe, snapshot.atr_pct

**Requête vectorielle** : `LessonsStore().search(f"{symbol} {timeframe} regime atr {atr_pct:.4f}", k=6)` → retrouve les leçons passées du journal de réflexion (apprentissage du passé)

**Outputs** : `state.relevant_lessons` → max 6 lessons texte

---

#### Étape 4 : Bull & Bear Researchers (parallèle)

**Agents** : `/src/hermes/agents/researcher.py:35-81`

**Inputs** : 
- snapshot
- skill_signals
- news
- sentiment
- relevant_lessons
- régime

**LLM** : Router LLM (DeepSeek V4 (primaire, direct via api.deepseek.com) ; fallback OpenRouter -> inclusionai/ring-2.6-1t)

**Bull** : Construit un JSON structuré : `{thesis, skill_id, confidence, counter_risk}`
**Bear** : Idem mais side peut être SELL ou HOLD

**Outputs** :
- `state.bull_view` : JSON string
- `state.bear_view` : JSON string

---

#### Étape 5 : Trader (séquentiel, après researchers)

**Agent** : `/src/hermes/agents/trader.py:13-52`

**Inputs** :
- symbol, snapshot
- bull_view, bear_view
- skill_signals (liste des signaux)
- lessons (top 6)

**LLM call** : 
- System prompt : `/config/prompts/trader.md` (pas lue ici mais chargée dynamiquement)
- Tool calling : `propose_trade(symbol, side, confidence, stop_pct, tp_pct, thesis, skill_id)`
- Fallback : si pas de tool call, défaut HOLD avec confidence=0.0

**Output** :
- `state.trader_proposal` : {symbol, side, confidence, stop_pct, tp_pct, thesis, skill_id}

**Note critique** : En mode sélectif (selective_mode=true), ce proposal est **advisory** — le déterministe prend le dessus

---

#### Étape 6 : Risk Manager (séquentiel après trader)

**Agent** : `/src/hermes/agents/risk_manager.py:28-165+`

**Inputs** :
- trader_proposal
- state.ohlcv (pour TRM encoding)
- state.skill_signals
- positions ouvertes actuelles

**Logique en 4 substeps** :

##### 6a. Selective Mode Override (déterministe)
- Fichier `/src/hermes/risk/selective.py:139-157` — fonction `drive_signal()`
- SI `selective_mode=true` dans config :
  - Récupère la plus haute conviction parmi les signaux qualifiés (allowed_skills, conviction_min >= 1.0 pour ts_mom)
  - Applique le gate no-pyramid : refuse si (symbol, skill) déjà ouvert
  - **OVERRIDE le trader LLM** — la rule déterministe devient driver
  - Log `selective_override` si côté change

##### 6b. TRM Refinement (verifier NN)
- Fichier `/src/hermes/trm/verifier.py` — transformer model trained
- Encode : market state (OHLCV grid 40x8) → tensor input
- Encodes : action proposée (side, size_pct, stop_pct, tp_pct) → tensor
- Inférence TRM : `verifier.refine(grid, draft, instrument_id=0)` → `verdict: {refined_action, confidence, used_trm}`
- **Règle stricte** : TRM peut downgrade à HOLD (si confidence > 0.7), JAMAIS upgrade HOLD→BUY/SELL
- **Confidence ceiling** : `proposal.confidence = min(proposal.confidence, trm_confidence)` (TRM peut réduire, pas augmenter)

##### 6c. Confidence Calibration (isotonic)
- Fichier `/src/hermes/risk/calibration.py` — model fitted sur historical realized P(win)
- Maps proposal.confidence (brut) → calibrated (honnête historical hit-rate)
- Identity (no-op) jusqu'à enough trades (mode identity ou fitted)

##### 6d. Deterministic Risk Engine (`RiskEngine.evaluate()`)
- Fichier `/src/hermes/risk/guardrails.py:119-150+`
- **Hard gates** (tous sont REJECT):
  - side != HOLD et symbol not in universe → REJECT
  - calibrated_confidence < confidence_floor (défaut 0.55) → REJECT
  - equity < $5 → REJECT (cold-floor)
  - daily_loss_today <= -kill_pct * equity → REJECT (kill-switch)
  - open_positions >= max_open_positions → REJECT
  - symbol already open AND no-pyramid → REJECT
  - stage = spot-only ET side=SELL ET no existing long → REJECT (cannot short)
  - stage = paper ET configured_venue != paper → REJECT

**Sizing** : Fichier `/src/hermes/risk/sizing.py`
- Inputs : equity_usd, confidence, win_rate (du journal), avg_win_loss_ratio, ATR, price, trades_in_journal
- **Kelly contribution** : 
  - Si trades >= 100 : `kelly_f = max(0, p*b - (1-p)) / b` où p=win_rate, b=avg_win_loss
  - Sinon : 0
- **Risk calculation** : 
  - `risk_pct = min(stage_cap_pct, max(floor_pct, fractional_kelly * kelly * confidence))`
  - Haircut corrélation : `risk_pct *= 1 / sqrt(1 + n_existing * rho)` (anti-stacking)
- **Notional** : `notional = (risk_pct * equity) / (k*ATR/price)` où k=2.0
- **Quantity** : `qty = notional / price`
- **Stop/TP** : `stop_distance = k * ATR / price` (2.0*ATR), TP = 3..4x stop

**Output** :
- `state.risk_decision` : {decision: "APPROVED"|"REJECTED", reason, order: {...}, sizing: {...}}
- `state.trm_refinement` : {side, size_pct, confidence, used_trm}

---

#### Étape 7 : Portfolio Manager (séquentiel après risk)

**Agent** : `/src/hermes/agents/portfolio_manager.py:66-119`

**Logique** :
1. Si `risk_decision.decision == "REJECTED"` → Stop, set `executed=False`
2. Sinon, **LLM final check** : appel le PM LLM avec (trader_proposal, risk_decision, TRM, snapshot) → peut downgrader à REJECTED (veto final) mais NEVER upgrade
3. Si risk approuve et PM n'a pas dit non → **Execute**

**Execute** :
- Crée un `Order` struct avec les params du risk_decision
- Appelle `exchange.place_order(order)` → 
  - **Paper** : `/src/hermes/execution/paper.py` — simule un fill market avec slippage
  - **Live** : Hyperliquid ou Bybit
- Journal entry via `TradeJournal.open()` → record dans la DB
- Log `pm_executed` avec trade_id

**Output** :
- `state.executed` : bool
- `state.trade_id` : int (DB primary key)
- `state.error` : str si failed

---

### 1.4 Post-tick : Activity logging

**Fonction** : `/src/hermes/scheduler.py:40-51`

Après chaque tick, record un entry dans `/data/activity.json` (ring buffer 200 entries) :
```json
{
  "kind": "decision",
  "symbol": "BTC/USDT:USDT",
  "executed": true/false,
  "side": "BUY",
  "decision": "APPROVED",
  "reason": "passed risk gate",
  "skill": "ts_mom_vol_scaled",
  "regime": "TREND",
  "selective_gate": {...}
}
```

Utilisé par le dashboard pour l'activity feed en temps réel.

---

### 1.5 Pulse horaire (LLM-free)

**Fonction** : `/src/hermes/runtime/pulse.py:92-140`

Runs chaque heure (indépendant des décisions 4h).

**Sans LLM, déterministe** :
1. Fetch OHLCV + funding pour chaque symbol
2. Compute regime, skill signals
3. Evaluate selective_gate (would it execute?)
4. Record pulse entry dans activity.json :
```json
{
  "kind": "pulse",
  "symbol": "BTC/USDT:USDT",
  "regime": "TREND",
  "efficiency_ratio": 0.65,
  "signals": [{"skill": "ts_mom_vol_scaled", "side": "BUY", "conf": 1.2}],
  "gauges": [
    {"sleeve": "Momentum #1", "metric": "conviction", "val": 1.2, "thr": 1.0, "armed": true}
  ],
  "would_act": "BUY",
  "would_skill": "ts_mom_vol_scaled",
  "reason": "would enter (passes selective gate)"
}
```

Dashboard affiche les gauges en direct entre les ticks 4h → opérateur voit que l'agent "respire".

---

### 1.6 Stop Watcher (toutes les 30s)

**Fonction** : `/src/hermes/runtime/stop_watcher.py:30-76`

Boucle infinie, in-band async :
1. Pour chaque position ouverte, check si price >= take_profit OU price <= stop_loss
2. Si déclenché, close via `exchange.close_position()`
3. Calcule realized PnL : `(exit_price - entry_price) * qty - fee`
4. Update journal : `journal.close(trade_id, exit_price, pnl_usd, exit_reason="stopped_out")`
5. Trigger Reflexion (`reflect_on_trade()`) → extract lessons asynchronously
6. Log `stop_watcher_closed` avec count

**Key point** : FIFO matching journal → trade_id identifie la ligne journal, PnL est attribué au plus vieux open pour ce symbol

---

### 1.7 FastAPI surface + orchestration

**Fichier** : `/src/hermes/api.py`

**Endpoints** :

| Route | Méthode | Auth | Fonction |
|-------|---------|------|----------|
| `/health` | GET | — | Container healthcheck → `{status: ok}` |
| `/status` | GET | x-api-key | État live : stage, equity, positions, journal stats, stage gates |
| `/dashboard` | GET | — | HTML dashboard (read-only telemetry) |
| `/dashboard/data` | GET | — | JSON : equity_curve, strategies A/B, activity, scheduler jobs |
| `/tick/{symbol_base}` | POST | x-api-key | Force manual tick pour un symbol (debugging) |
| `/chat` | POST | x-api-key | Opérateur demande "où en es-tu?" → Hermès répond en français (advisory-only, pas d'exécution) |
| `/learn/nightly` | POST | x-api-key | Trigger nightly reflexion manuelle |
| `/learn/weekly` | POST | x-api-key | Trigger weekly skill rerank manuel |

**Lifespan** : `/src/hermes/api.py:49-70`
- Startup : APScheduler.start(), load skill library from DB
- Shutdown : APScheduler.shutdown(), close DB — **positions ouvertes ne sont PAS fermées automatiquement** (opérateur doit le faire manuellement)

---

### 1.8 Combinaison des stratégies (combination layer)

**Fichier** : `/src/hermes/strategies/combination.py`

Pure advisory (no sizing impact) :
- **Route regime** : Hysteresis router TREND ↔ RANGE (enter 0.4 ER, exit 0.3)
- **Select active skills** : Selon regime, active ts_mom (trend) + mean_reversion_regime (range) + funding-independent (always-on)
- **Inverse-vol weights** : Blend skills ∝ 1/variance(recent_returns) — least volatile gets most capital
- Live blend = 0 (ne s'active que quand per-skill journal returns sont plombés)

---

### 1.9 Résumé de la boucle 4h par symbol

```
1. market_analyst.run()          ← ohlcv(4h) + funding + regime + skill_signals
2. {news, sentiment}_analyst     ← headlines + sentiment score
3. researcher.fetch_lessons()    ← vector search past learnings
4. researcher.{bull, bear}()     ← LLM builds hypotheses
5. trader.run()                  ← LLM proposes trade
6. risk_manager.run()            ← selective override + TRM + calibration + REJECT/APPROVE
7. portfolio_manager.run()       ← LLM final veto + EXECUTE order + journal.open()
8. activity.record(decision)     ← log to activity feed
9. stop_watcher (parallel 30s)   ← monitor stops, close + reflexion + journal.close()
```

**Résultat final** :
- Si approved + executed : `Order` placé, `JournalEntry` opened, `state.executed=true`, `state.trade_id=N`
- Si rejected : aucun ordre, `state.executed=false`, `state.risk_decision.reason` explique pourquoi

---

### 1.10 Configuration clé (stages + règles)

**Stage 0 (Paper, actuel)** : `/config/default.yaml:17-34`
- Venue : paper (Binance prices pour fill)
- Instruments : perp (pour short)
- Risk cap/trade : 0.5% equity
- Max positions : 6
- Kill-switch daily : 5%
- Advance gate : 200 trades + Sharpe >= 1.0 + max drawdown <= 15%

**Skills live** : `[ts_mom_vol_scaled, mean_reversion_regime, funding_squeeze_continuation, funding_carry_harvest]`
- **#1 Momentum** : |z-score| >= 1.0 (conviction gated), ~48h hold
- **#2 Funding squeeze** : z-score funding >= 2.0, ~96h hold, orthogonal rho ~ 0.04 vs #1 (validated OOS Sharpe +1.1)
- **#3 Carry delta-neutral** : funding >= 2 bps, STRUCTURALLY short-perp+long-spot (NEVER naked), validated OOS Sharpe +2.4..+4.5, orthogonal rho~0.05
- **Shadow (log-only)** : options IV DVOL, on-chain stablecoin growth

---

### 1.11 Clés de traçage dans les logs

Pour déboguer une boucle end-to-end :
- `market_tick_failed` : market_analyst crash
- `agent_node_failed` : researcher/trader/risk/pm crash
- `selective_override` : deterministic rule a overridé le LLM
- `trm_downgrade_to_hold` : TRM a dit HOLD
- `risk_reject` : hard gate (avec raison)
- `pm_executed` : trade_id, symbol, side, qty (success)
- `stop_watcher_closed` : N closes, stop/TP hit
- `reflect_failed` : reflexion post-close crash
- `pm_rejected_by_llm` : PM LLM a dit non
- `pm_place_order_failed` : exchange error

**Activity feed** (`/data/activity.json`) :
- Last 200 entries, ring buffer
- Chaque entry timestampée + kind (decision|pulse|error)
- Dashboard raffraîchit toutes les 20s

---

## Conclusion

Hermès est un **multi-agent system asynchrone** qui :
1. **Tick toutes les 5 min** : orchestre 7 agents en graphe déterministe pour chaque symbol en parallèle
2. **Fetch données** : OHLCV 4h + funding perp live (cohérent avec backtest venue)
3. **Compute signaux** : 4 skills deterministic (+ 2 shadow)
4. **Route déterministe** : selective_mode enforce le rule-validated, LLM devient advisory
5. **Gate risk** : RiskEngine rejects si equity/drawdown/kill-switch/position-cap/confidence, sinon size avec Kelly + ATR
6. **Execute en paper** : simule fills market avec slippage/fees
7. **Monitor stops** : toutes les 30s, ferme et réfléchit
8. **Pulse hourly** : non-LLM status, feed le dashboard
9. **Learn overnight** : nightly reflexion + weekly rerank skills

**Edge** : sélectivité (trades ONLY when signal armed) + orthogonalité (3 stratégies décorrélées) + discipline (no-pyramid, no-naked-carry, kill-switches).

---

## 2. Les stratégies & la couche de combinaison

### Architecture générale

La bibliothèque de stratégies du fonds Hermes repose sur deux couches d'orchestration:

1. **Couche L1 (Skills)** : Fonctions déterministes `OHLCV → SkillSignal` (ou cross-asset) — chaque skill émet un signal structuré (`side`, `confidence`, `stop/tp`) produit via des indicateurs figés a priori (pas d'optimisation).

2. **Couche L2/L3 (Combination)** : Routeur de régime avec hystérésis + blend inverse-vol + sélection de skills actifs — une couche purement advisory (aucun sizing) qui enrichit chaque signal avec `regime_routed`, `active`, `blend_weight` pour les agents aval.

Configuration centralisée : `/Users/lacarbonnerie/Developer/hermes-hedge-fund/config/default.yaml` (lignes 88–108).

---

### **Stratégie #1 : Momentum vol-scaled (`ts_mom_vol_scaled`)**

**Fichier** : `/Users/lacarbonnerie/Developer/hermes-hedge-fund/src/hermes/skills/ts_mom.py` (44–94)

**Thèse** : Le momentum (tendance) est l'une des anomalies les plus robustes et répliquées. Le piège est *choisir* le lookback ; la solution = moyenner un panel **FIXE** de 4 lookbacks géométriques : `{12, 24, 48, 96}` barres 4h.

**Calcul** :
- Pour chaque lookback `L` : `ret_L = log(close[-1]) - log(close[-1-L])` (rendement log sur L barres)
- Vol-scaled : `vol_L = std(trailing_L_returns) × √L` (volatilité réalisée horizonnée)
- Score IR par lookback : `score_L = ret_L / vol_L` (information ratio, scale-free)
- **Signal final** : `score = mean({score_12, score_24, score_48, score_96})`

**Seuil** : `threshold = ±0.1` (figé a priori)
- `score > +0.1` → **BUY**, confidence = `min(1.0, |score|)` (saturée)
- `score < -0.1` → **SELL**
- `-0.1 ≤ score ≤ +0.1` → **HOLD** (dead-band)

**Stop/TP** : ATR(14)×2.5 / 5.0 (figés a priori, standards TSMOM)

**Régime** : Trend-gated (roué dans `TREND_SKILLS` de la couche L2)

**Live depuis** : config ligne 100 (approved_skills)

---

### **Stratégie #2 : Funding Squeeze Continuation (`funding_squeeze_continuation`)**

**Fichier** : `/Users/lacarbonnerie/Developer/hermes-hedge-fund/src/hermes/research/signals.py` (237–260)

**Découverte** : Walk-forward ~9y systématique (648-backtest sweep) sur 6 features microstructurales × directions × horizons × 3 assets (BTC/ETH/SOL). **Résultat** : seul le funding a survécu le gate cohérent strict (net>0 ET in-sample>0 ET OOS>0 ET chaque fold positif) et généralisé cross-asset.

**Thèse inversée** : L'ancien hypothèse `funding_extreme_fade` s'est avérée **FAUSSE** (signe inversé + horizon trop court 48h). Lesson enregistrée (lignes 179–187 : documented negative result). Le **vrai** signal est la **continuation** : funding extrême (leveraged demand) marque une tendance PERSISTANTE qui continue ~96h.

**Calcul** :
- Fenêtre 180 barres ; z-score du funding_rate (column attachée live depuis Binance USDT-M)
- **Arm condition** : `funding_z ≥ +2.0` (queue positive extrême SEULEMENT — asymétrique, validated 3/3 assets)

**Signal** :
- `funding_z < 2.0` → **HOLD** (carry la gauge `funding_z` au tableau de bord même en HOLD)
- `funding_z ≥ 2.0` → **BUY** (long uniquement), `confidence = min(1.0, fz/4.0)`
- Stop/TP larges : 4.0 × ATR / 8.0 × ATR (l'edge validée est l'horizon ~24x4h = 96h, pas une barrière étroite)

**Tenir-jusqu'au-maximum** : 24 barres 4h = ~96h (ligne 107 de default.yaml)

**Performances OOS validées** :
- Per-trade Sharpe ann. +1.30, calendar-day +1.18, OOS-40% +1.37
- Corrélation PnL quotidien vs Stratégie #1 = +0.04 (vrai diversificateur orthogonal — ensemble brut doublerait OOS Sharpe vs #1 seule)

**Caveat enregistrée** (ligne 228–230) : l'edge est LONG-only, validée sur bull secular (crypto appreciation 2014–2024). À tester explicitement sub-période baissière avant live-sizing (bien que tous walk-forward folds, incl. 2018/2022 bears, étaient positifs).

**Régime** : Regime-INDEPENDENT (vit dans `ALWAYS_ON` de la couche L2, arme sur le funding_rate perp, exogène à momentum)

**Live depuis** : config ligne 100

---

### **Stratégie #3 : Funding Carry Harvest (Delta-neutre, `funding_carry_harvest`)**

**Fichier** : `/Users/lacarbonnerie/Developer/hermes-hedge-fund/src/hermes/research/signals.py` (279–296)

**Découverte** : Cash-and-carry classique, validée OOS Sharpe +2.4…+4.5, tous folds positifs 3/3 assets, orthogonal à **BOTH** #1 (rho~0.05) et #2 (rho~0.05-0.10), actif en marché calme (funding >0 ~80% du temps).

**CAVEAT CRITIQUE** : Fonctionne **UNIQUEMENT EN COUVERTURE DELTA-NEUTRE** (short perp + long spot appairé). Une validation ~9y a prouvé qu'en **naked perp short** (tout ce qu'une venue perp-only peut faire), l'edge est **OOS-NEGATIVE** — la perte prix en bull markets où funding est élevé anéantit le carry.

**Statut** : **DISPLAY/RESEARCH-ONLY** (pas en `allowed_skills`, ligne 100 config)
- Enregistrée au library (dashboard montre la gauge carry), mais **toujours retourne HOLD** (ligne 287 signals.py)
- `selective_gate` rejette tout skill hors slate → **JAMAIS de trade nu** (sûr par construction)
- Le paper exchange routerait structurellement cette skill_id vers le chemin delta-neutre deux-jambes (short perp + long spot), jamais un ordre nu — mais l'edge n'est pas encore active live

**Calcul** :
- Arm condition : `funding_rate × 1e4 ≥ 2.0 bps` (per 8h) — figé a priori
- Armed → **BUY**, `confidence = min(1.0, bps/10.0)`, sinon **HOLD**
- Stop/TP ultra-larges : 5% / 10% (horizon ~8 jours, carry dominant, pas barrière)

**Promotion** : Après implémentation de la jambe delta-neutre d'exécution (short perp + long spot hedge sur Hyperliquid + Binance spot).

---

### Couche L2/L3 : Combination & Routeur de régime

**Fichier** : `/Users/lacarbonnerie/Developer/hermes-hedge-fund/src/hermes/strategies/combination.py` (77–152)

#### **Routeur de régime avec hystérésis**

```python
route_regime(efficiency_ratio, prev_state, enter=0.4, exit=0.3) → "trend" | "range"
```

**Thèse (DeMiguel-Garlappi-Uppal)** : À faible data, 1/N et inverse-vol battent les allocateurs covariés OOS. Routeur trivial intentionnel : prévient flip-flop churn (= frais) via hystérésis.

**Paramètres** (figés a priori, conception spéc) :
- **enter = 0.4** : Kaufman ER ≥ 0.4 → bascule à "trend"
- **exit = 0.3** : ER ≤ 0.3 → bascule à "range"
- **Dead-band 0.3…0.4** : Conserve `prev_state` si dans la bande (pas de churn)

**Kaufman ER** (détecté dans `/Users/lacarbonnerie/Developer/hermes-hedge-fund/src/hermes/data/regime.py` lignes 20–33) :
```
ER = |net move over 48 bars| / sum(|bar moves|)
ER ≈ 1.0 → tendance propre, 0.0 → range choppy
```
Régime détecté mais utilisé pour l'agrégation LLM (agent `market_analyst`). La couche combination.py utilise l'ER fourni par l'appelant (le scheduler / pulse.py).

#### **Sélection de skills actifs**

```python
select_active_skills(regime, trend_skills, range_skills, always_on) → list[str]
```

**Slate LIVE** (combinaison.py lignes 36–38, pinned par test anti-drift ligne 121–135 test_combination.py) :

| Régime | Trend-gated | Range-gated | Always-on (regime-indep) |
|--------|-------------|-------------|--------------------------|
| **trend** | `ts_mom_vol_scaled` | — | `funding_squeeze_continuation`, `funding_carry_harvest` |
| **range** | — | `mean_reversion_regime` | `funding_squeeze_continuation`, `funding_carry_harvest` |

**Signification** :
- **Trend** : Momentum vol-scaled domine (tendance propre), range-skill down-weightée (off-régime = 0.5x confidence via regime_signal_weight)
- **Range** : Réversion à la moyenne règne, trend off-régime
- **Funding sleeves** : Orthogonales (arment sur funding_rate perp, exogène prix) → toujours actives, évaluées chaque tick

**Anti-drift test** (test_combination.py:121–135) : La slate advisory (TREND+RANGE+ALWAYS_ON) doit matcher **EXACTEMENT** la slate tradable (config `allowed_skills`). Cela a déjà attrapé une drift : ensemble ancien listait `breakout_vol_expansion` / `pairs_eth_btc` / `intraday_tod_btc` (refutées/pas en allowed_skills) et omettait les sleeves funding (#2 & #3).

#### **Inverse-vol blend**

```python
inverse_vol_weights(returns_by_skill: dict[str, list[float]]) → dict[str, float]
```

**Logique** : Chaque skill actif dont on a ≥2 observations reçoit poids ∝ `1/variance(returns)` (least volatile → most capital), normalisé Σ=1.

**Fallback robustes** :
- Variance ≤0 ou <2 observations → fallback **1/N égal** (plutôt qu'un blow-up ou division-par-zéro)
- Input vide → output vide

**Test validé** (test_combination.py:23–70) : les poids additionnent exactement 1.0, "calm" obtient plus poids que "wild", zero-variance/short-history chutent à égal.

#### **Annotation finale : `annotate_active_and_blend`**

```python
annotate_active_and_blend(
    signals: list[dict],
    efficiency_ratio: float,
    prev_state: str | None,
    returns_by_skill: dict[str, list[float]]
) → (regime: str, enriched_signals: list[dict])
```

**Chaque signal reçoit** (copie shallow enrichie) :
- `regime_routed` : str ("trend" | "range")
- `active` : bool (est-ce dans l'active-set pour ce régime ?)
- `blend_weight` : float ∈ [0, 1], Σ=1 sur tous les signaux

**Statut** : **ADVISORY SHADOW** en live
- Routage + pondération validés en backtest (CPCV/DSR)
- Enrichissent contexte pour agents aval (trader/risk-manager) avant décision
- Ne taillent **PAS** d'ordres eux-mêmes (le sizing engine fait ça)
- Quand pas de returns_by_skill utiles → blend_weight=0 (pas de blend live encore), mais routing/active toujours appliqués

**Test live** (test_combination.py:143–186) : regime hysteresis, poids normalisés, active marking correct.

---

### Sleeves SHADOW (Diversificateurs new-data)

**Fichier** : `/Users/lacarbonnerie/Developer/hermes-hedge-fund/src/hermes/skills/newdata_shadow.py` (1–123)

Deux sleeves **log-only** (registered au library, dashboard gauges, **0 capital risqué**) :

#### **#1 : `options_iv_dvol_reversion`** (BTC/ETH, ~24h)

**Edge** : Deribit DVOL 90d z-score (fear/complacency reversion)
- `z ≥ +2` → fear spike, oversold → **LONG** ~24h
- `z ≤ -2` → complacency → **SHORT**
- Rho vs momentum = **-0.26** (orthogonal négatif — diversificateur)

**Honnête re-audit (2026-06-24)** : Per-trade OOS Sharpe ~0.08–0.15 (pas les 1.04 vieux, base annualisée stale). **FAILS mandatory deflation** (multiple-testing Bonferroni ~1.0 over grid) → demoted from "confirmed" to "real-but-fragile". Needs deflation + date-level basket null.

**Statut** : SHADOW, size 0 (zéro capital). Promotion = beating null + deflation + beta-controlled date basket null.

#### **#2 : `onchain_stablecoin_growth`** (BTC/ETH, ~3d)

**Edge** : USDT+USDC 60d circulating-supply-growth z-score (dry powder indicator)
- `z ≥ +1` (supply expanding) → **LONG** ~3d
- `z ≤ -1` (contracting) → **SHORT**
- Rho vs funding = **~0** (orthogonal), vs momentum ~0 (diversificateur)

**Honnête re-audit (2026-06-24)** : Per-trade OOS Sharpe ~0.12–0.13 (pas les 1.07/1.21 vieux, base annualisée). **FAILS mandatory deflation** (Bonferroni ~1.0) → demoted. Basket-null borderline/failing under production ≥3d lag, beta-controlled.

**Statut** : SHADOW, size 0. Même promotion criteria.

**Discipline SHADOW prouvée** : (cf. `funding_carry_harvest`) Accumuler evidence OOS réelle, size 0, promouvoir après hold-forward.

---

### Intégration : Selective Gate (Risk Manager)

**Fichier** : `/Users/lacarbonnerie/Developer/hermes-hedge-fund/src/hermes/risk/selective.py` (47–88)

Quand `config.strategy.selective_mode = true` (PAPER-forward, ligne 94 default.yaml) :

```python
selective_gate(side, symbol, skill_id, skill_signals, open_keys, cfg) → (bool, str)
```

**Règles déterministes validées OOS** :
1. **Skill dans slate** : `skill_id ∈ allowed_skills` (ligne 70–71)
2. **Conviction-gated** : `ts_mom_vol_scaled` doit `confidence_raw ≥ 1.0` (ligne 73–83). Autres skills self-select à extremes.
3. **No-pyramid** : Un seul position par (symbol, skill_id) à la fois (ligne 85–86) — permet 2 sleeves coexister (Strategy #1 BTC + Strategy #2 BTC ≠ pyramiding)

**Entrée** : proposal(skill_id, side) + open_keys (positions ouvertes)
**Sortie** : (allowed, reason) — bloquerait une proposition momentum basse-conviction, mais laisserait funding continuer

Test : test_combination.py implicitly (blend_weight=0 si off-régime).

---

### Résumé architectural

```
┌─────────────────────────────────────────────┐
│ L0: OHLCV + funding_rate (Binance live)     │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ L1: Skills (deterministic, frozen params)  │
│  ├─ ts_mom_vol_scaled (trend, conviction)  │
│  ├─ mean_reversion_regime (range)          │
│  ├─ funding_squeeze_continuation (ALWAYS_ON)
│  ├─ funding_carry_harvest (ALWAYS_ON, HOLD-only)
│  └─ SHADOW: options_iv_dvol, onchain_stbl  │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ L2/L3: Combination layer (advisory)        │
│  ├─ Regime router (Kaufman ER, hyst 0.3/0.4)
│  ├─ Active-set selector (TREND_SKILLS ...) │
│  ├─ Inverse-vol blend (1/var weights)      │
│  └─ Signal enrichment (regime, active, wt) │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ L4: Risk Manager (selective_gate)           │
│  ├─ Skill-in-slate check                    │
│  ├─ Conviction floor (ts_mom only)          │
│  └─ No-pyramid (one per sleeve per symbol) │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ L5: Sizing engine (ATR-kelly, per-trade)   │
└─────────────────────────────────────────────┘
```

**Points-clés** :
- Slate **dual-régime** + sleeves **orthogonales** (funding) coexistent sans churn
- Anti-drift test (combination.py) force matching avec live allowed_skills
- Sleeves SHADOW accumulent evidence OOS, size 0 jusqu'à promotion
- Selective gate imposerait rule OOS-validée (~9y) en stage 0 (paper-only)

---

## 3. Exécution & gestion du risque

### 3.1 Moteur d'exécution paper exchange

Le **PaperExchange** (`src/hermes/execution/paper.py:36–406`) implémente l'échange simulé avec persistance d'état dans `/data/paper_state.json`. Trois composants clés :

#### 3.1.1 Modèle de trésorerie (cash accounting)

Ligne 1–17 : le `cash_usd` représente la **cash libre de l'opérateur**. Chaque trade débite/crédite la trésorerie selon la direction :
- **Long** (ligne 297–300) : débite `notional + fee` à l'ouverture
- **Short** (ligne 299–300) : crédite `notional - fee` à l'ouverture  
- À la fermeture, les PnL réalisés sont imputés au cash : `(prix_close - prix_entry) * qty - fee`

Pas de position synthétisée depuis les balances brutes ; l'état provient uniquement des trades enregistrés dans le book interne.

#### 3.1.2 Équité (equity_usd)

Ligne 129–146 : **`total = cash + Σ(signed_qty * mark_price)`** pour tous les legs ouverts.
- **Longs** (qty > 0) : `cash - |q|*entry + |q|*px = cash + |q|*(px-entry)`
- **Shorts** (qty < 0) : `cash + |q|*entry - |q|*px = cash + |q|*(entry-px)`  

Les **paires de carry delta-neutres** (short perp + long spot) contribution uniquement via `accrued_funding_usd` (ligne 144–145) — les jambes de prix s'annulent (delta ≈ 0).

#### 3.1.3 Paires de carry structurelles (Stratégie #3)

Ligne 188–264 : la **Stratégie #3 (funding_carry_harvest)** se route structurellement vers `_open_carry()` (jamais une jambe nue) :
- **Ouverture** (`_open_carry`, ligne 191–211) : SHORT perp + LONG spot à notionnel égal
  - Frais **MAKER bilatéraux** : `5 bps × 2 jambes = 10 bps` à l'ouverture
  - Débite le cash de la totalité des frais (`notional * 0.10%`)
  - Jambes de prix nettes ≈ 0 ; seuls les frais touchent le cash à l'ouverture
  - Enregistre : `entry_price`, `entry_funding_bps`, `accrued_funding_usd` (initialisé à 0)

- **Règlement de funding** (`settle_carry_funding`, ligne 213–224) : à chaque cycle 8h,
  - `funding_usd = notional × (funding_bps / 10,000)`
  - Crédite le short (reçoit funding positif), accumule dans `accrued_funding_usd`

- **Fermeture** (`_close_carry`, ligne 226–264) : atomiquement les deux jambes
  - Frais MAKER de fermeture : `5 bps × 2 = 10 bps`
  - PnL réalisé : `accrued_funding_usd - fees_close` (les jambes de prix s'annulent)
  - Si `accrued_funding_usd == 0`, tente une **réconciliation rétroactive** via `market.funding_history()` (ligne 246–249) — fallback gracieux en cas d'erreur

#### 3.1.4 Routage des ordres et idempotence

Ligne 267–371 (`place_order`) : entrée unique pour tous les ordres.
- **Client ID persistant** (`Order.client_id`, ligne 24 `base.py`) : `hermes-{uuid4()[:10]}`
- **Cloid sur Hyperliquid** (ligne 37–41 `hyperliquid.py`) : `derive_cloid()` = SHA256 du client_id → 0x{32 hex}
  - Déterministe : une relance du réseau avec le même `client_id` réutilise le cloid → échange déduplique au lieu d'ouvrir une 2e position
- **Paper mode** : pas de cloid natif ; l'échange simule fill immédiat à prix de marché

**Routing par skill_id** (ligne 270–271) : si `skill_id == CARRY_SKILL`, force via `_open_carry()` — jamais de naked order de carry.

#### 3.1.5 Concurrence et durabilité

Ligne 12–16 : **`asyncio.Lock()`** sérialise TOUS les appels `_state` :
- Tick marché 5-minute (boucle `market_tick`)
- Stop watcher 30-second (boucle `stop_watcher`)
- Lecture sans verrouillage : non disponible (L3 sûreté)

Écriture atomique : **tmp + `os.replace()`** (ligne 103–114) — un crash après un flush JSON incomplet ne tronque jamais l'état.

Réconciliation des closes : **FIFO par (symbol, skill_id)** (ligne 48–58 `stop_watcher.py`) — lorsqu'une position sur (symbol, skill) ferme, la première ligne journal ouverte pour cette paire est créditée du PnL.

---

### 3.2 Décision et gestion des risques (RiskEngine)

Fichiers : `src/hermes/risk/guardrails.py:94–286`, `sizing.py`.

#### 3.2.1 Barrières d'entrée (hard rejects)

Ligne 115–227 (`evaluate()`) : sequenciel rejet AVANT sizing :

| Condition | Seuil | Source |
|-----------|-------|--------|
| **Confiance** | `< confidence_floor` (défaut 0.0) | config |
| **Equity froide** | `< $5` | ligne 139 |
| **Positions ouvertes** | `>= max_open_positions` (défaut 3) | stage config |
| **Pyramiding** | position déjà ouverte sur (symbol, skill) | ligne 156–157 |
| **Kill-switch daily** | PnL réalisé ≤ `-daily_loss_kill_pct` d'equity | ligne 145–150 |
| **Notionnel min** | `< venue_min_notional` (Hyperliquid $10, paper $5) | ligne 196–199 |
| **Notionnel max** | `> min(5×equity, 0.5×equity×10)` = 5×equity | ligne 203–207 |
| **Spot shorting** | SELL sur stage_0/1/2 spot sans position longue existante | ligne 166–171 |
| **Paper mode** | stage_0 remet en papier si exchange ≠ paper | ligne 160–161 |

Kill-switch daily-loss : ligne 145–150
```python
kill_pct = stage_cfg.get("daily_loss_kill_pct", 2.0) / 100
if realized_pnl_today_usd / max(equity, 1e-9) <= -kill_pct:
    reject(f"daily loss kill triggered ...")
```

**Stage 0 (paper)** : `daily_loss_kill_pct = 5.0%` → kill à `-5% equity/jour`  
**Stage 1 (micro Hyperliquid)** : `daily_loss_kill_pct = 2.0%` → kill à `-2% equity/jour`

#### 3.2.2 Sizing (fractional Kelly + ATR-normalized)

Fichier `sizing.py:49–108` : `PositionSizer.size()`.

Inputs :
- `equity_usd` : equity actuelle
- `confidence` : 0–1 du PM agent
- `win_rate`, `avg_win_loss_ratio` : historique fermé
- `atr` : ATR(14) du marché
- `trades_in_journal` : compte fermé
- `stage_cap_pct` : cap par-trade (0.5% stage 0, 2% stages 1–2)
- `correlation_haircut` : réduction pour positions corrélées (BTC/ETH/SOL rho ~0.85)
- `vol_scale` : multiplicateur vol-targeting (1.0 = neutral)

**Algorithme** (ligne 49–108) :

1. **Stop distance ATR-normalisée** (ligne 52–55) :
   ```python
   atr_pct = ATR / price; si NaN/0 → atr_pct = 0
   stop_dist_pct = max(0.3%, 2.0 * atr_pct)  # 2.0x ATR, floor 0.3%
   ```

2. **Contribution Kelly** (ligne 58–67) : uniquement après 100 trades fermés
   ```python
   si trades >= 100 et win_rate/ratio disponibles :
       kelly = max(0, (p*b - q) / b)
   sinon :
       kelly = 0
   ```

3. **Risk synthétisé** (ligne 69–87) :
   ```python
   floor_frac = 0.25% (per_trade_floor_pct)
   stage_cap_frac = stage_cap_pct (0.5%/2%)
   
   si kelly > 0 :
       raw = 0.20 * kelly * confidence
   sinon (cold-start) :
       raw = floor + (cap - floor) * confidence * 0.5
   
   risk_pct = clamp(floor, cap, raw)
   risk_pct *= correlation_haircut * vol_scale
   ```

4. **Notional & qty** (ligne 89–93) :
   ```python
   risk_usd = risk_pct * equity
   notional = risk_usd / stop_dist_pct
   qty = notional / price
   ```

**Exemple Stage 0** : equity $100, confidence 0.7, ATR 500 (BTC ~60k) :
- `stop_dist_pct = max(0.3%, 2 × 500/60000) = 1.67%`
- Cold-start, kelly=0 : `raw = 0.25% + (0.5% - 0.25%) × 0.7 × 0.5 = 0.288%`
- `risk_usd = 0.288% × $100 = $0.288`
- `notional = $0.288 / 1.67% = $17.24` (< Hyperliquid $10 min ✓)

#### 3.2.3 Frais & minima de notionnel par venue

Configuration `config/default.yaml:110–119` :

| Instrument | Taker | Maker | Slippage |
|-----------|-------|-------|----------|
| Spot | 7 bps | 4 bps | 3 bps (paper) |
| Perp | 4.5 bps | 1.5 bps | — |

Minima notionnel (ligne 41 `guardrails.py`) :
- **Paper** : $5
- **Hyperliquid** : $10 (l'échange rejette < $10)
- **Bybit** : $5

Frais paper (ligne 42–43 `paper.py`) : taker par défaut 7 bps (spot Hyperliquid), slippage 3 bps.  
Carry maker : 5 bps/jambe × 2 = **10 bps ouverture, 10 bps fermeture = 20 bps round-trip pair** (ligne 189).

#### 3.2.4 Stops natifs & temps de hold max

**Stops natifs sur Hyperliquid** (ligne 44–65 `hyperliquid.py:build_tpsl_orders()`) :
- Après `place_order()`, attach réduction-seule triggers (SL/TP) sur l'exchange
- Position protégée même si l'agent/LLM/Tailscale tombe

**Paper mode server-side** (ligne 229–270 `guardrails.py:check_stops()`) :
- Stop hit : `px <= stop_price` (long), `px >= stop_price` (short)
- TP hit : `px >= take_profit` (long), `px <= take_profit` (short)
- Max hold vertical barrier (ligne 264–269) :
  - Config `strategy.max_hold_bars_by_skill` : Strategy #1 (momentum) ~12 bars/48h, Strategy #2 (funding) ~24 bars/96h, Strategy #3 (carry) ~48 bars/8 jours
  - Timeexit via `hermes.risk.timeexit:is_past_max_hold()` (ligne 40–45)

#### 3.2.5 Gating sélectif (OOS-validated rule enforcement)

Fichier `src/hermes/risk/selective.py:47–88` : avant sizing, applique le rule déterministe validé.

- **Compétences autorisées** : `allowed_skills = [ts_mom_vol_scaled, mean_reversion_regime, funding_squeeze_continuation, funding_carry_harvest]`
- **Conviction floor** (conviction-gated) : `ts_mom_vol_scaled` doit `|score| >= 1.0` ; les autres (funding, reversion) self-select
- **No-pyramid** : une position max par (symbol, skill) — permet coexistence sleeve : momentum + funding sur BTC concurrente

---

### 3.3 Flux exécution complèt : place_order → journal

1. **PM approuve** (`portfolio_manager.py:66–119`) : exécute `exchange.place_order(order)` → `Fill` (symbol, qty, price, fee_usd, ts)
2. **Journal ouvert** (ligne 102–114) : `TradeJournal.open(JournalEntry)` → trade_id enregistré dans state
3. **Stop watcher** (30s loop, `stop_watcher.py:30–76`) :
   - `check_stops()` → détecte stop/TP/max-hold hits
   - `close_position(symbol, reason, skill_id)` → `Fill` de fermeture
   - PnL = `(close_px - entry_px) * qty - close_fee` (open_fee déjà débité)
   - **FIFO match** : première ligne journal ouverte sur (symbol, skill) reçoit le PnL, marquée `closed_at`
4. **Réconciliation** : `JournalRow.pnl_usd`, `closed_at`, `exit_reason` immuables après premier close (ligne 69–75 `journal.py:close()`)

---

### 3.4 Résumé exécution-risque

| Aspect | Implémentation |
|--------|-----------------|
| **Venue données/prix** | Binance USDT-M futures (ccxt MarketData, `data/market.py`) |
| **Taker round-trip** | 7 bps spot + 3 bps slippage = ~10 bps RT (paper) ; 4.5 bps perp |
| **Maker round-trip** | 4 bps spot + 0 slip = 4 bps ; 1.5 bps perp |
| **Carry RT pair** | 10 bps (5 bps × 2 jambes) ouverture + 10 bps fermeture = 20 bps RT validés |
| **Kill-switch** | Daily PnL ≤ `-kill_pct × equity` → HOLD forcé |
| **Sizing** | ATR-normalized (min 0.3%), Kelly + confidence floor, correlation haircut, vol-targeting |
| **Idempotence** | Client ID persistant → cloid SHA256 → dedup exchange-side retry |
| **Stops natifs** | Hyperliquid trigger, paper server-side ; max-hold vertical barrier per-skill |
| **Réconciliation** | FIFO (symbol, skill) → journal match ; PnL immuable après close |

---

## 4. Le cerveau (LLM, agents, TRM, mémoire) & le dashboard

### 4.1 Routeur LLM — dérive gracieuse et adaptation de schémas JSON

**Fichier:** `src/hermes/llm/router.py`

Le `LLMRouter` est l'unique point d'entrée aux LLM pour tous les agents. Il implémente une stratégie **primaire → fallback cloud** :

- **Cible primaire:** DeepSeek V4 (Pro/Flash/Escalation, OpenAI-compatible) activée si `DEEPSEEK_API_KEY` est défini  
  - `deepseek-v4-pro` (primary, escalation)  
  - `deepseek-v4-flash` (secondary, rapide)  
- **Fallback local:** llama.cpp sur la machine (`llm_primary_base_url`, modèle local) si DeepSeek est injoignable  
- **Retries transients:** 3 tentatives avec backoff exponentiel (1–8s) sur `APIConnectionError`, `APITimeoutError`, `RateLimitError` (lignes 40–47); les 4xx (auth invalide, modèle introuvable) échouent **immédiatement** sans retry pour éviter de brûler le budget API

**Adaptation de schémas JSON** (`structured()`, lignes 219–265):
Dégradation gracieuse à 3 niveaux, testé notamment pour contourner le rejet 400 de DeepSeek sur `json_schema` :
1. Essai strict `json_schema` avec `strict: True` (schéma nommé `hermes_output`)  
2. Fallback à `json_object` (type supporté par DeepSeek)  
3. Ultime fallback **pas de `response_format`** + extraction regex du premier bloc `{...}` du contenu brut

La résilience est garantie : un appel `structured()` ne hard-fail jamais; le dernier bloc JSON extractible (ou `{}` vide) est retourné (ligne 264).

**Cibles par rôle agent** (ligne 135–140): Le routeur accepte des cibles par agent role via `config/agents` YAML; les rôles non configurés dégradent vers `"primary"`. Cela permet aux testeurs de router différents agents vers des modèles différents.

### 4.2 Agents — orchestration LangGraph-style

**Fichier:** `src/hermes/graph/orchestrator.py`, agents dans `src/hermes/agents/`

**Flux linéaire parallélisé** (orchestrator.py:4–6):
```
market_analyst  ┐
news_analyst    ├─→  fetch_lessons  →  bull ∥ bear  →  trader  →  risk_manager  →  portfolio_manager
sentiment       ┘
```

Les trois analystes de marché/news/sentiment (lignes 31–35) tournent **en parallèle** (`asyncio.gather`); la fetch des lessons dépend de market_analyst pour `state.snapshot`, puis bull/bear cherchent les leçons similaires via Chroma, enfin trader/risk/PM séquentiels.

**Agents clés:**

1. **Market Analyst** (`market_analyst.py:24–100`):
   - OHLCV 300-bar + ATR 14-période, détection de régime (breakout efficiency ratio)
   - Attache funding perps (if disponible) et features new-data shadow (IV z-score, on-chain stablecoin z)
   - Évalue toutes les skills via `get_skill_library().evaluate_for_symbol()`; pondère les signaux par `regime_signal_weight()` — une skill hors régime (ex. mean-reversion en trend) voit son confidence multiplié par w<1.0  
   - **Selective gate:** route + tag les skills actives avec poids blend inverse-vol (live blend=0 jusqu'à plumbage du journal par-skill)

2. **Researcher — Bull & Bear** (`researcher.py:35–81`):
   - Charge leçons pertinentes via `LessonsStore.search(query, k=6)` (requête vectorielle Chroma: symbole + timeframe + ATR % + régime)
   - `bull()`: structured JSON with `{"thesis", "skill_id", "confidence", "counter_risk"}`
   - `bear()`: idem, avec `"side": enum["SELL", "HOLD"]`
   - Contexte: snapshot, news headlines, sentiment, skill_signals (pondérés), relevant_lessons

3. **Trader** (`trader.py:13–52`):
   - Tool call `propose_trade` avec arguments `{"symbol", "side", "thesis", "confidence", "stop_pct", "take_profit_pct", "skill_id"}`
   - Fallback HOLD si aucun outil n'est appelé (robustesse contre parse failures)
   - Contexte: bull/bear views + lessons + skill_signals

4. **Risk Manager** (`risk_manager.py:28–208`):
   - **Selective mode** (si `sel_cfg.enabled`): déterministe `drive_signal()` OVERRIDE la proposition du trader; aucune pyramiding par sleeve via clés `(symbol, skill_id)`
   - **TRM verifier** (lignes 83–136): encode OHLCV en grille d'état 256 tokens + encode action LLM en 4 tokens action, passe au TRM pour récursion; confidence = sigmoid(q_halt_logits)
     - TRM ne peut DOWNGRADER à HOLD que si (`used_trm=True` AND `refined.side="HOLD"` AND `confidence>0.7`), jamais flipper BUY↔SELL
   - **Confidence calibration** (lignes 138–158): isotonique fitted sur paires (confidence, 1/0) de trades clos; transforme la confiance brute LLM → P(win) vraie; seul baisse (identity function si < min fitted trades)
   - **Vol targeting** (lignes 160–175): HAR-RV forecast vs target vol → scale du risk budget
   - **Meta-labeling** (lignes 177–187, shadow/log-only): fitted `sklearn.isotonic.IsotonicRegression` sur `meta_features(snapshot, raw_conf)` → p_win prédite; JAMAIS gate une trade
   - **Risk engine déterministe** (lignes 189–207): confidence floor, position limits, no-pyramiding, Kelly sizing, ordre avec stops/TP

5. **Portfolio Manager** (`portfolio_manager.py:39–119`):
   - **Failsafe:** si risk engine rejette, PM rejettera aussi
   - **LLM final reviewer**: demande au PM LLM (role=portfolio_manager, schema strict APPROVED/REJECTED) de veto; LLM CAN downgrader mais NEVER upgrader
   - **Exécution & journaling:** `exchange.place_order()` → `TradeJournal.open(JournalEntry)` avec skill_id, thesis, confidence, snapshot marché

**État partagé** (`src/hermes/graph/state.py`):
```python
@dataclass
class HermesState:
    symbol, timeframe, ohlcv, snapshot, regime, news, sentiment,
    skill_signals, relevant_lessons,                    # analystes
    bull_view, bear_view,                               # researchers
    trader_proposal,                                    # trader
    risk_decision, trm_refinement,                      # risk_manager
    confidence_calibration, vol_forecast, meta_label,   # models
    executed, trade_id, error                           # outcome
```

### 4.3 TRM Verifier — Samsung-SAIL, confiance probabiliste

**Fichier:** `src/hermes/trm/verifier.py`, encoder en `src/hermes/trm/encoder.py`

**Architecture:**
- **TRMVerifier (real)** ou **NullTRMVerifier (fallback)**: gradation gracieuse si torch absent, source non importée, ou checkpoint manquant
- **Checkpoint:** `.pt` file loadé une fois (`lru_cache`); chemin via `settings.trm_checkpoint`
- **Modèle:** `TinyRecursiveReasoningModel_ACTV1` de Samsung-SAIL (vendored via `PYTHONPATH` dans Dockerfile, cloner `SamsungSAILMontreal/TinyRecursiveModels` en `/opt/trm`)

**Encoding d'état** (encoder.py:46–141):
- **Vocab disjoint:** PAD=0, SIDE (1–12), SIZE (13–28), STOP (29–60), TP (61–92), STATE (96–159, 64 bins)
- **8 channels × 32 bars = 256 tokens état:** ret_1 (1-bar log), ret_log (20-bar cumulé), vol_z (ATR/price z-score), RSI_14, EMA fast/slow diff, vol_ratio, regime_bit (breakout flag)
- **Quantiles fixes:** largeur rolling 200-bar pour stabiliser les frontières bin d'un tick au suivant

**Inference** (verifier.py:123–156):
- Concatène état + action draft (4 tokens) → `seq_len=260`
- Boucle `halt_max_steps` (défaut 6), forward récursif avec carry jusqu'à convergence ou max_steps
- Extrait `action_logits[state_len:]` (derniers 4 tokens) → `argmax` → refined action
- **Confiance:** `sigmoid(q_halt_logits)` du modèle; seuil `confidence > 0.7` requis pour downgrader vers HOLD
- Si TRM est NullVerifier (fallback), retourne `confidence=1.0` (aucune dégradation), `used_trm=False`

**Prise de décision** (risk_manager.py:111–126):
- TRM agit comme **confidence CEILING** lorsqu'il est réel: `proposal.confidence = min(proposal.confidence, verdict.confidence)`
- Downgrader vers HOLD est **asymétrique:** uniquement si refined_side="HOLD" et confidence>0.7
- Flipper BUY↔SELL ou remonter HOLD est **jamais autorisé** — ces choix restent au LLM

### 4.4 Couche mémoire & learning

**Trade Journal** (`src/hermes/memory/journal.py`):
- SQLite async via SQLAlchemy 2.0, table `trade_journal` avec full détail: symbol, side, qty, entry_price, confidence, skill_id, thesis, market_state (JSON), pnl_usd, exit_reason, reflection
- `TradeJournal.open(entry)` INSERT + commit, retourne trade_id
- `TradeJournal.close(trade_id, exit_price, pnl_usd, fee_usd, exit_reason, reflection)` UPDATE; idempotent (déjà fermé = no-op sauf si reflection non vide)
- `stats()` agrégation: win_rate, avg_win_loss_ratio, pnl total sur trades clos
- `confidence_outcomes()` retourne paires (confidence, 1/0 win) pour calibration

**Vector Store & Lessons** (`src/hermes/memory/vector.py`, `src/hermes/memory/lessons.py`):
- Chroma DB (PersistentClient) avec collection "hermes_lessons", embedding MiniLM ONNX défaut
- `LessonsStore.add(text, tags, source_trade_ids)`: **write-order critique** — vecteur D'ABORD (abort si failed), puis SQL row; garantit qu'aucune leçon SQL n'est orpheline
- `LessonsStore.search(query, k=6)`: requête sémantique → `[{"id", "text", "meta", "distance"}]`
- Lessons récupérées par researcher via query `f"{symbol} {timeframe} regime atr {atr_pct:.4f}"`

**Calibration de confiance** (`src/hermes/risk/calibration.py`):
- Isotonic regression `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` fitted sur `confidence_outcomes()` des trades clos
- `calibrate(raw_conf) → P(win)_honest` ; ne baisse jamais (monotone croissant)
- Identity si < min trades fermés ou pas encore fitted

**Meta-labeler** (`src/hermes/risk/meta_labeling.py`):
- Idem isotonic, sur `meta_features(snapshot, raw_conf)` = (price, atr_pct, ret_1, vol_avg_20, raw_conf, regime_int, ...)
- Prédiction: `p_win` bayésienne; **log-only, jamais gate** (shadow)
- Fitted progressivement sur les 2-class trades

### 4.5 Endpoint `/chat` — read-only, contextué

**Fichier:** `src/hermes/api.py:335–356`

Endpoint `POST /chat` (protégé par `x-api-key`, optionnel si pas configuré):
- **Système:** Hermès français, strict: discussion/explication seulement, **AUCUNE action** (no trades, no config, no fund moves)
- **Contexte live** (lignes 288–332):
  - Equity + bankroll + universe
  - Stratégies A/B: labels, status (live/wired/validated), n trades, PnL, win_rate
  - Positions ouvertes: symbol/skill/qty/unrealized PnL
  - **Pulses horaires:** jauges par sleeve (conviction, funding z, funding bps, DVOL z, stablecoin z) avec seuils d'arme
- **Messages:** historique limité à 10 derniers (window taille < 4000 chars par message pour safety)
- **Température:** 0.4 (baisse créativité, rehausse cohérence)
- **Fallback:** si LLM indisponible, retour erreur 502 avec cause
- **Réponse:** `{"reply": str, "model": str}`

Endpoint `GET /chat` retourne l'UI HTML (forme simple: textarea + scroll log, stockage local de clé API)

### 4.6 Dashboard — observations read-only, sans LLM

**Fichier:** `src/hermes/api.py:126–178`, HTML inline lignes 400–681

Endpoint `GET /dashboard/data` (JSON, read-only, pas d'auth):
- **Stratégies A/B** (`_strategies_ab()`, lignes 226–274): 3 sleeves affichées côte-à-côte
  - Strategy #1 (Momentum sélectif): `ts_mom_vol_scaled`, breakout, mean-reversion legacy → live/wired
  - Strategy #2 (Funding squeeze): `funding_squeeze_continuation` (OOS Sharpe +1.1, rho~0.04 vs #1) → live si trades clos, else "wired"
  - Strategy #3 (Carry delta-neutral): `funding_carry_harvest` (OOS Sharpe +2.4…+4.5, rho~0.05 vs 1&2) → live si pair ouverte ou close, validé (jamais nude naked perp, 9y test perdant)
  - Per-sleeve: `{"n": n, "pnl_usd": $, "win_rate": %}` sur closed trades
  - Research annex: OOS validated bricks, candidates tested, note

- **Equity curve** (`_equity_curve()`, lignes 202–212): cumulative realized PnL depuis la startup, pts=[{"t": closed_at iso, "eq": $cumul}], depuis bankroll init

- **Activity feed** (`_recent_activity()`, lignes 181–185): last 30 events from `data_dir/activity.json`, timestamps ISO

- **Gauges par stratégie** (livepulse): affichées progressively sur le dashboard `#live` à droite de chaque symbol
  - Momentum #1: conviction (raw score) vs seuil 1.0, barre progress
  - Funding #2: funding z vs seuil 2.0
  - Carry #3: funding bps vs arming threshold (defaut 2 bps)
  - Statut: ARMED (vert #a6e3a1) ou "holding" (bleu #89b4fa)

- **Shadow sleeves** (`#shadow`): log-only, jamais tradées, esthétique distincte (#cba6f7 violet)
  - Options IV—DVOL reversion: DVOL z gauge
  - On-chain—stablecoin dry-powder: supply growth z gauge

**Hourly pulse (LLM-free)** (`src/hermes/runtime/pulse.py:92–140`):
- Horaire indépendant de la décision 4h, aucun LLM, aucun ordre
- Par symbol: `df = ohlcv(4h) + funding + dvol_z + stbl_z`
- Signaux skills: bruts (non LLM) + évaluation sélective gate → `pulse_entry()` JSON
- Gauges (UpBar dans le JSON): jauges brutes par sleeve ([sleeve, metric, val, thr, armed, (shadow=bool)])
- Stocké dans `ActivityLog` anneau circulaire (buffer 200 derniers events)

**Dashboard HTML** (lignes 400–681):
- Auto-refresh 20s fetch `/dashboard/data`
- Grille responsive (3 cols min 320px)
- Cards:
  - **Live** — what each strategy sees right now (gauges barre progress, armed/holding, next decision ETA)
  - **Shadow** — research sleeves, log-only
  - **Equity** — paper $, bankroll
  - **Stage gate** — dépôts et progression vers live
  - **Journal** — closed n, win %, avg W/L, PnL réalisé
  - **Models** — calibrator fitted ?, meta_labeler fitted ?
  - **Open positions** — tableau symbol/qty/entry/uPnL/stop/TP/skill
  - **Equity curve** — SVG line chart, start→now, n trades
  - **Strategies A/B** — 3 colonnes: label, live status, n/win/PnL, research annex
  - **Activity feed** — filtrable (text), toggle hide HOLD pulses, kind/symbol/regime/signal/reason
  - **Breakdown** — by_symbol & by_skill: n trades, win %, PnL (piste SOL lean)
  - **Recent closed trades** — rhythm: when, symbol, side, PnL, reason, skill, conf
  - **Work cycle** — scheduled jobs (market_tick, nightly reflection, weekly rerank) + next_run ETA

**Couleurs:** Catppuccin dark (mocha):
- Bg: #0b0e14, cards #11151f, border #1d2230
- Text: #cdd6f4, sub #7f8aa3, muted #6c7693
- Pos: #a6e3a1, neg: #f38ba8
- Highlight: #89b4fa, violet #cba6f7

### 4.7 Synoptique cerveau ↔ observabilité

```
┌─ DÉCISION (tick 4h, LLM-enabled) ─────────────────────────────┐
│                                                                │
│  market_analyst | news | sentiment  ──┐                       │
│                                       ├─ fetch_lessons ────┐  │
│                                   ┌──┴─ bull ∥ bear      │  │
│                                   │    ├─ trader         │  │
│                                   │    ├─ risk_manager   │  │
│                                   │    └─ portfolio_mgr  │  │
│  OUTPUT: HermesState:                                     │  │
│    ├─ trader_proposal (side, conf, stop%, tp%)           │  │
│    ├─ trm_refinement (refined_side, confidence_ceil)     │  │
│    ├─ confidence_calibration (raw → honest P(win))       │  │
│    ├─ meta_label (shadow p_win)                          │  │
│    ├─ selective_gate verdict (deterministic override)    │  │
│    └─ executed, trade_id, risk_decision reason           │  │
│                                                           │  │
│  → TradeJournal.open(symbol, side, qty, entry_price,    │  │
│                     confidence, skill_id, thesis, ...)   │  │
│                                                           │  │
└────────────────────────────────────────────────────────────┘
          │
          ├─→ ActivityLog.record({kind:'trade', executed, decision, ...})
          │
          ├─→ LessonsStore.add() [nightly reflection]
          │
          └─→ SkillStat.update() [weekly rerank]

┌─ OBSERVABILITÉ (continu, sans LLM) ───────────────────────┐
│                                                            │
│  Hourly pulse() — LLM-free heartbeat:                    │
│    ├─ per-symbol: regime, efficiency_ratio              │
│    ├─ skill_signals: raw confidence + features         │
│    ├─ gauges: [sleeve, metric, val, threshold, armed]  │
│    └─ would_act: sélective gate verdict                │
│                                                            │
│  → ActivityLog.record({kind:'pulse', gauges, would_act})  │
│                                                            │
│  /dashboard/data JSON:                                     │
│    ├─ equity_usd, stage, journal stats                   │
│    ├─ strategies A/B: live vs research                   │
│    ├─ equity_curve: PnL cumulé                          │
│    ├─ activity: 30 events (trades + pulses)             │
│    ├─ breakdown: by_symbol, by_skill PnL/win           │
│    └─ scheduler: next_run times                          │
│                                                            │
│  /dashboard HTML UI (auto 20s refresh):                   │
│    ├─ Live gauges: barre progress ARMED vs holding      │
│    ├─ Shadow sleeves: log-only, distinctes              │
│    ├─ A/B stratégies: colonnes côte-à-côte             │
│    └─ Activity table: why trade/hold, filtre symbol    │
│                                                            │
│  /chat (read-only, français):                            │
│    ├─ contexte live: équité, positions, strats         │
│    ├─ système: Hermès, no trading via this endpoint    │
│    └─ réponse: discussion + explication seule          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 4.8 Résilience & failsafes

| Couche | Mécanisme |
|--------|-----------|
| **LLM router** | Fallback local DeepSeek↔llama.cpp; dégradation json_schema→json_object→none |
| **TRM verifier** | NullTRMVerifier (confidence=1.0) si torch/checkpoint absent; jamais hard-fail |
| **Confidence calibration** | Identity function (no-op) si < min trades fitted; seul baisse (monotone) |
| **Selective gate** | Déterministe, override LLM, forbid pyramiding |
| **Risk engine** | Rejette si confidence<floor (post-TRM, post-calibration) |
| **Portfolio manager** | Failsafe: risk=REJECTED → PM=REJECTED toujours; LLM CAN veto, NEVER upgrade |
| **Pulse** | LLM-free, bruit-tolerant; meurt proprement sur any error |
| **Activity log** | Anneau circulaire 200 entrées, atomique (tmpfile + replace) |
| **Lessons vector write** | Abort tout si vecteur fails; aucun orphelin SQL |

---

## 5. La machine d'honnêteté (validation & backtest)

### Mandat & philosophie

La machine d'honnêteté valide que chaque sleeve (stratégie multi-instruments ou signal unique) possède une arête **véritable** et non du bruit de surapprentissage ou des biais de backtesting classiques. C'est l'organe central du projet : plutôt que de juger chaque sleeve isolément par son Sharpe solo, le projet évalue chaque sleeve par sa **contribution** au livre combiné (leave-one-out Δ-Sharpe), puis exécute une batterie de tests adversariels qui capturent les trois pièges majeurs d'une industrie financière obsédée par le cherry-picking.

### Architecture générale

```
sleeve_returns (dict[str, pd.Series])
        ↓
    align_sleeves() : union grid, fill 0.0
        ↓
    sleeve pures + combined book series
        ↓
    causal_inverse_vol_weights() : forward weights only
        ↓
    combine_book() : weighted series (inv_vol ou equal)
        ↓
    leave_one_out_delta_sharpe() : per-sleeve contribution
        ↓
    GATE (CPCV/PBO/DSR) → verdict
```

### 1. Causal Book Combination (src/hermes/validation/book.py)

#### `align_sleeves()` - Grille commune, honnête par construction

**Code : book.py:47-59**

```python
def align_sleeves(sleeve_returns: dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame(sleeve_returns)
    frame = frame.sort_index()
    return frame.fillna(0.0)
```

Les sleeves tradent à des cadences différentes (COT hebdomadaire, GVZ quotidien, breakout intraday). La solution naïve — interpoler, forward-fill — **crée une fuite forward**. Hermes construit au lieu :
- **Union chronologique** de tous les marks de tous les sleeves.
- **Période sans mark = flat 0.0** : la sleeve ne risque rien, n'earning rien, n'interfère pas avec les autres.

Cela garantit un livre que le trader aurait pu réellement exécuter à l'open de chaque période, sans donnée du futur.

#### `causal_inverse_vol_weights()` - Poids Forward, Zéro Lookahead

**Code : book.py:62-91**

```python
def causal_inverse_vol_weights(
    aligned: pd.DataFrame, min_periods: int = 8
) -> pd.DataFrame:
    """Expanding inverse-variance weights, set from data strictly before each period."""
    for t in range(n_periods):
        past = arr[:t]  # strictly before t — the causality guarantee
        if past.shape[0] < min_periods:
            weights[t] = equal
            continue
        var = past.var(axis=0, ddof=1)
        inv = 1.0 / var
        weights[t] = inv / inv.sum()
```

**Le piège classique** : si tu pondères le livre avec variance *totale* sur la période, tu fuis dans l'avenir et tu gonfles ton Sharpe.

**La solution Hermes** : 
- Poids en *t* basé **uniquement** sur variance de *t-1, t-2, ... vers le passé*.
- Jusqu'à `min_periods` observations : fallback égal-poids (DeMiguel-Garlappi-Uppal robustness).
- Résultat : le livre est **véritablement causal**, exécutable à l'ouverture.

#### `combine_book()` - Série pondérée (inv_vol / equal)

**Code : book.py:94-117**

```python
def combine_book(
    sleeve_returns: dict[str, pd.Series],
    weighting: str = "inv_vol",
    min_periods: int = 8,
) -> pd.Series:
    aligned = align_sleeves(sleeve_returns)
    if weighting == "inv_vol":
        w = causal_inverse_vol_weights(aligned, min_periods=min_periods)
        return (aligned * w).sum(axis=1)
```

Produit une série unique `ret_book[t] = Σ_s w_{s,t} · r_{s,t}`, où chaque weight est causal.

#### `leave_one_out_delta_sharpe()` - Critère d'acceptation central

**Code : book.py:120-143**

```python
def leave_one_out_delta_sharpe(
    sleeve_returns: dict[str, pd.Series],
    weighting: str = "inv_vol",
    min_periods: int = 8,
) -> dict[str, float]:
    """Δ_s = Sharpe(full book) - Sharpe(book rebuilt WITHOUT sleeve s)."""
    ids = list(sleeve_returns)
    full = sharpe_ratio(combine_book(sleeve_returns, weighting, min_periods).tolist())
    out: dict[str, float] = {}
    for sid in ids:
        rest = {k: v for k, v in sleeve_returns.items() if k != sid}
        without = sharpe_ratio(combine_book(rest, weighting, min_periods).tolist())
        out[sid] = full - without
```

**Philosophie centrale (§7 Renaissance)** :
- Un sleeve solo avec Sharpe 0.2 mais ρ ≈ 0 au reste ajoute énormément au livre combiné.
- Un sleeve solo bon mais très corrélé peut être redondant ou destructeur.
- Le **vrai test** : enlever cette sleeve rend-il le livre *moins bon* ?
- Si Δ_s > 0 → la sleeve améliore l'ensemble, elle est acceptée.
- Si Δ_s ≤ 0 → redondante ou loss-making, rejetée.

#### `orthogonality_matrix()` - Rho pairwise

**Code : book.py:146-157**

```python
def orthogonality_matrix(sleeve_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise Pearson rho of the aligned per-sleeve return streams."""
    aligned = align_sleeves(sleeve_returns)
    return aligned.corr()
```

Chaque nouvelle sleeve doit montrer |ρ| < 0.3 contre chaque sleeve existante pour être "vraiment nouvelle information, pas du beta rechapé".

---

### 2. Triple-Barrier Labeling (src/hermes/validation/triple_barrier.py)

**Code : triple_barrier.py:14-44**

```python
def triple_barrier_label(
    highs, lows, closes, entry_idx, entry_px, side,
    stop_pct, tp_pct, horizon
) -> tuple[int, float]:
    """Scan up to horizon bars: TP hit -> win (1), SL hit -> loss (0), else sign(ret)."""
    if stop_hit:  # stop has precedence (conservative)
        return 0, -stop_pct
    if tp_hit:
        return 1, tp_pct
    final = closes[end]
    ret = (final - entry_px) / entry_px
    return (1 if ret > 0 else 0), float(ret)
```

**Honnêteté** :
- Quand high et low de la même barre englobent SL *et* TP, **le stop est appliqué d'abord** (hypothèse conservatrice : l'adverse move survient avant le favorable).
- Évite le biais classique "je ne vois que mes TP d'une barre volatilité élevée".

---

### 3. FXGate : realisation et tests adversariels (research_fx/fxgate.py)

#### `realize_event_returns()` - No-overlap, honest cost

**Code : fxgate.py:34-76**

```python
def realize_event_returns(
    close, ts, events, horizon: int, cost_rt: float,
    two_leg: bool = False, warmup: int = 0,
):
    """Non-overlapping trades: once entered at bar i, blocks new entry until i + horizon.
    A run of identical signals becomes ONE held position (pays cost once)."""
    close = np.asarray(close, dtype=float)
    cost = cost_rt * (2.0 if two_leg else 1.0)
    busy_until = -1
    for i, side in sorted(events, key=lambda e: e[0]):
        if i < busy_until:
            continue
        h = min(horizon, n - 1 - i)
        gross = float(side) * (close[i + h] / close[i] - 1.0)
        out.append({
            "ret_gross": float(gross),
            "ret_net": float(gross - cost),
            ...
        })
        busy_until = i + horizon
```

**Piège attrapé : overlapping trades** :
- Sans le `busy_until`, une persistance de signal (z >= threshold, jour après jour) se compte comme N trades indépendants.
- Résultat : variance artificellement comprimée, p-values abusives.
- Ici : UN seul trade (une position tenue) sur la persistance, dès l'ouverture, coût payé *une fois*.

Cost par actif (book.py:16) :
- Or/FX majors ≈ 1.6 bps (causal, cotation réelle).
- Crosses ≈ 2.1 bps.
- Argent ≈ 7.1 bps.

#### `oos_sharpe_by_date()` - Date split (pas count split)

**Code : fxgate.py:79-94**

```python
def oos_sharpe_by_date(trades, frac: float = 0.4) -> float:
    """Out-of-sample Sharpe of the last `frac` of the timeline BY CALENDAR DATE."""
    ts = np.array([t["entry_ts"] for t in trades], dtype=float)
    boundary = lo + (1.0 - frac) * (hi - lo)
    return _sharpe(r[ts >= boundary])
```

**Piège attrapé : date-decay de l'arête** :
- Si on split par count (100 trades, last 40 sont OOS), on capture un regime-change dans ces 40 trades.
- Si le signal s'est appauvri sur 2024 mais avait été fort en 2023, le count-split te le cache.
- Ici : la limite est **calendaire** (t_min + 60% du temps = cutoff date) → OOS are vraiment les trades récents, même s'ils sont moins nombreux.
- Attrape ainsi l'edge qui a decayed.

#### `mc_random_entry_null()` - Random entry vs skill

**Code : fxgate.py:111-149**

```python
def mc_random_entry_null(
    close, ts, observed_mean_net: float, n_trades: int,
    horizon, n_long, n_short, cost_rt, n_sims: int = 1000, ...
) -> float:
    """p-value that observed mean beats random-entry null on the SAME path."""
    base_sides = np.array([1] * n_long + [-1] * n_short, dtype=float)
    rng = np.random.default_rng(seed)
    beats = 0
    for _ in range(n_sims):
        idx = rng.integers(lo, hi, size=n_trades)  # random bars
        sides = rng.permutation(np.resize(base_sides, n_trades))
        gross = sides * (close[idx + horizon] / close[idx] - 1.0)
        if float((gross - cost).mean()) >= observed_mean_net:
            beats += 1
    return beats / n_sims
```

**Piège attrapé : bull-beta trap** :
- Un trend-following banal sur or peut paraître bon si l'or monte depuis des années.
- Le MC null : tire aléatoirement n_trades entrées dans le même series de prix, **avec le même ratio long/short**.
- Si même l'aléatoire gagne (p ≈ 0.5), la "stratégie" est juste un achat du bull-market, pas un skill.
- OR trend-following FX fut débunkée exactement ainsi : random entry battait le signal.
- Threshold : want p < 0.05 (la stratégie vraie bat l'aléatoire significativement).

#### `event_basket_gate()` - Multi-instrument synchronized

**Code : fxgate.py:207-283**

```python
def event_basket_gate(
    events, instruments, horizon, cost, n_sims=2000, lag_days=1,
    frac_oos=0.4, no_overlap=True,
) -> dict:
    """Honest gate for a SYNCHRONIZED multi-instrument EVENT sleeve
    (e.g. one FOMC signal across the whole USD complex).
    
    The fix: N instruments firing on SAME event dates are ~1 correlated bet, NOT N independent.
    So:
      * sleeve = PER-EVENT BASKET AVERAGE (one return per event)
      * MC null relocates event DATES JOINTLY (one random anchor, shared across instruments)
    """
```

**Piège attrapé : synchronized-instruments trap** :
- Si FOMC exprime un signal unique sur USD, et tu trades {EURUSD, GBPUSD, AUDUSD}, ce sont *une* hypothèse, pas trois.
- Naïvement, compter 3 trades indépendants → variance comprimée, p-value spurious.
- Ici :
  - **Per-event basket average** : une entrée = une date d'événement, un retour (moyenne égal-poids sur les instruments).
  - **Null au niveau date** : le MC tire `len(events)` dates aléatoires, **même ancre** pour tous les instruments → préserve la corrélation cross-instrument.
- Résultat : variance du null correcte (le vrai risque), p-values honnêtes.

`_dedup_basket_events()` (line 169-185) : greedy no-overlap au niveau panier (horizon min entre events).

---

### 4. CPCV, PBO, DSR (src/hermes/validation/)

#### CPCV : Combinatorial Purged Cross-Validation (cpcv.py)

**Code : cpcv.py:33-84**

```python
def combinatorial_purged_splits(
    n_obs: int, n_groups: int = 6, k_test: int = 2, embargo: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """All C(n_groups, k_test) (train, test) paths with purge/embargo."""
    blocks = np.array_split(all_idx, n_groups)
    for combo in itertools.combinations(range(n_groups), k_test):
        test_idx = np.concatenate([blocks[b] for b in combo])
        # Purge: drop from train any index in test window
        drop = set(test_set)
        # Embargo: remove `embargo` indices after END of each maximal run
        if embargo > 0:
            for block_i in combo:
                if (block_i + 1) not in test_block_set:
                    last = int(blocks[block_i][-1])
                    for off in range(1, embargo + 1):
                        nxt = last + off
                        if nxt < n_obs:
                            drop.add(nxt)
        train_idx = all_idx[~np.isin(all_idx, list(drop))]
        splits.append((train_idx, test_idx))
```

**Piège attrapé : look-ahead via labels** :
- Partitionner le temps en 6 blocs, tester chaque combinaison de 2.
- **Purge** : aucun indice test ne fuit dans train (obvious).
- **Embargo** : les indices *immédiatement* après test sont supprimés du train (les labels des barres post-test peuvent chevaucher forward dans la fenêtre test, surtout pour triple-barrier).
- Résultat : C(6,2) = 15 chemins test → distribution Sharpe (pas un point).
- Si true edge : positif sur 10+/15 chemins.
- Si overfit : positif sur peut-être 3/15.

**TrialRegistry (cpcv.py:87-119)** : compte persistant du nombre de configs backtestées pour déflatage honnête DSR.

#### PBO : Probability of Backtest Overfitting (pbo.py)

**Code : pbo.py:19-47**

```python
def probability_backtest_overfitting(
    returns_matrix: Sequence[Sequence[float]], n_splits: int = 10
) -> float:
    """IS-best config rank in OOS: fraction landing in bottom half."""
    m = np.asarray(returns_matrix, dtype=float)  # T x N: time x configs
    blocks = np.array_split(np.arange(t), n_splits)
    for is_blocks in itertools.combinations(range(n_splits), n_splits // 2):
        # Pick the best config in-sample
        best = int(np.argmax(is_perf))
        # Rank it out-of-sample (1=worst, N=best)
        rank = int(np.sum(oos_perf <= oos_perf[best]))
        if rank in bottom_half:
            overfit += 1
    return overfit / total
```

**Piège attrapé : cherry-picking across configs** :
- T x N matrice : T periodes, N sleeves.
- Chaque partition : qui était in-sample-best ? Est-il aussi bon OOS ?
- Si 70% du temps l'IS-best rank bottom-half OOS → PBO = 0.7 = very overfit.
- Si 20% → PBO = 0.2 = healthier (la meilleure IS n'est pas une chimère).

Applied in book.py:229-232 : `probability_backtest_overfitting(aligned.to_numpy().tolist(), n_splits=n_blocks)`.

#### DSR : Deflated Sharpe Ratio (learning/stats.py)

**Code : stats.py:81-86**

```python
def deflated_sharpe_ratio(
    returns: Sequence[float], sr_variance: float, n_trials: int
) -> float:
    """PSR deflated by the expected-maximum-Sharpe benchmark for n_trials."""
    sr_star = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_star=sr_star)
```

`expected_max_sharpe()` (stats.py:65-78) : given sr_variance (dispersion across sleeves) and n_trials (configs tested), what Sharpe would you expect purely from noise ?

```python
def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """E[max Sharpe] across n_trials independent strategies."""
    if n_trials < 2 or sr_variance <= 0.0:
        return 0.0
    g = _EULER_MASCHERONI  # 0.5772...
    max_z = (1-g)*norm.ppf(1 - 1/n_trials) + g*norm.ppf(1 - 1/(n_trials*e))
    return math.sqrt(sr_variance) * max_z
```

**Exemple** :
- 10 sleeves, sr variance = 0.3, tu observes Sharpe = 0.8 sur 200 trades.
- expected_max ~ 0.4 (expectation under pure luck).
- DSR = probabilistic_sharpe_ratio(rets, sr_star=0.4) = P(true Sharpe > 0.4 | observed).
- Si DSR = 0.7 → 70% chance la vraie arête dépasse 0.4 (deflated).

Applied in book.py:239 : `deflated_sharpe_ratio(book_list, sr_var, effective_trials)`.

---

### 5. Full Gate & Integration (book.py:198-259 + fxgate.py:286-340)

#### `book_report()` - The verdict bundle

**Code : book.py:198-259**

```python
def book_report(
    sleeve_returns: dict[str, pd.Series],
    weighting: str = "inv_vol",
    min_periods: int = 8,
    n_trials: int | None = None,
    n_blocks: int = 10,
) -> dict:
    """Honest BOOK verdict: combine causally, then run the full gate.
    
    Returns:
      * book: n, sharpe, DSR (deflated by n_trials), PBO, CPCV path distribution
      * sleeves: per-sleeve solo Sharpe, LOO-Δ-Sharpe, mark count
      * orthogonality: rho matrix
    """
    ids = list(sleeve_returns)
    aligned = align_sleeves(sleeve_returns)
    book = combine_book(sleeve_returns, weighting, min_periods)
    
    solo = {sid: sharpe_ratio(aligned[sid].tolist()) for sid in ids}
    sr_var = float(np.var(list(solo.values()), ddof=1)) if len(ids) >= 2 else 0.0
    effective_trials = int(n_trials) if n_trials else max(1, len(ids))
    
    loo = leave_one_out_delta_sharpe(sleeve_returns, weighting, min_periods)
    rho = orthogonality_matrix(sleeve_returns)
    pbo = probability_backtest_overfitting(aligned.to_numpy().tolist(), n_splits=n_blocks)
    
    return {
        "book": {
            "n": len(book_list),
            "sharpe": sharpe_ratio(book_list),
            "dsr": deflated_sharpe_ratio(book_list, sr_var, effective_trials),
            "pbo": pbo,
            "cpcv": cpcv_book_distribution(book_list),
        },
        "sleeves": {
            sid: {
                "solo_sharpe": solo[sid],
                "loo_delta_sharpe": loo[sid],
                "n_marks": int((aligned[sid] != 0.0).sum()),
            }
            for sid in ids
        },
        "orthogonality": {...},
        "weighting": weighting,
        "n_trials": effective_trials,
    }
```

**Verdict logic** :
1. **Solo Sharpe per sleeve** : baseline, à titre informatif.
2. **LOO-Δ-Sharpe** : **critère d'acceptation** (§7). Si Δ_s > 0 → sleeve contributes.
3. **Orthogonality (ρ matrix)** : |ρ| < 0.3 against existing sleeves → vraie information.
4. **Book Sharpe** : the combined result.
5. **DSR** : P(true book Sharpe > threshold), deflated par sr_var et n_trials.
6. **PBO** : fraction of IS-best sleeves ranking bottom-half OOS.
7. **CPCV** : distribution of Sharpe across C(6,2) = 15 test paths.

#### `gate_report()` (fxgate.py:286-340) - Single-instrument gate

```python
def gate_report(
    close, ts, events, horizon, cost_rt, two_leg=False, warmup=0,
    frac_oos=0.4, n_folds=3, n_sims=1000, seed=0, winsor_k=1,
) -> dict:
    """Full §3 gate for one instrument's events."""
    trades = realize_event_returns(close, ts, events, horizon, cost_rt, two_leg, warmup)
    r = np.array([t["ret_net"] for t in trades], dtype=float)
    
    mc_p = mc_random_entry_null(
        close, ts, obs_mean, len(trades), horizon, n_long, n_short,
        cost_rt, n_sims, seed, two_leg, warmup,
    ) if trades else 1.0
    
    return {
        "n_trades": len(trades),
        "sharpe": _sharpe(r),
        "oos_sharpe": oos_sharpe_by_date(trades, frac_oos),
        "subperiod_positive_frac": subperiod_positive_frac(trades, n_folds),
        "mc_p": mc_p,  # p-value vs random entry
        "winsor_net_mean": winsor_mean,
        "trades": trades,
    }
```

**Gate rules** :
- `sharpe > 0.2` (very loose, pure baseline).
- `oos_sharpe > 0.1` (decay trap).
- `subperiod_positive_frac >= 0.67` (regime persistence → not in one golden window).
- `mc_p < 0.05` (vs random-entry null, must be significant).
- `winsor_net_mean > 0` (not one tail-event).

---

### 6. Pièges attrapés (résumé)

| Piège | Détection | Mécanisme |
|-------|-----------|-----------|
| **Bull-beta trap** | `mc_p < 0.05` | Random-entry null sur le même prix path. Si aléatoire gagne, c'est juste du buy-and-hold d'un bull market. |
| **Overlapping trades** | `no_overlap=True`, `busy_until` | Persister signal → ONE position (coût payé une fois), pas N pseudo-indépendants. Comprime sinon variance → p-values spurious. |
| **Synchronized instruments** | `event_basket_gate()` avec null au niveau DATE | Multi-instrument on same event = ONE bet. MC null tire dates aléatoires ENSEMBLE (pas par instrument), préserve corrélation. |
| **Date-decay** | `oos_sharpe_by_date()` | Split par *calendrier*, pas par count. Attrape l'edge qui decays au fil du temps. |
| **Forward-leak en poids** | `causal_inverse_vol_weights()` | Poids basés UNIQUEMENT sur passé (avant t), exécutable à l'ouverture. |
| **Héterogénéité non-honnête** | `align_sleeves()` + 0.0 fill | Sleeves cadences différentes : absent = flat. Jamais NaN-propagate. |
| **In-sample cherry-picking** | CPCV (15 chemins) | Pas un split 50/50 unique : C(6,2) paths. Real edge > 10/15 paths positif. |
| **Over-optimization des configs** | PBO + DSR | Is-best config ranking bottom-half OOS → over-optimized. DSR deflate par trial count. |
| **Under-counting trials** | TrialRegistry | True n_trials (persistent count), pas le count de sleeves. |
| **Solo Sharpe illusion** | `leave_one_out_delta_sharpe()` | Replace per-trade Sharpe par contribution au livre. Sleeve 0.2 solo mais ρ≈0 → +Δ > 0. |

---

### 7. Fichiers clés (références)

- **book.py** (59 lines) : align, causal weights, combine, LOO-Δ-Sharpe, orthogonality, cpcv_book, book_report.
- **fxgate.py** (341 lines) : realize_event_returns (no-overlap), oos_sharpe_by_date, mc_random_entry_null, event_basket_gate (null date-level), gate_report, to_periodic_returns.
- **triple_barrier.py** (45 lines) : triple_barrier_label (stop precedence, conservative).
- **cpcv.py** (119 lines) : combinatorial_purged_splits (purge + embargo anti-lookahead), TrialRegistry (persistent n_trials).
- **pbo.py** (48 lines) : probability_backtest_overfitting (IS-best OOS rank).
- **report.py** (124 lines) : cpcv_sharpe_distribution, backtest_report, deflated DSR logic.
- **stats.py** (131 lines) : sharpe_ratio, deflated_sharpe_ratio, expected_max_sharpe (selection bias deflation), PSR, skew/kurtosis correction.
- **backtest.py** (249 lines) : run_backtest (replay skills, triple-barrier label), _run_cross_asset (pairs ETH-BTC), chronological ledger for CPCV/DSR time-ordering.

---

### 8. Flux end-to-end

```
FX Signal (hunt.py) → (bar_index, side) events
    ↓
gate_report() : realize_event_returns (no-overlap)
    ├─ Sharpe, OOS Sharpe (by date), subperiod_positive_frac
    ├─ mc_random_entry_null (p < 0.05 ? )
    └─ → per_event returns, per_sleeve return series
          ↓
    to_periodic_returns() : bucket onto common calendar (W, D)
          ↓
          sleeve_returns dict[str, pd.Series]
                ↓
    book_report()
    ├─ align_sleeves() → common grid, 0.0 fill
    ├─ causal_inverse_vol_weights() → forward weights
    ├─ combine_book() → book series
    ├─ leave_one_out_delta_sharpe() → per-sleeve Δ (ACCEPT if > 0)
    ├─ orthogonality_matrix() → rho (want |ρ| < 0.3)
    ├─ cpcv_book_distribution() → C(6,2) paths
    ├─ pbo / dsr
    └─ → verdict dict : book metrics, per-sleeve LOO, rho
```

La machine d'honnêteté n'est **pas** une boîte noire : chaque piège a une contre-mesure explicite, chaque coefficient (min_periods, embargo, cost_rt, frac_oos) est ajustable et documenté. L'industrie financière enveloppe souvent ces problèmes dans des disclaimers ; Hermes les rend **inévitables à contourner**.

---

# PARTIE II — Les recherches du labo (extension)

## 6. Les recherches du labo (extension)

### (a) PROGRAMME MATIÈRES-PREMIÈRES / FX — Compte JustMarkets XAU+FX

**Verdict global:** Feasibility opérationnelle GO (`MT5-demo`, `src/hermes/infra/justmarkets_mt5.py`), mais tous les signaux directionnels REFUTÉS. Aucun déploiement.

#### B1 — Positionnement COT extrême (40 ans, 6 instruments)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:23-25`
- **Thesis:** contrarian sur les positionnements spéculatifs extrêmes en or/argent/devises
- **Résultat:** REFUTÉ — le short (contrarian) s'oppose à la tendance haussière; ne généralise sur 0/6 instruments; l'argent seul échoue OOS>0.3; MC-null p=0.059 (FAIL p<0.05)
- **Statut:** Mort

#### B2 — Choc taux réels (DFII10) → or
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:26`
- **Thesis:** or monte quand les taux réels baissent (demande d'assurance)
- **Résultat:** REFUTÉ — bull-beta pur; NÉGATIF sur toute la période 2011-15 (baisse de l'or); MC-null p=0.059 (marginal FAIL); biais long intrinsèque
- **Statut:** Mort (bull-beta)

#### B3 — GVZ (volatilité implicite or) fear/complacency timing
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:27`
- **Thesis:** acheter or quand GVZ monte (peur → fuite vers la sécurité)
- **Résultat:** REFUTÉ — drift long simplement; MC-null p=0.144; argent négatif; timing volatilité ≠ timing prix
- **Statut:** Mort

#### B5 — Divergence flux ETF GLD
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:28`
- **Thesis:** flux ETF → signaux monétaires avancés
- **Résultat:** REFUTÉ — flux décalé sur le prix; MC-null p=0.203; poolé négatif
- **Statut:** Mort (lag structurel)

#### B4 — ForexFactory surprise macro
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:29`
- **Thesis:** signal surprise économique sur yieldcurve → FX
- **Résultat:** BLOQUÉ — zéro panel libre sur 20+ ans (actual, forecast) sans accès FF payant
- **Statut:** Données verrouillées

#### E1 — Inventaire physique COMEX
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:30`
- **Thesis:** registered/eligible ounces → primemium carry
- **Résultat:** BLOQUÉ — ToS CME interdit l'automation; chaque mirror gratuit est payant ou jour-courant uniquement
- **Statut:** Données verrouillées par régulation

#### E2 — WTI term-structure roll-yield → CAD/USD / AUD/USD
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:31`
- **Thesis:** cointegration(WTI backwardation, carry currencies) → directional FX
- **Résultat:** REFUTÉ — deux jambes net-négatif; MC-null FAIL sur 18 configs (min p=0.169); inverse du signe prévu (contango ≠ carry)
- **Statut:** Mort (signe inversé)

#### E3 — FOMC tone-change surprise (corpus 216→219)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:32`
- **Thesis:** choc ton monétaire (taupe) → repricing rapide devises/or
- **Résultat:** REFUTÉ — fully repriced at 2pm print; basket MC-null p=0.515; XAU p=0.440; pas de lag exploitable
- **Statut:** Mort (reprice immédiat)

#### X1 — Crypto DVOL fear → safe-haven (XAU/JPY/CHF)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:33`
- **Thesis:** peur crypto (DVOL hausse) pousse vers haven = or + JPY + CHF
- **Résultat:** REFUTÉ — or bull-drift crypto +53bps ≈ unconditional +48bps = bull-beta pur; basket p=0.174; zero corrélation lead-lag
- **Statut:** Mort (juste drift d'or haussier)

#### X2 — Crypto stablecoin dry-powder → risk-on FX
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:34`
- **Thesis:** accumulation stablecoin → cash-prêt crypto → rebond risk-assets → AUD/JPY correlated
- **Résultat:** REFUTÉ — AUD/JPY sont UNE SEULE alpha USD (beta-corrélés); basket p=0.2375; zéro lead-lag cross-asset
- **Statut:** Mort (facteur USD unique)

#### X3 — Conditioning (signal × régime)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:35`
- **Thesis:** même signal faible, si conditionné sur régime → edge manifeste
- **Résultat:** REFUTÉ — conditionné ≈ base, le *complément* score meilleur; shuffle-p=0.42
- **Statut:** Mort (non-événement)

#### D1 — ML blend (8 features orthogonales)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:36`
- **Thesis:** combination robustante sur tous les axes
- **Résultat:** NO_EDGE — OOS accuracy 0.518–0.543 ≤ 0.556 (long base rate); MC-null p=1.000
- **Statut:** Mort

#### Héritage (canaux précédemment confirmés morts)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:38-39`
- Tous les signaux OHLCV rapides/tout-cadence
- Cointegration/pairs (half-lives 19-99 jours) → MC random entry p non robuste
- Session/time-of-day (zéro mean per-session)
- TSMOM or pur (bull-beta, MC null p=0.358)
- CTA trend 33-futures (0/112 configs battent la dérive 0.431)
- FX carry G10 (post-2008 crush, rank-shuffle p≈0.28)
- Carry cross-sectionnel (rank-shuffle p=0.80)

#### Premia physiques — roll-yield Gorton-Rouwenhorst
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_phys/ROLL_YIELD_VERDICT.md`
- **Proxy validé:** ETF-roll-return (USO/UNG/…) vs true c1/c2 = corr +0.41 → proxy fidèle (ce n'est pas un artefact données)
- **Inversion décisive ground-truth:**
  - WTI vraie structure terme (284 mois, 2000-2024)
  - IC(trailing roll → fwd ret) = **−0.10** (2000-07) / **−0.35** (2008-14) / **−0.27** (2015-24)
  - Long-backwardation timing = **−17%/yr** (null p=0.98)
  - Hold WTI long-when-backwardated = **−3%/yr vs +11%/yr always-long**
  - **Timing roll-yield DÉTRUIT la valeur, non la crée**
- **Kill 2:** rank-shuffle null p=0.090 mean/0.099 max (FAIL p<0.05); DSR 0.12; meilleure cellule 12-cellule grid-max cherry-pick
- **Kill 3:** deflation brutal — book Sharpe +0.011, DSR 0.12, PBO 0.11

**Statut:** Mort (signe INCORRECT ground-truth, pas un artefact proxy)

#### Saisonnalité commodités physiques
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_phys/SEASONALITY_VERDICT.md`
- **Priors pré-enregistrés:** nat-gas/heating-oil hiver long, essence pré-été long, grain semis long/moisson short, bétail printemps long
- **Book:** Sharpe +0.469, IS +0.473 / OOS +0.464
- **Kill 1:** MC rotation-null p_max=0.068 (FAIL stably <0.05)
- **Kill 2:** deflation DSR(7 syms)=0.212, DSR(35 cells)=0.006 (bar 0.95) FAIL
- **Leave-one-commodity-out:** essencé seul = −0.053 book (tout le signal), autres +0.42–0.67 → essence pré-summer SEULE
- **Essence seule:** Sharpe 0.896 OOS +0.587, mais DSR(12)=0.878, DSR(19)=0.815 (tous <0.95 famille implicite)
- **Data caveat:** RB=F roll RBOB hiver→été = artefact roll mécanique; données back-adjusted neutraliseraient l'essencé → overstatement libre

**Statut:** Mort (essence seule signal pré-registered, mais roll-artifact + mine-exhaustive noise floor)

#### COT hedging-pressure (cross-sectionnel dollar-neutral)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_phys/PHYS_VERDICT_AND_ACQUISITION_MAP.md:29`
- **Résultat:** REFUTÉ — mean −5.9 bps/wk, Sharpe −0.03; rank-shuffle p≈0.55
- **Statut:** Mort

#### Inventory surprise EIA (front future, build→short / draw→long)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_phys/PHYS_VERDICT_AND_ACQUISITION_MAP.md:31`
- **Pseudo-win:** crude h=3 mc_p<0.01 → knife-edge (seul h=3, lags 5-6 seulement)
- **Death-blow:** 76% de PnL = 1 trade (2020-04-15 short into −$37 oil) = un-tradeable + yfinance artifact
- **Ex-trade:** Sharpe 0.043 (rien)
- **Statut:** Mort (queue event non-tradeable)

#### WTI pure carry (signed C1/short-C2 calendar spread)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_phys/PHYS_VERDICT_AND_ACQUISITION_MAP.md:32`
- **Carry beta:** large et réel (+66…+206 bps/trade unconditional)
- **Résultat timed:** random-sign null p≈0.97–1.00 partout → REFUTÉ timed
- **Leçon:** carry force existe, mais UNHARVEST-ABLE en stratégie signée single-name
- **Statut:** Mort (information existante, but non-extractible)

#### Data-scout acquisition map
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_phys/PHYS_VERDICT_AND_ACQUISITION_MAP.md:(d)`

| Tier | Resource | Coût | Verdict |
|------|----------|------|---------|
| **0** | EIA API key free (register) | $0 | DO NOW — hygiene seulement |
| **1** | Databento $125 credit CME curves | $0 out-of-pocket | **DO NOT SPEND** — premium refuté ground-truth |
| **2** | Databento paid CME live, ~$25–80/mo | $ | DO NOT SPEND — Tier-1 refuté |
| **3** | Alt-data satellite/AIS, $100s–$1000s/mo | $$$ | DEFER/NEVER — free pre-print edge contradiction |

**Synthèse:** DÉPENSER RIEN MAINTENANT. Garder $125 crédit. Tous les primes (roll carry, hedging, inventory, seasonality) refutés free evidence.

---

### (b) RE-AUDIT LIVE CRYPTO BOOK — `research_crypto2/LAB_JOURNAL_live_reaudit.md`

**Machinery réutilisée:** `research_fx/fxgate.py` (MC random-entry null, OOS-by-date, subperiod, winsor, event_basket_gate), `src/hermes/validation/book.py` (causal inverse-vol, LOO-Δ-Sharpe, ρ matrix, DSR, PBO, CPCV), `research_crypto2/cryptolib.py` (LIVE 4h replicat, 15bps crypto taker RT, unconditional_long_drift, split_bull_bear).

**Honesty verification:** S1 max abs err 1e-13 (40/40 sides), S2 max abs err 1e-12 (30/30 sides) vs live agents → events fed to gate = événements live générés.

#### S1 — ts_mom_vol_scaled
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_crypto2/LAB_JOURNAL_live_reaudit.md:23-34`

- **Per-asset MC null:** anti-conservative PASS BTC/SOL, FAIL ETH (p=0.22)
- **Synchronized BTC/ETH/SOL joint null DATE-LEVEL:** **p=0.69** — random entry bat 69% temps si longs co-movement bull comptés UNE seule bet corrélée
- **Regime split proof bull-beta:** pooled long edge-above-tide +40..+96bps, mais BEAR-only long mean **NÉGATIF** sur tous 3 (BTC −32, ETH −109, SOL −54bps) → edge = artifact régime taureau (momentum = plus-long pendant bulls)
- **Short leg:** net-négatif tous 3 (BTC −19.5, ETH −50.8, SOL −8.3bps) → shorting crypto perd
- **Deflation:** live cell NOT grid-max (BTC rank 12/28, ETH 26/28); DSR ETH 0.13
- **OOS-40 collapse:** BTC 0.07, ETH −0.07, SOL 0.01
- **Book LOO-Δ-Sharpe = −0.146** → removing momentum IMPROVES book

**Verdict:** WEAKER THAN BELIEVED / largely bull-beta. **DROP or de-weight.**

#### S2 — funding_squeeze_continuation
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_crypto2/LAB_JOURNAL_live_reaudit.md:36-53`

- **Synchronized basket date-level joint null:** **p=0.017, stable (max over 5 seeds 0.024)** ← THE decisive test — killed S1 (0.69) + shadow sleeves; funding PASSES
- **Edge above tide:** per-asset +147/+284/+242bps; pooled +198bps at H=24
- **BEAR-only stability:** pooled mean POSITIVE (+72bps H=24, +154bps H=48) → NOT just bull-beta (SOL bear −397bps is noisy 5 trades; pooled de-noised picture solid)
- **Honest magnitudes (15bps taker, H=24 fixed):** per-trade Sharpe 0.32 (BTC) / 0.37 (ETH) / 0.26 (SOL), annualized ~0.7–1.1, OOS-40 ~0.28–0.33. Original "OOS Sharpe ~+1.1" was per-asset ANNUALIZED, NOT OOS-40 (~0.3)
- **Deflation haircut real:** live cell ranks 7/27 (BTC), 8/27 (ETH), 15/27 (SOL) — 648-sweep donc NOT grid-max. DSR positive (0.68/0.98/0.82) mais ETH PBO 0.60 flag
- **SOL stability:** per-asset MC fails (p=0.086) → SOL weak leg; BTC/ETH carry sleeve
- **Orthogonality CORRECTION:** docstring claims ρ~0.04 momentum. Daily granularity = 0.09 (fair). But S1-LONG×S2 daily = 0.18, weekly BOOK grid = **0.35** (>0.30 bar) → both sleeves crowd longs same bull windows; regime-level co-movement masked daily noise. **S2 LESS orthogonal advertised** mais ONLY positive book contributor (LOO-Δ-Sharpe +0.015)

**Verdict:** GENUINELY SURVIVES, recalibrate headline. **KEEP** — seule vraie edge cryptomonnaie.

#### S3 — funding_carry_harvest (delta-neutral)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_crypto2/LAB_JOURNAL_live_reaudit.md:55-68`

- **Hedged carry honest 30bps TWO-LEG taker RT:** net +16/+23/+16bps per trade, per-trade Sharpe 0.35–0.54, mais **OOS-40 ≈ 0** (BTC −0.02, ETH +0.05, SOL +0.24)
- **"+2.4…+4.5 OOS Sharpe" claim does NOT reproduce** sous coût honnête → needed near-zero cost et/ou overlap-inflated annualization per-trade
- **Cost cliff:** edge dies 30↔40bps two-leg (+6bps at 40bps, +1bps at 40bps+5bps basis). Viable MAKER-only 12bps (+34bps net, Sharpe ~0.99)
- **Time-weighted:** ~2–3%/yr capital, ~15% deployment → small accretive carry
- **Deflation brutal:** DSR=0.000 tous 3; live cell ranks 10/16; "Sharpe ~1.4" grid-max cells thin high-arm (n=16–33) avec OOS-40=0.000 → noise
- **Naked-short:** BTC −0.8, ETH −94, SOL −479bps, tous OOS-negative. Live HOLD-only decision correct, validated
- **Book LOO-Δ-Sharpe = −0.081 honest cost** (weekly grid) → ne contribue pas net 30bps

**Verdict:** real but THIN, maker-only, deflation-fragile. **VIABLE MAKER-ONLY, otherwise MARGINAL.**

#### Book résumé (3 live sleeves)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_crypto2/LAB_JOURNAL_live_reaudit.md:70-75`

- **LOO-Δ-Sharpe:** S1 −0.146, **S2 +0.015 (only positive)**, S3 −0.081
- **ρ:** S1×S2 0.35, S1×S3 0.24, S2×S3 0.27
- **Only sleeve earning allocation harder bar = S2**, even though more correlated to momentum regime-level than docstring claims

---

### (c) NOUVELLES CHASSES CRYPTO — tous REFUTÉS sauf 2 data-gated

**Fichier principal:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_crypto2/JOURNAL_options_surface_hunt.md` et `LAB_JOURNAL_positioning_flow_hunt.md`

#### Surface options (Deribit BTC/ETH)
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_crypto2/JOURNAL_options_surface_hunt.md`

- **1024-config grid** (VRP, TERM, VOV, RSKEW) × fade/follow × z∈{1.0,1.5,2.0,2.5} × h∈{12,24,48,72h} × {BTC,ETH}
- **Grid-max per-trade Sharpe = 0.1934** < E[max Sharpe null 1024 trials] = 0.1950 → **single best config BELOW coin-flip expected value**
- **Family grid-bests all fail full gate:**

| Family | Config | per-trade Sharpe | vs E[maxSR]=0.195 | Joint-p | Verdict |
|--------|--------|-----|-----|-----|---------|
| VRP | vrp30_z_30d follow z2.5 h72 | 0.107 | FAIL | 0.264 | REFUTED |
| TERM | term_z_90d follow z1.5 h72 | 0.049 | FAIL | 0.194 | REFUTED |
| VOV | vov_z_30d follow z2.0 h72 | 0.071 | FAIL | 0.851 | REFUTED |
| RSKEW | rskew14_z_90d fade z2.0 h72 | 0.119 | FAIL | 0.000(*) | REFUTED |

**RSKEW rescue attempt:** basket joint-p=0.000 looked alive → orthogonality audit found **RSKEW-fade = long-after-crash = SAME fear-reversion/dip-buy as killed DVOL**:
- ~34% RSKEW longs overlap DVOL-fear (dvol_z≥1) ou ≥5% drawdown
- BTC & ETH crash together → 131 basket events = far fewer independent bets → joint-null anti-conservative (synchronized dip-buys)
- Strip fear/drawdown overlap → residual collapses: BTC Sh 0.126→0.104 (n49), ETH Sh 0.112→0.032 (n40, ~zero)
- Even full per-trade Sharpe (0.126/0.112) below 0.195 deflation bar

**Verdict:** Surface options REFUTED. DVOL level + VRP + term + realized-skew + VOV ne produit pas edge orthogonal funding/momentum livre. **Do not deploy.**

**Forward path:** literal 25d risk-reversal/butterfly (true OTM put/call skew) seul unexplored axis, but **NOT backtestable retroactively.** Only way = forward-log via deribit_iv.py cron (skew_term_log.csv currently 2 rows). Revisit ~6–12 months.

#### Positioning / Flow / Basis edges
**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_crypto2/LAB_JOURNAL_positioning_flow_hunt.md`

| Hunt | Signal | Best | Joint-p | Verdict |
|------|--------|------|---------|---------|
| **H1** | On-chain NET-FLOW exchange (outflow=long), daily +1d lag | BTC +116bps | **0.193** | REFUTED |
| **H2** | Funding TERM+DISPERSION cross-section | 0.7bps GROSS (« 30bps cost) | n/a | REFUTED (cost/wrong-sign) |
| **H3** | OI+funding crowding / LSR positioning | top_lsr ρ **−0.27/−0.33/−0.11** BTC/ETH/SOL | only 30d data | **UNTESTABLE — data lead** |
| **H4** | Stablecoin-supply acceleration | basket +47bps mean | **0.428** | REFUTED |
| **H5** | Liquidation-cascade reversion proxy | BTC +34bps fold 1.00 | **0.907** | REFUTED |

**H1 — on-chain exchange net-flow:** best single result hunt entier, BTC +116bps mc_p<0.05 stable, but OOS-40 collapses 0.09, deflation rank 2/27 grid-max PBO 0.51, ETH fails (−66bps edge). **Synchronized BTC/ETH joint-null p=0.193** → random entry matches once colinear outflow longs = ONE bet. Real-but-fragile BTC artifact.

**H2 — funding term/dispersion:** short-crowded loses 187–251bps (catastrophically wrong-signed), spread **0.7bps GROSS/24h** order of magnitude under 30bps cost. Every funding functional collapses onto "high funding continues up" (S2) ou "buy dips bull" (tide).

**H3 — OI+LSR positioning:** Binance free LSR retains ~30 days only (2026-05-24→06-24). Cannot gate honestly (no OOS-by-date, no multi-seed MC). **BUT 30d probe surfaced single most theory-aligned NEW signal:** **top_lsr CONTRARIAN** — **ρ(top_lsr, fwd 6h) = −0.27 BTC / −0.33 ETH / −0.11 SOL**, sign-STABLE both window-halves (never flips). When top perp traders net-long → forward returns negative. **This is one DATA lead worth investment** (continuous Binance LSR self-collection ou paid Coinglass history) before verdict. Cannot deploy 30 days.

**H4 — stablecoin acceleration:** designed orthogonal existing 60d sleeve (ρ −0.018). But acceleration noise: BTC/ETH basket **p=0.428**, DSR 0.096, edge-above-tide positive only H=3 (~+19bps) negative H=5/10. Existing 60d sleeve remains better expression.

**H5 — liquidation cascade proxy:** free historical liquidation data unavailable (Binance `allForceOrders` HTTP-400; Coinglass paid). Built OHLCV+funding proxy (down-bar z≤−1.5 + lower-wick≥0.33 + funding>0 → long). Gross +132bps looks great, but **BTC/ETH/SOL joint null p=0.907** → random entry beats 91% time. Per-asset MC fails all three; bear legs strongly negative (−67, −144bps = bull-beta). "Flush bounce" just co-movement buying-any-dip.

**Conclusion hunt:** de-overlapped synchronized DATE-LEVEL joint null (`event_basket_gate`) matters for crypto positioning. Every positive single-asset positioning edge here (H1 BTC, H5 BTC) had bull-beta removed YET STILL died joint-null (0.19, 0.91). Only S2 funding-squeeze (p=0.017) ever passed this gate. **H3 top-trader LSR contrarian ONLY data lead** (bottleneck = 30d → ≥12-18m LSR history needed, NOT refuted hypothesis). Everything else REFUTED.

---

### (d) SLEEVES SHADOW — DVOL, stablecoin — rétrogradés à 0 CONFIRMÉ

**Fichier:** `/Users/lacarbonnerie/Developer/hermes-hedge-fund/research_fx/HONEST_NEGATIVE_REPORT.md:83-100`

Upgraded machinery (event_basket_gate, DSR/PBO deflation) applied to 2 existing crypto shadow sleeves blessed *before* honesty bar existed:

- **DVOL (peur-reversion):** per-trade Sharpe ~0.099 < grid-max-under-null (~0.154) → 7/96 configs beat null threshold → **fails deflation → real-but-fragile**
- **Stablecoin (60d growth):** Bonferroni p=1.0 grid → **fails deflation → demoted**. Net bps/trade reproduce (BTC/ETH +24/+50 et +42/+63), chacun bat null per-asset, signaux ≠ rien — just don't clear deflated bar

**Net result:** machinery is lab-wide honesty upgrade that *demoted book's own "wins"* — exactly what honest gate should do. **Both stay SHADOW, size 0 (zero capital risked).**

**Standing promotion bar (any sleeve, any book):** beat **beta-controlled date-level basket null** (de-overlapped, production lag) **AND** survive multiple-testing **deflation** (DSR vs grid-max / PBO<0.5) before "confirmed" label. Random-entry null surviving = floor, not bar.

---

### Conclusion synthétique

**TOUS les archétypes testés matières-premières/FX REFUTÉS:**
- COT, taux réels, DVOL-fear timing, flux ETF, FOMC tone, carry cross-sectional, roll-yield carry (signe inversé ground-truth), seasonalité, inventory surprise, hedging-pressure
- **Mécanisme unique:** chaque signal libre = BRUIT (fail MC random-entry null) ou BULL-BETA pur (edge = juste drift haussier or/complexe)
- Crypto↔FX ne transfère pas (lead-lag zéro, ρ régime-dependent, unreliable forward)

**Cryptomonnaie: 1 SEULE edge retail-accessible survit hardest bar:**
- **S2 funding_squeeze_continuation:** p=0.017 synchronized joint-null, OOS-40 ~0.3, ρ momentum 0.35 regime-level, LOO-Δ-Sharpe +0.015
- S1 momentum = drag (LOO −0.146, joint-p 0.69)
- S3 carry = marginal maker-only (<0 cost ~30bps+)

**Nouvelles chasses:** 
- Options surface (1024-configs): grid-max 0.193 < null 0.195 → **REFUTÉES**
- Positioning/flow: H1 on-chain (real-but-fragile BTC), H4 stablecoin-accel (orthogonal noise), H5 liq-cascade (worse momentum), **H3 top_lsr CONTRARIAN −0.27/−0.33/−0.11 seule lead data-gated** (30d insufficient, besoin 12-18m Binance LSR history)

**Verdict final:** edges en **CRYPTOMONNAIE SEUL** (retail-accessible), pas matières-premières/FX (marchés institutionnels arbitrés). Lab-wide honesty bar refusé ~8 faux positifs session. Tous primes physiques (roll carry, seasonality, hedging) closed free evidence + WTI ground-truth signe inversé. **Dépenser $0 physiques/FX données. Hold Databento $125. Seul invest data = Binance LSR historique continu.**
---

# PARTIE III — L'incident du 25 juin 2026 & les réajustements

## 7. L'incident : l'entreprise s'est arrêtée « toute seule »

**Le 25 juin 2026 à 04:07:13 UTC (06:07 CEST), l'agent live s'est arrêté** et l'API a répondu
`502 Bad Gateway`. Le fait troublant : **aucun commit ni push depuis plus de 24 h** (HEAD = `1d4ca1c`,
daté du 23 juin). Le système s'est donc éteint sans qu'on y touche — ce qui imposait de comprendre
*pourquoi* avant tout redéploiement (sinon, récidive assurée).

Dernier battement de cœur sain, à 04:00 UTC :
> equity **$104.04**, 0 position ouverte, **6 trades**, **83 % win**, PnL réalisé **+$4.15**.

Puis, à 04:07:13, un arrêt **propre** d'uvicorn (`Shutting down -> Application shutdown complete`) — pas
un crash, pas d'exception, pas d'OOM.

## 8. Le diagnostic forensique

Via l'**API Docker de Dokploy** (`docker.getConfig` / `docker.getContainersByAppNameMatch`), état réel
des conteneurs au moment de la panne :

| Conteneur | État | Détail |
|---|---|---|
| `hermes-agent` | **Exited (143)** | SIGTERM (128+15) · `OOMKilled: false` · `RestartCount: 0` · démarré 23/06 14:49, arrêté 25/06 04:07:14 |
| `hermes-ts` (sidecar Tailscale) | **running** | « Up 4h » — **recréé** vers 04:07 |
| `hermes-redis` | **Exited (128)** | tombé en même temps |

Et le **smoking gun**, dans l'inspection de l'agent :
> `Error: "cannot join network of a non running container: <id hermes-ts>"`

Les **Audit Logs** de Dokploy sont une fonctionnalité *enterprise* (réponse `403`), donc le déclencheur
exact (reboot vs action manuelle) n'est pas lisible — **mais le mécanisme, lui, est concluant.**

## 9. La cause racine — la course `network_mode: service:`

L'agent utilisait **`network_mode: service:hermes-ts`** : il **partageait la pile réseau** du sidecar
Tailscale pour router l'appel LLM vers le Mac (llama.cpp) via le tailnet.

1. Un événement **au niveau de l'hôte** (reboot du VPS / redémarrage du démon Docker) a **recréé le
   sidecar** Tailscale (d'où son « Up 4h » et la chute simultanée de redis).
2. Docker a tenté de redémarrer `hermes-agent` **avant** que le sidecar ne soit prêt → l'agent n'a pas
   pu rejoindre la pile réseau → `cannot join network of a non running container` → **il est resté mort**.
3. `depends_on` **n'ordonne que le `compose up` initial**, PAS le redémarrage par conteneur après un
   reboot. C'est la **course classique de `network_mode: service:`**.

**Ce n'est ni le code, ni un push, ni la recherche du labo** : c'était une **fragilité structurelle de
l'infrastructure**.

## 10. La décision (et son affinage décisif)

- **Option écartée** — un *boot-hook* qui referait `docker compose up` au démarrage : trop fragile (il
  dépend d'un hook qui doit se déclencher correctement à *chaque* reboot).
- **Choix** — rendre l'agent **auto-suffisant** en le **découplant** de Tailscale.
- **Affinage décisif (apport de l'opérateur)** : le fallback Mac llama.cpp était devenu **vestigial**.
  DeepSeek V4 est le cerveau **primaire 24/7**, et le llama-server du Mac n'a jamais été un service
  fiable (session shell, derrière NAT, relais DERP, s'endort avec le Mac). Préserver ce fallback ne
  justifiait pas un sidecar Tailscale entier. **Décision : supprimer Tailscale ENTIÈREMENT** et
  remplacer le fallback Mac par un **fallback cloud indépendant**.

## 11. La nouvelle configuration

**Un seul conteneur auto-suffisant** (+ redis). Plus de sidecar Tailscale, plus de dépendance Mac, plus
de SOCKS5, plus de course réseau possible.

- `hermes-agent` : **IP propre** sur `dokploy-network`, **ingress Traefik sur lui-même**, joint tout
  par internet — **Binance** (données + paper), **DeepSeek V4** (cerveau primaire, **direct**), et un
  **fallback cloud**.
- **Fallback** : **OpenRouter -> `inclusionai/ring-2.6-1t`** (utilisé uniquement si DeepSeek est
  injoignable). **Zéro changement de code** : le routeur LLM prend déjà `LLM_PRIMARY/SECONDARY_*` comme
  jeu de secours dès que `DEEPSEEK_API_KEY` est posée — il suffit de pointer ces variables vers
  OpenRouter dans Dokploy.
- **Pourquoi c'est robuste** : un agent **mono-conteneur** ne peut plus « rater de rejoindre le réseau
  d'un conteneur non démarré » — *il n'y a plus de second conteneur dont dépendre*. Un reboot de l'hôte
  ne peut plus le coincer.
- **Sûreté** : paper-only, **aucun capital à risque**, l'état (equity, journal, positions) **persiste
  dans le volume `/data`** et survit au redémarrage.

**Vérifié** : `272 tests passed`, ruff clean, `docker compose config` **valide**. Commit du correctif :
**`3630c07`** (`fix(infra): decouple the agent from Tailscale`). Les `research_*/` (~63 Mo) ont été
sortis du git (`.gitignore`) — leurs *résultats* vivent ici et en mémoire, pas les données brutes.

## 12. Lien avec le re-audit (décision séparée, en attente)

En parallèle, le re-audit du book crypto live (Partie II) a montré que **S1 (momentum) est un passager
bull-beta** à dé-pondérer/sortir, **S2 (funding_squeeze) la seule vraie edge**, **S3 (carry) mince et
maker-only**. Cette décision-**stratégie** est volontairement **dissociée** du correctif d'**infra** :
elle aura son propre commit, une fois Hermès remonté — on ne mélange jamais un fix d'infrastructure et
un changement de portefeuille dans le même déploiement.

---

## Annexe — chronologie de la séance

1. Recherche d'extension (matières premières/FX, nouvelles données crypto) — *cf. Partie II*, l'essentiel réfuté.
2. Re-audit du book live avec la machine d'honnêteté durcie — seul S2 survit.
3. Panne du 25 juin 04:07 UTC → diagnostic forensique → cause racine `network_mode: service:`.
4. Découplage de l'agent → suppression de Tailscale → fallback cloud OpenRouter/ring-2.6-1t.
5. Commit `3630c07` du correctif d'infra + ce dossier. Décision sur S1 reportée à un commit dédié.

