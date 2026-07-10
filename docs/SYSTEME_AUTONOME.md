# Le Système de Trading Autonome Hermès — inventaire définitif (2026-07-10)

> Directive opérateur : « un système de trading autonome qui combine les capacités de l'agent
> Hermès (toutes ses aptitudes), tous les algorithmes applicables aux cryptos sur Binance et/ou
> JustMarkets, le plus safe possible — pense à Markov, Jim Simons et les chercheurs éminents. »
>
> Méthode : 4 inventaires par agents (aptitudes, algorithmes, sûreté, venues), chacun sourcé
> fichier:ligne + endpoints prod, puis un audit adversarial indépendant de 40 vérifications
> contre le code réel (verdicts : algorithmes FIABLE ; aptitudes/sûreté/venues À-CORRIGER —
> corrections appliquées ci-dessous et défauts trouvés CORRIGÉS EN CODE le jour même).
> Les FINDINGS des streams et le DeadEndsLedger priment sur tout document antérieur.

## Corrections issues de l'audit (appliquées avant publication)
- `ticker()` est ACTIF (appelé par paper.last_price) — seul `order_book()` est dormant.
- `tests/test_stage.py` existe (l'audit sûreté s'était trompé) ; le pin manquant
  « evaluate() ne mute jamais la stage » a été AJOUTÉ (commit 89cd000).
- Lignes venues corrigées : stages 1/2/3 étaient aux lignes 44/57/70 (pas 34/49/63).
- L'écart registre prod (1892) vs repo (1910) = exactement 6 sources de provenance
  absentes du volume prod (18 essais) — le repo est canonique, item de seed au déploiement ouvert.
- Le commentaire « carry display-only » était mensonger dans DEUX fichiers (library.py ET
  research/signals.py) — les deux restatés (commit 89cd000).

## Défauts réels trouvés par cet inventaire et CORRIGÉS le jour même (commits 89cd000, 493e3b7)
1. **No-pyramid cross-skill** (prouvé par exécution) : la porte finale rejetait toute seconde
   position par symbole tous skills confondus, étranglant le design multi-sleeves testé et
   l'ancre S2. → clé (symbol, skill_id).
2. **Plancher de confiance vs sleeves auto-sélectives** : le floor 0.55 rétrécissait
   silencieusement les seuils VALIDÉS (S2 : z≥2.2 effectif au lieu de 2.0 ; carry : bps≥5.5 au
   lieu de 2) — même classe que le bug maker-fees. → skills auto-sélectifs exempts (>0),
   halte TRM (conf=0) rejette toujours, propositions LLM restent soumises au floor.
3. **Résilience LLM** : une panne LLM totale avortait le tick entier, règle déterministe
   comprise. → trader.run fail-soft (HOLD par défaut), le driver déterministe décide.
4. **Adapter Binance manquant + étages pointés sur un venue non possédé** : écrit
   (BinanceUSDMExchange, contrat de sûreté complet) + étages dormants re-pointés binanceusdm.

## Constats restants à l'attention de l'opérateur (non-bloquants aujourd'hui, datés)
- **PM fail-open** : si le LLM du Portfolio Manager meurt, il défaut à APPROVED (veto-only par
  design — cohérent avec « le déterministe décide », documenté ici pour transparence).
- **TRM = NullVerifier vraisemblable en prod** (aucun checkpoint /data/trm/latest.pt visible) —
  le downgrade-only est correct mais inopérant sans modèle ; à trancher : entraîner ou retirer.
- **Pas de blackout macro** (FOMC/CPI) — recommandé AVANT stage_1 (la leçon des $12k).
- **Pas de circuit-breaker de dérive multi-jours** au-delà du kill quotidien — recommandé avant stage_1.
- **Fallback OpenRouter** : mécanisme codé + testé ; la clé réelle vit dans l'UI Dokploy,
  invérifiable depuis le repo — à vérifier par l'opérateur.
- **Routage carry réel** : structurel en paper, NotImplementedError sur Binance réel — S3 ne
  peut PAS trader nu sur aucun venue câblé ; la jambe spot réelle reste à construire et valider.


---

# PARTIE I — LES APTITUDES DE L'AGENT (vérifiées code + prod)

**Synthèse :** Cartographie vérifiée des aptitudes Hermès (code + prod api.betterlifewithmk.com, 2026-07-10). 8 couches auditées avec fichier:ligne. Trois écarts factuels significatifs trouvés: (1) funding_carry_harvest N'EST PAS "display-only HOLD" comme l'affirme son commentaire dans skills/library.py:202-206 — le code réel (research/signals.py:279-296) émet BUY dès que le funding clear le seuil, ET il est bien dans config allowed_skills (default.yaml:114); la structure delta-neutral existe réellement (execution/paper.py:187-228). (2) Le cron forward.run n'est PAS câblé dans le scheduler in-process (scheduler.py n'importe jamais hermes.forward.run) ni dans aucun GitHub workflow — pourtant la prod accumule des données fraîches (top_lsr last_ts=aujourd'hui): le mécanisme réel est externe au repo (VPS cron probablement), non auditable ici. (3) Le compteur d'essais "crypto" affiché par la prod LIVE (1892) diverge du fichier committé governance/trials_by_book.json (1910) — la prod lit un registre séparé (/data/governance, docker-compose.yml) qui n'est pas synchronisé avec le repo local.


## 1. DONNÉES — feeds, caches, datalake gelé, collecteurs forward

| Aptitude | Où (fichier:ligne) | État | Preuve |
|---|---|---|---|
| OHLCV multi-timeframe (1m..1d) via CCXT | `src/hermes/data/market.py:25-40` (`MarketData.ohlcv`); config `data.ohlcv_timeframes` (`config/default.yaml:194`) | ACTIF | Consommé par `pulse.py:123` et `scheduler.py:37` (market_tick 4h) |
| Funding rate live Binance USDT-M (public REST) | `src/hermes/data/market.py:42-71` (`funding_history`, endpoint `fapi.binance.com/fapi/v1/fundingRate`) | ACTIF | Alimente `funding_squeeze_continuation`/`funding_carry_harvest`; fail-soft (retourne df vide sur erreur, jamais de crash tick) |
| Order book (depth) | `src/hermes/data/market.py:76-77` (`order_book`) | **DORMANT** — méthode définie mais 0 appelant dans tout `src/hermes` (`grep order_book` ne trouve que la définition) | Aucun test, aucun appelant |
| Ticker | `src/hermes/data/market.py:73-74` | DORMANT (idem, pas d'appelant trouvé en dehors de la classe) | — |
| News feed (CryptoPanic + fallback Reddit) | `src/hermes/data/news.py:15-67` | ACTIF (agent news_analyst) | `agents/news_analyst.py` l'appelle |
| Sentiment | `src/hermes/data/sentiment.py` (26 lignes) | ACTIF (agent sentiment_analyst) | wired dans `graph/orchestrator.py:19,34` |
| Régime Kaufman ER | `src/hermes/data/regime.py:20-33` (`detect_regime`) | ACTIF | Utilisé par `pulse.py:134`, `mean_reversion_regime.py:87`, `combination.py` (indirectement via efficiency_ratio) |
| Datalake historique (Stream 0, GELÉ) | `src/hermes/datalake/{core,run}.py` + sources (`binance*.py`,`coinglass*.py`,`deribit.py`,`hyperliquid*.py`,`fred.py`,`defillama.py`,`gold.py`) | **GELÉ depuis 2026-06-29** | `data/manifest/DATA_MANIFEST.md:1-5` : run_as_of 2026-06-29T22:45:38Z, 94 datasets (84 ok/9 skip/1 blocked), 1 422 506 lignes |
| Collecteur forward `skew` (Deribit 25Δ) | `src/hermes/forward/collect_skew.py` + `forward/run.py:79-104` | ACTIF EN PROD | dashboard live: `forward.skew` = 22 lignes, last_ts 2026-07-10T02:00 |
| Collecteur forward `top_lsr` (Binance long/short ratio, PIT, ~30j retention) | `src/hermes/forward/collect_top_lsr.py` + `forward/run.py:131-140` | ACTIF EN PROD | dashboard live: 666 lignes, 9 symboles (dont XAU/XAG!), last_ts=2026-07-10 — **NOTE**: uniquement accrual PIT, le DeadEndsLedger a tué son usage directionnel (mémoire ledger #25) |
| Collecteur forward `comex` (courbe COMEX) | `src/hermes/forward/collect_comex.py` + `run.py:107-119` | **NO_DATA en prod** | dashboard: `comex_curve`/`comex_spreads` = 0 lignes, status no_data |
| Collecteur forward `macro` (SEP vs Fed-Funds futures) | `src/hermes/forward/collect_macro.py` + `run.py:122-128` | **NO_DATA en prod** | dashboard: `macro_dot_vs_ff` = 0 lignes |
| Shadow `funding_squeeze_shadow` (accrual continu, l'ancre S2) | `src/hermes/forward/shadow_funding_squeeze.py` + `run.py:143-150` | ACTIF | dashboard: `funding_squeeze` = 7 lignes |
| Shadow `xs_reversion_shadow` | `src/hermes/forward/shadow_xs_reversion.py` + `run.py:48,153-161` (flag `WIRE_XS_REVERSION=False` par défaut) | **DÉFAUT REPO = OFF, mais prod montre 30 lignes 'collecting'** — incohérence non tranchée (cron externe lancé avec `--wire-xs-reversion` ?) | dashboard: `xs_reversion` = 30 lignes |
| **Mécanisme du cron forward** | Aucun import de `hermes.forward.run` dans `src/hermes/scheduler.py` (grep confirmé), aucun job GitHub Actions planifié (`.github/workflows/deploy.yml` = push-only), aucun CMD Docker ne l'invoque (`Dockerfile:64` = uvicorn seul) | **EXTERNE AU REPO** — la cadence "daily" du docstring (`forward/run.py:3`) n'est qu'une intention documentée, pas un fait câblé dans ce dépôt ; les données prod fraîches prouvent qu'un mécanisme existe ailleurs (VPS crontab non visible) | Fetch direct `/dashboard/data` du 2026-07-10 19:39 UTC |
| DeadEndsLedger check (informatif) | `forward/run.py:257-265` appelle `prereg.dead_ends_verdicts()` | ACTIF mais **INFORMATIF SEUL** — ne bloque jamais un collecteur | commentaire ligne 257-260 explicite; confirmé par dashboard `governance.dead_ends_verdicts` |



## 2. SIGNAUX — bibliothèque de skills, routeur de régime, jauges du pulse

Registre complet dans `src/hermes/skills/library.py:_build_registry()` (lignes 176-247) + `_build_cross_registry()` (250-275).

| Skill | Où (fichier:ligne) | État réel | Preuve |
|---|---|---|---|
| `funding_squeeze_continuation` (Stratégie #2, l'ancre) | `library.py:194-198` → impl `research/signals.py:237-260` | **ACTIF, dans allowed_skills** (`config/default.yaml:114`), `ALWAYS_ON` dans `strategies/combination.py:41` | joint-null p=0.0005 (contexte mission); gauge dashboard 'Funding #2' `pulse.py:52-55` |
| `funding_carry_harvest` (Stratégie #3) | `library.py:199-208` → impl `research/signals.py:279-296` | **ACTIF ET TRADABLE** (dans allowed_skills, `config/default.yaml:114`) — **le commentaire `library.py:202-206` ("DISPLAY-ONLY: always HOLD") est FAUX/PÉRIMÉ**: `signals.py:286-296` émet bien `BUY` dès `funding_bps >= 2.0`; routé structurellement vers une paire delta-neutral (short-perp+long-spot) dans `execution/paper.py:187-228` — jamais un ordre naked | Test `tests/test_research.py:238` (`test_funding_carry_harvest_arms_above_threshold_else_holds`) prouve que ça arme, pas que ça HOLD toujours |
| `mean_reversion_regime` (range, distinct) | `library.py:214-218` → impl `skills/mean_reversion_regime.py:67-124` | ACTIF, dans allowed_skills, `RANGE_SKILLS` `combination.py:40` | gate régime strict Kaufman ER (`regime=="range"` requis, ligne 103); paramètres figés Connors RSI(2)+BB(20,2)+MA200 |
| `ts_mom_vol_scaled` (ex-Stratégie #1 momentum) | `library.py:189-193` → impl `skills/ts_mom.py` | **RETIRÉ du slate** (pas dans `allowed_skills`), `TREND_SKILLS=[]` vide dans `combination.py:39` | `config/default.yaml:103-110` (bull-beta drag, LOO Δ Sharpe −0.146, joint-null p=0.507); gauge forcée `armed=False, retired=True` dans `pulse.py:39-51` |
| `breakout_vol_expansion` | `library.py:209-213` → impl `skills/library.py:115-139` | RETIRÉ (jamais dans allowed_skills ni combination.py buckets) | commentaire `combination.py:38` "Previously retired families" |
| `intraday_tod_btc` (BTC seul, ≤1 trade/jour) | `library.py:219-224` → impl `skills/intraday_tod.py` | RETIRÉ du slate live (calculé, non routé) | idem, listé retiré dans `combination.py:38` |
| `pairs_eth_btc` (cross-asset, Engle-Granger) | `library.py:270-274` (cross registry) → impl `strategies/pairs.py` | RETIRÉ | idem |
| `options_iv_dvol_reversion` (SHADOW, BTC/ETH) | `library.py:235-240` → `skills/newdata_shadow.py` | **SHADOW** — calculé/loggé chaque tick, JAMAIS routable (selective_gate rejette tout skill hors slate) | mémoire `hermes-newdata-survivors.md`: DVOL démasqué double-comptage, downgradé |
| `onchain_stablecoin_growth` (SHADOW, BTC/ETH) | `library.py:241-246` → `skills/newdata_shadow.py` | SHADOW, idem | même mémoire: stablecoin rétrogradé |
| Routeur de régime (L2) + hystérèse Kaufman ER | `strategies/combination.py:80-97` (`route_regime`, enter=0.4/exit=0.3) | **ACTIF EN CALCUL, SHADOW EN EXÉCUTION** — `annotate_active_and_blend`/`blend_live_shadow` (lignes 116-196) enrichissent le contexte mais NE sizent NI ne placent d'ordre (`shadow: True, size: 0.0` ligne 193-194) | docstring ligne 130-134: "SHADOW in live" |
| Blend inverse-vol (L3) | `combination.py:53-77` (`inverse_vol_weights`) | SHADOW (même statut que routeur) | — |
| Jauges du pulse horaire | `runtime/pulse.py:22-104` (`pulse_entry`) — gauges par sleeve (Momentum #1 retraité, Funding #2, Carry #3) + shadow gauges (lignes 84-93) | ACTIF, cron `pulse` horaire `0 * * * *` (`scheduler.py:102-108`, `config/default.yaml:151`) | `would_act`/`would_skill`/`reason` calculés via `selective_gate` sans exécuter (ligne 63-75) |



## 3. DÉCISION — orchestrateur multi-agents LLM, selective_mode, TRM verifier

Graphe: `src/hermes/graph/orchestrator.py:28-63` — flux linéaire/parallèle: `market_analyst ∥ news_analyst ∥ sentiment_analyst` → `fetch_lessons` → `bull ∥ bear` → `trader` → `risk_manager` → `portfolio_manager` (lignes 30-47).

| Agent LLM | Rôle réel | Où | Rôle LLM (config/default.yaml:157-169) |
|---|---|---|---|
| market_analyst | Snapshot technique (prix/ATR/regime) | `agents/market_analyst.py:24-100` | secondary |
| news_analyst | Résumé headlines CryptoPanic/Reddit | `agents/news_analyst.py` (12 lignes) | secondary |
| sentiment_analyst | Score sentiment | `agents/sentiment_analyst.py` (19 lignes) | secondary |
| bull/bear researcher | Débat contradictoire, alimenté par `fetch_lessons` (LessonsStore) | `agents/researcher.py:28-81` | primary |
| trader | Propose side/confidence/stop/tp — **ADVISORY seulement si selective_mode ON** | `agents/trader.py` (52 lignes) | primary |
| risk_manager | Override déterministe + TRM + calibration + sizing + risk engine | `agents/risk_manager.py:28-208` | primary |
| portfolio_manager | Décision finale d'exécution | `agents/portfolio_manager.py` (119 lignes) | primary |

**Qui décide VRAIMENT — la règle ou le LLM ?** → **LA RÈGLE**, de façon câblée et testée. `config/default.yaml:101` `selective_mode: true` (PAPER-FORWARD actif). Dans `risk_manager.py:41-80`: si `sel_cfg.enabled`, le signal déterministe (`drive_signal`, `risk/selective.py:142-160`) **remplace** la proposition du LLM (`proposal = TradeProposal(...)` ligne 61-65), et si aucun signal ne qualifie → HOLD forcé (ligne 77-80), quel que soit ce que le LLM a proposé. Le LLM devient strictement advisory. Preuve : `tests/test_selective.py` (21 tests, ex. `test_blocks_skill_outside_selective_slate`, `test_no_pyramid_blocks_when_already_holding_same_symbol_and_skill`).

**TRM verifier — tourne-t-il réellement ?** `config/default.yaml:172` `trm.enabled: true`. Mais `get_trm()` (`trm/verifier.py:159-170`) ne renvoie un `TRMVerifier` réel QUE si un checkpoint existe à `TRM_CHECKPOINT` (défaut `/data/trm/latest.pt`, `settings.py:58`) ET que torch/le module TRM s'importent (`trm/verifier.py:78-82,111-121`) ; sinon **fallback `NullTRMVerifier`** (pass-through, confidence=1.0, lignes 55-72). Aucun fichier `.pt`/checkpoint trouvé dans le repo local — statut réel du volume prod `/data/trm` **non vérifiable sans accès SSH** (caveat). Mécanisme downgrade-only confirmé câblé : `risk_manager.py:111-126` — le verdict TRM ne peut QUE (a) forcer HOLD si `used_trm AND refined.side==HOLD AND confidence>0.7`, jamais inverser BUY↔SELL ni upgrader HOLD→trade (commentaire ligne 107-110), et (b) plafonner (jamais relever) la confidence (ligne 125-126). |



## 4. EXÉCUTION — venues câblées, sizing, ordres/stops, stop-watcher

Factory: `src/hermes/execution/factory.py:32-60` (`get_exchange`, résolu par `stage_config.venue`).

| Venue | Où | État réel |
|---|---|---|
| `paper` | `execution/paper.py` (405 lignes) | **ACTIF — stage_0_paper courant** (`config/default.yaml:17-18`). Route `funding_carry_harvest` structurellement en paire delta-neutral short-perp+long-spot (`paper.py:187-228`) |
| `hyperliquid` | `execution/hyperliquid.py` (237 lignes) | DORMANT côté exécution réelle — c'est la venue CIBLE des stages 1-3 (`config/default.yaml:44,57,70`), mais le stage courant est stage_0 (paper) donc jamais appelée en prod actuellement |
| `bybit` | `execution/bybit.py` (150 lignes) | **BACKUP, non câblé dans la stage ladder** — aucun `stages.*.venue` du config ne pointe vers bybit ; adapter existe (position-tracking via journal, ligne 1-9) mais dormant |
| `binance` (exécution directe) | **ABSENT** — Binance n'apparaît QUE comme `data_venue` (klines/funding, `config/default.yaml:6`), aucun adapter d'exécution Binance dans `execution/factory.py` | **NON CÂBLÉ** — la directive mentionne Binance comme venue d'exécution ; le code ne l'implémente que comme source de données |
| `justmarkets` / MT5 (FX/métaux) | **ABSENT** — aucun fichier `src/hermes/execution/*justmarkets*` ni MT5 | **PAS DÉPLOYÉ** — confirmé par mémoire `hermes-fx-metals-expansion.md`: "PARKED, deploy nothing, don't stand up MT5" |

**Sizing** — Kelly fractionnaire + ATR + vol-target + corrélation: `risk/sizing.py:44-108` (`PositionSizer.size`). Kelly activé seulement après `MIN_TRADES_FOR_KELLY=100` trades (ligne 45,58-67); avant ça, floor confidence-scalé (lignes 76-78). `fractional_kelly=0.20` (`config/default.yaml:86`). Correlation haircut `sizing.py:9-16` (`1/sqrt(1+n*rho)`). Vol-target HAR-RV: `risk/volforecast.py` (`har_rv_forecast`, appelé `risk_manager.py:166-168`).

**Ordres/stops/TP** — calculés dans `risk/guardrails.py:209-218` (stop/tp depuis `stop_pct`/`take_profit_pct`); exécution paper via `execution/paper.py`.

**Stop-watcher** — `runtime/stop_watcher.py:30-76` (`tick`), cadence **30s exacte** confirmée `scheduler.py:86-92` (`IntervalTrigger(seconds=30)`). Ferme FIFO le plus ancien journal row par (symbol, skill_id) (docstring lignes 7-13). Déclenche `reflect_on_trade` après chaque clôture (ligne 70). Test: `tests/test_risk_engine.py` couvre les rejets, pas de test dédié stop_watcher trouvé dans la recherche (à vérifier séparément si besoin). |



## 5. RISQUE — caps, kill switches, guardrails, no-pyramid, corrélation

Moteur central: `src/hermes/risk/guardrails.py:94-227` (`RiskEngine.evaluate`), 12 règles dures listées dans le docstring (lignes 6-16).

| Protection | Ligne | Preuve (test) |
|---|---|---|
| Symbole hors univers → rejet | `guardrails.py:131-132` | — |
| Cold-floor equity ≤ $5 | `guardrails.py:138-140` | — |
| Kill switch quotidien (daily_loss_kill_pct, 5% stage_0) | `guardrails.py:142-150`, config `default.yaml:30` | `tests/test_risk_engine.py::test_daily_kill_switch_fires` |
| Max positions ouvertes (6 en stage_0) | `guardrails.py:152-155`, config `default.yaml:29` | — |
| No-pyramid (1 position/symbole) au niveau risk engine | `guardrails.py:156-157` | — |
| No-pyramid PAR (symbole, skill) au niveau selective gate | `risk/selective.py:88-91` | `tests/test_selective.py::test_no_pyramid_blocks_when_already_holding_same_symbol_and_skill`, `::test_no_pyramid_allows_different_skill_same_symbol` |
| Refus SELL sur stage spot-only (pas de short) | `guardrails.py:163-171`, `SPOT_ONLY_STAGES` ligne 97 | `tests/test_risk_engine.py::test_spot_sell_rejected` |
| Notional < minimum venue ($5 paper/bybit, $10 hyperliquid) | `guardrails.py:41,196-199` | — |
| Notional cap 5x equity | `guardrails.py:203-207` | — |
| Confidence < floor TRM → rejet | `guardrails.py:133-136` | — |
| Guard stage paper cohérence (refuse le trade réel si stage_0 mais exchange≠paper) | `guardrails.py:160-161` | — |
| Haircut corrélation (rho=0.85 univers BTC/ETH/SOL) | `risk/sizing.py:9-16`, appliqué `guardrails.py:191` | `tests/test_correlation_sizing.py` (3 tests) |
| Vol-targeting (scale down si HAR-RV > target_vol_per_bar) | `risk/volforecast.py`, appliqué `risk_manager.py:160-176`, `sizing.py:86-87` | — |
| Selective slate gate (skill hors allowed_skills → rejet) | `risk/selective.py:73-74` | `tests/test_selective.py::test_blocks_skill_outside_selective_slate` |



## 6. APPRENTISSAGE — calibration, meta-labeler, reflexion, weekly rerank, LessonsStore

| Composant | Où | État réel (prod) |
|---|---|---|
| Calibration isotonique de confiance | `risk/calibration.py:22-49` (`ConfidenceCalibrator`) | **IDENTITY (non-fitted) en prod** — `models.calibrator_fitted: false` (dashboard live 2026-07-10). Nécessite `min_samples=30` + 2 classes (win/loss) présentes (ligne 35); avec seulement 8 trades clôturés au journal (dashboard `journal.count=8`), reste identité |
| Meta-labeler (López de Prado, filtre P(win)) | `risk/meta_labeling.py:1-30` (LogisticRegression), branché SHADOW dans `risk_manager.py:177-187` | **SHADOW, non-fitted** — `models.meta_labeler_fitted: false` (dashboard). Ne gate JAMAIS un trade (docstring ligne 4: "never gating") |
| Nightly reflection (par trade + distillation journalière) | `learning/reflexion.py` (per-trade), `learning/nightly.py` (distillation LLM) | ACTIF, cron quotidien `0 23 * * *` (`scheduler.py:109-113`, `config/default.yaml:153`) — appelle aussi `reset_confidence_calibrator`/`reset_meta_labeler` (`nightly.py:11-12`) pour refit |
| Weekly skill rerank (Deflated Sharpe Ratio, anti-cherry-pick) | `learning/weekly.py` + `learning/stats.py` (PSR/DSR/MinTRL, Bailey & López de Prado 2014) | ACTIF, cron hebdo `0 22 * * 0` (`scheduler.py:114-118`) — archive un skill seulement si trace réelle ET absence statistique d'edge (docstring `weekly.py:1-8`) |
| Monthly compile (export DSPy/GEPA) | `learning/monthly.py:1-20` | **STUB/no-op** — exporte juste un training-set JSON pour itération OFFLINE manuelle par l'opérateur (docstring lignes 1-7); cron mensuel câblé (`scheduler.py:119-123`) mais ne fait rien d'automatique côté prompts |
| LessonsStore (mémoire épisodique vectorielle) | `memory/lessons.py:22-72` (Chroma via `memory/vector.py`) | ACTIF, branché sur `agents/researcher.py:10,28-29` (`fetch_lessons`, appelé à chaque tick, `graph/orchestrator.py:40`); config `lessons_top_k: 6` (`default.yaml:191`) |



## 7. GOUVERNANCE — registre déflation, DeadEndsLedger, préenregistrements, portes datées, stage-ladder

Module central: `src/hermes/research/s10_governance.py` (1484 lignes).

| Composant | Où | État |
|---|---|---|
| Registre d'essais par livre (déflation) | `s10_governance.py:471-` (`BookTrialRegistry`), fichier `governance/trials_by_book.json` | Repo local committé: `{cross:5, crypto:1910, fx_metals:53}`, `seeded:true`. **PROD LIVE diverge**: dashboard montre `{cross:5, crypto:1892, fx_metals:53}` — écart de 18 essais crypto entre le fichier committé et le registre réellement lu en prod (`HERMES_GOVERNANCE_DIR=/data/governance`, `docker-compose.yml`), preuve que ce ne sont PAS le même fichier |
| DeadEndsLedger — matching durci (2026-07-10 / 2026-07-03) | `s10_governance.py:1004-1102` (classe), `_match` lignes 1042-1065 | ACTIF. Règle durcie: le nom exact DOIT apparaître verbatim, OU ≥2 tags distincts, OU 1 tag ultra-spécifique (≥4 mots), avec ancrage sur limites de mots (`\b`) pour éviter les faux positifs substring (ex. "ofi" dans "profit") — commentaire lignes 1043-1051 référence l'incident `research_toplsr/JOURNAL_taker.md` du 2026-07-03 |
| Fichier ledger committé | `governance/dead_ends_ledger.json` | 38 entrées (`len(json.load(...))==38`), conforme au contexte mission |
| Vérification appelée en prod | `forward/prereg.py` → `dead_ends_verdicts()`, invoqué `forward/run.py:262-265` | **INFORMATIF SEULEMENT** — jamais dans le flux de contrôle de `run_forward` (commentaire explicite lignes 257-261); dashboard prod montre 2 verdicts "NOT-A-DEAD-END" (funding_squeeze_continuation, xs_funding_reversion) |
| Acquisition map (gouvernance $ data) | `s10_governance.py:1276-` (`AcquisitionMap`), `governance/acquisition_map.json` | présent, non audité en détail dans cette mission (hors scope profond) |
| Stage-ladder (4 stages) | `risk/stage.py:15,28-` (`StageManager`), gates dans `config/default.yaml:16-80` | **Avancement NON automatique** — docstring ligne 4: "l'opérateur doit fixer HERMES_STAGE manuellement après revue". Stage actuel confirmé prod: `stage_0_paper` (dashboard `stage`) |
| Gate `edge_paper_trades` | `config/default.yaml:39` (target 40) | **2/40** confirmé en direct (dashboard `gates.edge_paper_trades: {target:40, observed:2, passed:false}`) — ne compte QUE les trades clos dont le skill est dans l'`allowed_skills` COURANT (commentaire lignes 32-38), contre 8 trades clos au total (`journal.count:8`) |



## 8. INTERFACES — dashboard, chat anti-fabrication, manifest cron

| Composant | Où | État |
|---|---|---|
| Dashboard public | `src/hermes/api.py:156` (`GET /dashboard/data`), `:505` (page HTML) | ACTIF, public, confirmé `https://api.betterlifewithmk.com/dashboard/data` répond HTTP 200 avec stage/equity/gates/journal/forward/governance/scheduler (2026-07-10) |
| Chat opérateur | `src/hermes/api.py:651-679` (`POST /chat`, `GET /chat`) | ACTIF, protégé par API key (`Depends(require_api_key)`) |
| Contexte chat (assemblage anti-hallucination) | `api.py:516-595` (`_chat_context`) | Assemble stage/journal/gauges/gouvernance réels avant chaque appel LLM (ligne 517: "answers from reality, not vibes") |
| Règle anti-invention | `api.py:48-55` (dans `_CHAT_SYSTEM`, prompt système) | **MÉCANISME = INSTRUCTION DE PROMPT, PAS DE GARDE CODÉE** — la règle "réponds 'cette donnée n'est pas dans mon contexte' plutôt que d'inventer" (lignes 48-55) est une consigne au LLM, pas un filtre programmatique qui vérifie après-coup que la réponse ne contient aucun fait hors contexte. Par la doctrine R3 (mécanisme, pas déclaration) du fichier CLAUDE.md, ceci reste une garantie faible tant qu'aucun vérificateur post-génération ne l'impose |
| Manifest cron (jobs APScheduler) | `src/hermes/scheduler.py:79-124` | 7 jobs actifs confirmés en prod (dashboard `scheduler`): `stop_watcher` (30s), `market_tick` (4h, `0 */4 * * *`), `pulse` (horaire), `status_heartbeat` (horaire), `nightly_reflection` (23h), `weekly_rerank` (dim 22h), `monthly_compile` (1er du mois 21h). **Absent de cette liste : `forward.run`** (voir section 1) |
| Healthcheck | `Dockerfile:60-61`, `api.py:118` (`GET /health`) | ACTIF, confirmé `{"status":"ok"}` (2026-07-10) |



**Limites déclarées :**
- Le cron forward.run (data/forward/*) n'a pas de trigger identifiable dans le repo (ni scheduler.py, ni .github/workflows, ni docker-compose CMD) — sa cadence exacte en prod (probablement un crontab VPS externe) n'a pas pu être vérifiée par le code seul; seule la fraîcheur des données (dashboard /dashboard/data → forward.*.last_ts, interrogé en direct) confirme qu'il tourne.
- TRM: le repo local n'a pas de checkpoint /data/trm/latest.pt (chemin par défaut, settings.py:58); get_trm() (trm/verifier.py:159-170) tombe sur NullTRMVerifier en l'absence de checkpoint. Je n'ai pas d'accès SSH au VPS pour confirmer si un checkpoint existe réellement sur le volume /data monté en prod — 'TRM actif' reste donc une inférence par absence de preuve, pas une négative confirmée.
- Le comptage exact des lignes forward par flux (comex/macro = no_data en prod) suggère que ces deux fluxs ne collectent RIEN actuellement malgré leur présence dans default_collector_names — cause non investiguée (fetcher réseau mort, ou jamais invoqué par le cron externe).
- xs_reversion apparaît 'collecting' (30 lignes) dans la prod alors que WIRE_XS_REVERSION=False par défaut dans forward/run.py:48 — implique que le cron externe est invoqué avec --wire-xs-reversion, ou une version antérieure du flag; non tranché faute d'accès au crontab VPS.
- Analyse limitée à un repo local en lecture seule + un fetch HTTP du dashboard prod; aucune commande destructive, aucun commit.


---

# PARTIE II — LA CARTE DES ALGORITHMES (Markov, Simons, la littérature × 1910 essais)

**Synthèse :** Carte complète des familles d'algorithmes de la littérature quant × leur état chez Hermès, sourcée fichier:ligne / ledger / FINDINGS. Un seul pilier vivant sans réserve (S2 funding_squeeze_continuation, joint-null p=0.0005), un deuxième vivant conditionnel (mean_reversion_regime, gated Kaufman ER), le reste du corpus classique (HMM regime-switching, stat-arb façon Simons, TSMOM, cointegration, carry FX, microstructure market-making, positionnement, VRP/skew, ML feature-zoo) est soit MORT sur preuve (ledger #1-#38), soit en PORTE datée (métaux funding, top_lsr, skew 25Δ), soit structurellement inaccessible (capital/données/coût), jamais un vrai jamais-testé faute de curiosité.


## 1. Chaînes de Markov / HMM regime-switching (Hamilton 1989)

**Référence canonique** : Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle", Econometrica — régimes latents à switch markovien estimés par filtrage/Baum-Welch.

**Ce qu'Hermès en a** : un HMM a bel et bien été construit et testé — ce n'est PAS un jamais-testé. `src/hermes/research/s8_regime_engine.py` implémente `CausalGaussianHMM` (émissions gaussiennes diagonales, Baum-Welch, fenêtre causale, zéro look-ahead prouvé 5 façons) + un détecteur BOCPD + un bucket de volatilité (`research_streams/S8/FINDINGS_S8.md:1,15,29,39`). Les états sont **OOS-stables** : excès-sur-hasard +0.315 vs 200 permutations, p=0.005 (`FINDINGS_S8.md:15,69`) — le moteur en lui-même est validé comme mécanisme de segmentation.

Ce qui a été testé ET tué : utiliser ce HMM (ou BOCPD, ou un bucket de vol) comme **conditionneur/gate** du sleeve live `funding_squeeze_continuation`. Verdict : **aucun conditionneur ne bat le shuffle-null à p<0.05** — le meilleur (`state==trend`, n=38) plafonne à p=0.110 (sens correct, non significatif) ; `state==stress` (n=3) est sous-alimenté (`FINDINGS_S8.md:85-101`). Une extension `regime_aware_purged_splits` de la CPCV a aussi été tentée puis tuée (perm-p=0.135 = bruit) → ledger #32 `S8 regime conditioning of funding_squeeze (HMM/BOCPD)` (`governance/dead_ends_ledger.json:502-516`).

Ce qui existe aujourd'hui côté ROUTAGE réel (production) : PAS le HMM — un **Kaufman Efficiency Ratio** trivial et déterministe (`src/hermes/data/regime.py:1-33`), qui classe range/trend_up/trend_down sur une fenêtre de 48 barres et pondère la confiance des skills (0.5× hors-régime). Le HMM ne bat pas cet ER trivial de façon prouvée sur la tâche de conditionnement (`FINDINGS_S8.md:104-105`) — c'est pourquoi le routeur en production reste l'ER, pas le HMM.

**Statut** : MOTEUR VIVANT (mais non déployé comme conditionneur), CONDITIONNEMENT MORT#32. Le HMM sert aujourd'hui uniquement de features forward (`research_streams/S8/features/*.parquet`) pour re-test futur, size 0.

**Pour rouvrir** : un sleeve de base qui a DÉJÀ un edge standalone à conditionner (le conditionnement arbitre l'alpha existant, il ne le crée pas — leçon répétée sur X3, S8) — hors de portée tant qu'il n'y a qu'un seul pilier (S2).


## 2. Stat-arb façon Simons/RenTec (mean-reversion multi-signaux court terme, exécution massive)

**Référence** : le style Medallion (Renaissance Technologies) — myriade de signaux faibles à courte échéance, combinés statistiquement, exécutés à très haute fréquence avec un edge de microstructure et un coût marginal quasi nul (colocation, maker rebates, accès L2).

**Ce qu'Hermès en a** : 11 recherches indépendantes et rigoureuses convergent TOUTES vers la même impossibilité structurelle (mémoire `hermes-fast-edge-search-converged.md:10`) : "il n'existe pas d'edge fréquent, rentable, généralisant cross-actif dans OHLCV + funding + open-interest + positionnement au coût taker honnête (0.0015 RT)". Le mode d'échec récurrent : l'edge brut à cadence rapide est < 15 bps et le coût taker (15 bps RT, `docs/RESTRUCTURATION_2026-07.md:156`) l'enterre = whipsaw.

**Pourquoi le style Medallion est structurellement hors de portée ici — les chiffres** :
- **Coût d'exécution** : Hermès paie 15 bps RT taker (`docs/RESTRUCTURATION_2026-07.md:156`, `research_maker/*` maker fill-aware). RenTec négocie en interne à des fractions de bps grâce à la colocation et au maker rebate net-positif. Le maker chez Hermès a été testé fill-aware et est un MIRAGE d'anti-sélection : fill ~99% mais on remplit les perdants et on rate les gagnants (+59 à +204 bps) — ledger "Maker-execution free lunch" (`governance/dead_ends_ledger.json:309-322`). Sans rebate net, chaque signal doit dégager >15 bps net pour survivre — un seuil que RenTec n'a jamais à franchir sur son carnet interne.
- **Pas de L2** : aucune donnée order-book/tape n'est dans le lake (CoinGlass = agrégats journaliers/horaires, pas le carnet). Cont-Kukanov-Stoikov OFI, Kyle (1985) λ, Avellaneda-Stoikov market-making — tous nécessitent le carnet L2 ou la tape trade-level. Ce n'est pas un choix, c'est une porte fermée par le coût des données (voir famille 6).
- **Capital** : le sizing actuel est stage_0_paper, $103.74 (contexte opérateur). RenTec répartit des milliers de paris microscopiques sur un capital de plusieurs milliards — la loi des grands nombres lisse le bruit d'exécution. À $103, une seule position domine le portefeuille ; il n'y a pas de "volume" pour amortir un edge de 2-5 bps par pari.
- **Latence/infra** : le lake est en cadence 1d/4h (CoinGlass REST), pas en microsecondes ; le routeur LLM (DeepSeek V4) ajoute des secondes de latence par décision — incompatible par construction avec un edge qui vit sur des microsecondes.

**Statut** : STRUCTURELLEMENT FERMÉ, pas MORT-par-manque-d'essai — 11 recherches ont exploré l'espace (lead-lag, cointégration rapide, momentum court, mean-reversion intraday, positionnement, maker) et toutes meurent sur le même mur de coût, pas sur un défaut de conception.

**Pour ouvrir** : un axe de données radicalement nouveau (tape CME futures intraday réelle, carnet L2 payant) ET un budget de capital d'un tout autre ordre de grandeur ET l'abandon du LLM-in-the-loop pour les décisions à cette cadence — aucun des trois n'est sur la table aujourd'hui.


## 3. Momentum / TSMOM (Moskowitz-Ooi-Pedersen 2012)

**Référence** : Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum", JFE — continuation du signe du rendement passé (12 mois typiquement) à travers classes d'actifs.

**Ce qu'Hermès en a** : testé et déployé, puis retiré sur preuve.
- Crypto : `ts_mom_vol_scaled` (Strategy S1) était live puis retiré. Chiffres post-fix look-ahead (`docs/RESTRUCTURATION_2026-07.md:18-19,80-90,302-305`) : LOO-Δ Sharpe = **−0.146** (le livre gagnerait +0.146 en le retirant), joint-null panier S1 **p=0.507** (post-fix, contre 0.693 avant fix — toujours indiscernable du hasard), ρ(S1,S2)=0.348 (même risque, pas diversifiant). Retiré du slate live (`config/default.yaml`, `combination.py:TREND_SKILLS=[]`, `risk/selective.py:_CONVICTION_GATED=set()`).
- Or (gold TSMOM) : ledger #1 — "le null MC à entrée aléatoire BAT la stratégie ; net-NÉGATIF sur 26 ans ; perd le bear 2011-15. Une entrée longue aléatoire capte déjà le drift de l'or — les règles de trend n'ajoutent rien" (`governance/dead_ends_ledger.json:3-18`).
- 33-futures trend (yfinance, mémoire `hermes-cta-data-scout.md`) : passe le null book-level (p=0.011) mais ÉCHOUE l'OOS-par-date (≈0 après 2016) et la déflation (grid-max 0.28 < null-max 0.42) — mirage de sur-ajustement de grille, pas un edge.

**Statut** : MORT#S1-retrait (crypto), MORT#1 (or), MORT (33-futures trend, déflation échouée).

**Pour rouvrir** : une confirmation non-prix qui de-bêtaïfie le drift séculaire de l'actif sous-jacent (condition de résurrection ledger #1) — jamais trouvée à ce jour.


## 4. Mean reversion (Poterba-Summers 1988 ; Engle-Granger cointégration ; Avellaneda-Lee stat-arb)

**Référence** : Poterba & Summers (1988) "Mean Reversion in Stock Prices" ; Engle & Granger (1987) cointégration ; Avellaneda & Lee (2010) stat-arb PCA/ETF.

**Ce qu'Hermès en a** :
- **`mean_reversion_regime` — VIVANT (slate live)**. `src/hermes/skills/mean_reversion_regime.py:67-124` : RSI(2) Connors + Bollinger %B(20,2) + filtre MA200, strictement **gated sur régime range** via `detect_regime` (Kaufman ER) — thèse explicite dans le docstring (ligne 3-9) : "la réversion à la moyenne n'a un edge QU'EN range", correction du piège de l'ancien `mean_reversion_atr` qui tradait la réversion en tendance. Paramètres figés a priori (aucune optimisation), confiance fixe 0.6. Fait partie du slate live actuel (`docs/RESTRUCTURATION_2026-07.md:54`).
- **Pairs / cointégration — MORT#16**. "Le tueur est le half-life (structurel), pas le coût : half-lives de spread 20-99 JOURS (gold-silver ~10j). Pas de réversion rapide à récolter dans un horizon de détention honnête" (`governance/dead_ends_ledger.json:242-257`). Confirmé aussi crypto (ETH/BTC pairs, mémoire `hermes-fast-edge-search-converged.md:14` : cointégré mais half-life 24-70j, continue à cadence rapide au lieu de revenir, ~38 trades/an à cadence lente = trop peu).
- **Mean-reversion OHLCV rapide (intraday)** — MORT#15, inclus dans "Fast/any-cadence single-asset OHLCV families" (`governance/dead_ends_ledger.json:224-240`) : whipsaw net du coût honnête.

**Statut** : VIVANT (mean_reversion_regime, forme range-gated lente) ; MORT#16 (pairs/cointégration, tuée par le half-life) ; MORT#15 (mean-reversion intraday rapide).

**Pour rouvrir les formes mortes** : une paire dont le half-life de spread est réellement court (jours, pas semaines) — jamais rencontrée sur l'univers testé (gold-silver, ETH-BTC).


## 5. Carry (Koijen, Moskowitz, Pedersen, Vrugt 2018 "Carry")

**Référence** : Koijen, Moskowitz, Pedersen, Vrugt (2018), "Carry", JFE — prime de portage generalisée (funding, roll-yield, dividend yield, taux) à travers classes d'actifs.

**Ce qu'Hermès en a** :
- **`funding_carry_harvest` (S3) — VIVANT mais HOLD-only** : "0 trade à ce jour" (`docs/RESTRUCTURATION_2026-07.md:94`), delta-neutral funding carry (mémoire `hermes-strategy2-funding.md`, `hermes-fast-edge-search-converged.md:18`). Fait partie du slate live, size passif.
- **Carry FX/G10 — MORT** (mémoire `hermes-cta-carry.md`) : décomposition carry-accrual +38.5%/20 ans MAIS drift spot rend −23.2% (60% de l'accrual mangé, UIP partielle) ; Sharpe brut 0.12 ; **null rank-shuffle p=0.28** (le rang de carry n'apporte rien vs un split $-neutre aléatoire) ; même crash-protégé (vol_gate, Sharpe→0.27, OOS→+0.54) le null reste p≈0.073 (stable 5 seeds), DSR 0.46, jamais <0.05 ; crash 2008 −18% confirmé (`docs/RESTRUCTURATION_2026-07.md:174-175` confirme "carry G10 rank-shuffle p=0.28 même crash-protégé, carry cross-sectionnel p=0.804").
- **Roll-yield (crude → CAD/AUD)** — MORT#8 : les deux jambes net-négatives, le null MC n'est battu sur AUCUNE des 18 configs, signe du roll-yield inversé (`governance/dead_ends_ledger.json:113-128`).
- **Funding term-spread littéral (H2)** — MORT#21 : edge brut trivial et signe inversé au coût ; se réduit soit à "la persistance du funding élevé" (déjà S2) soit à "acheter les creux" (bêta haussier) (`governance/dead_ends_ledger.json:323-336`).

**Statut** : VIVANT-HOLD (funding_carry_harvest crypto) ; MORT (carry G10 FX p=0.28) ; MORT#8 (roll-yield) ; MORT#21 (funding term-spread littéral).

**Pour rouvrir le carry FX** : aucune condition de résurrection crédible identifiée — le UIP mange 60% de l'accrual par construction, ce n'est pas un problème de données.


## 6. Microstructure / market-making (Avellaneda-Stoikov 2008 ; Kyle 1985 ; OFI Cont-Kukanov-Stoikov 2014)

**Référence** : Kyle (1985) modèle d'impact de prix informé ; Avellaneda & Stoikov (2008) market-making à inventaire optimal ; Cont, Kukanov, Stoikov (2014) Order Flow Imbalance.

**Ce qu'Hermès en a** :
- **Maker/market-making complet** — testé fill-aware (`research_maker/maker_fill_test.py` cité mémoire `hermes-fast-edge-search-converged.md:15`) : "free lunch" = MIRAGE d'anti-sélection, fill ~99% mais on remplit les plats/négatifs et on rate les gros gagnants (+59 à +204 bps). 0/7 actifs cross-asset survivent. Ledger "Maker-execution free lunch" (`governance/dead_ends_ledger.json:309-322`).
- **OFI/CVD depuis le volume taker SPOT** — MORT#19 : redondant avec le momentum-prix (ρ 0.77-0.96) ou colinéaire au funding ; la seule config orthogonale meurt au coût honnête 2×RT (`governance/dead_ends_ledger.json:293-307`).
- **Taker perp imbalance (aggrégat CoinGlass, forme CONTINUATION)** — MORT#38 (2026-07-10, le plus récent) : joint-null 0.6225-0.6530 sur 5 seeds (le hasard bat la stratégie ~63% du temps), OOS-by-date basket −0.110, DSR=0.000215 (`governance/dead_ends_ledger.json:601-615`).
- **Un vrai market-making Avellaneda-Stoikov complet (spread bid/ask optimal, gestion d'inventaire, quotes actives)** — JAMAIS TESTÉ, et **structurellement fermé**, pas par manque d'idée : ça exige le carnet L2 (profondeur, empilement des ordres) qu'aucune source du lake ne fournit (CoinGlass = agrégats OHLCV/funding/OI/taker-volume, pas le carnet). Kyle λ et le vrai OFI (Cont et al., au niveau tick) nécessitent la même donnée absente.

**Statut** : MORT (maker free-lunch), MORT#19 (OFI/CVD spot), MORT#38 (taker perp continuation) ; market-making complet = PORTE fermée par la donnée, jamais testé faute d'accès, pas faute d'intérêt.

**Pour ouvrir le vrai market-making** : un flux L2 order-book payant (Binance fournit un flux de profondeur mais à un coût/complexité d'ingénierie non budgété) — condition de résurrection explicitement notée pour le côté OFI : "tape CME futures intraday réelle / CVD trade-level" (`governance/dead_ends_ledger.json:299,608`), jamais achetée à ce jour.


## 7. Positionnement / sentiment (COT ; long-short ratio)

**Référence** : littérature CFTC Commitment of Traders (positionnement contrarien des extrêmes) appliquée ici aux ratios long/short CoinGlass.

**Ce qu'Hermès en a** — trois morts distincts + une porte de collecte :
- **B1 COT positioning-extreme (FX/métaux)** — MORT#2 : ne généralise à AUCUN des 6 instruments testés ; seul l'argent bat le null MC (p=0.011) mais échoue OOS-Sharpe>0.3 ; la jambe courte combat le bull séculaire (`governance/dead_ends_ledger.json:19-34`).
- **Top-LSR contrarian (crypto)** — MORT#26 (S2.1) : la ρ(top_lsr, forward)=−0.27 était un mirage ~30 jours ; sur l'historique complet ρ≈0 ; le signe s'inverse bull/bear ; ne généralise à aucun instrument (`governance/dead_ends_ledger.json:406-421`).
- **Global-LSR (tous comptes) contrarian** — MORT#37 (2026-07-02) : panier date-level échoue les DEUX critères simultanément : joint-null MC p∈[0.1155,0.1415] (5 seeds, mauvais côté de la barre) ET OOS-by-date Sharpe=−0.1472 (≤0, 5 seeds) ; DSR vs N=1909 ≈2.5e-05 (`governance/dead_ends_ledger.json:585-600`).
- **top_lsr — collecte active, PORTE datée** : le module documente explicitement que le ledger n'interdit que l'usage DIRECTIONNEL (le S2.1 mort) — la collecte pure rend testable un futur axe sur historique profond (`docs/RESTRUCTURATION_2026-07.md:219-226`). Re-gate ≥ 2027-07, point d'étape 2027-01.

**Statut** : MORT#2 (COT FX/métaux), MORT#26 (top-LSR crypto), MORT#37 (global-LSR) ; collecte top_lsr = PORTE 2027-07.

**Pour rouvrir** : "une axe de positionnement à signe forward stable à travers les régimes et ≥2 instruments à canal distinct" (condition de résurrection ledger #37) — la forme quotidienne/directionnelle est épuisée, seule une forme intraday ou cross-venue serait un axe nouveau (non testé).


## 8. Vol/options (VRP, skew — Carr-Wu 2009 ; Bakshi-Kapadia 2003)

**Référence** : Carr & Wu (2009) variance risk premium ; Bakshi & Kapadia (2003) delta-hedged option returns / skew.

**Ce qu'Hermès en a** :
- **VRP-DVOL comme gate de sizing (4.3)** — MORT (négatif propre, `research_streams/S4/FINDINGS_S4.md:79-108`) : "le gate VRP-DVOL n'améliore PAS S2 hors-échantillon", abandonné sur les 4 formes testées (binaire/linéaire × signe) ; VRP moyen +8.3 pts de vol positif 80% des jours (le carry de vol existe) mais n'est PAS exploitable en gate directionnel/sizing → ledger #29 (`governance/dead_ends_ledger.json:452-466`, Δ+0.045 ≈ barre de déflation, shuffle-p 0.14-0.18, seulement 13 trades OOS).
- **Surface options grid (VRP/TERM/VOV/RSKEW, 1024 configs)** — MORT#23 : grid-max Sharpe 0.1934 < E[max|1024]=0.1950, mort de sur-ajustement de grille classique ; RSKEW-fade recouvre à ~34% le DVOL déjà mort (`governance/dead_ends_ledger.json:353-369`).
- **Skew 25Δ Deribit (4.1)** — "CHRONO DÉMARRÉ, NON BACKTESTABLE" (`research_streams/S4/FINDINGS_S4.md:114-120`) : Amberdata non acheté, `rr25/skew` = NaN sur le lake forward, échantillon insuffisant par construction, interdiction explicite de fabriquer un historique — **PORTE datée 2027-01→2027-07** (`docs/RESTRUCTURATION_2026-07.md:64,214`).

**Statut** : MORT#29 (VRP-sizing), MORT#23 (grille 1024-configs) ; skew 25Δ = PORTE 2027 (donnée payante Amberdata non achetée, pas un refus de tester).

**Pour ouvrir le skew** : acheter Amberdata (ou attendre que `deribit_surface_snapshot` accumule assez d'historique forward, actuellement 1 ligne/jour depuis 2026-06) — coût/temps, pas d'obstacle de conception.


## 9. ML (López de Prado : meta-labeling, triple-barrier, CPCV, DSR ; Gu-Kelly-Xiu)

**Déjà intégrés côté process — ce n'est pas un jamais-testé, c'est de l'infrastructure en production** :
- **Meta-labeling** : `src/hermes/risk/meta_labeling.py:36-73` (`MetaLabeler`, features de régime + confiance, shadow no-op jusqu'à 50 trades bi-classe), branché dans `agents/risk_manager.py`, `validation/bootstrap.py`, `validation/meta_sizing.py`.
- **Triple-barrier labeling** : `src/hermes/validation/triple_barrier.py:14` (`triple_barrier_label`), utilisé par `risk/timeexit.py`, `research/s3_universe.py`, `research/harness.py`, `forward/shadow_funding_squeeze.py`.
- **CPCV (Combinatorial Purged Cross-Validation)** : `src/hermes/validation/cpcv.py:33` (`combinatorial_purged_splits`), extension régime-aware ligne 87 (`regime_aware_purged_splits` — MORTE en tant que conditionneur, voir famille 1) ; `TrialRegistry` (ligne 151) tient le compteur de déflation.
- **DSR (Deflated Sharpe Ratio, Bailey-López de Prado)** : `src/hermes/learning/stats.py:31-120` (`sharpe_ratio`, `probabilistic_sharpe_ratio`, `expected_max_sharpe`, `deflated_sharpe_ratio`, `score_skills`) — utilisé dans `validation/book.py`, `research/s10_governance.py`, `research/s10_jewel_gate.py` pour arbitrer TOUTES les décisions de ce document (S2 DSR, book DSR/PBO, etc.).
- **Gu-Kelly-Xiu ML combination (feature-zoo GBM/LogReg)** — MORT#13 (D1) : "NO_EDGE : précision OOS ≤ le taux de base long ; +15bps net = juste le coût-d'être-long (bêta haussier) ; le null MC combiné n'est PAS battu du tout ; PBO=0 non pertinent (pas de compétence)" (`governance/dead_ends_ledger.json:190-207`), sur 8 features orthogonales causales, 6467 barres XAU, CV purgée/combinatoire.

**Statut** : infra ML VIVANTE et câblée dans TOUT le processus de gouvernance (c'est le mécanisme qui prononce chaque verdict de cette carte) ; l'application "prédire le signal via ML sur un zoo de features" = MORT#13.

**Pour rouvrir le feature-zoo ML** : une feature réellement nouvelle absente du zoo à 8 (order-flow CME intraday, skew OTC, inventaire COMEX, un vrai panneau actual/forecast) — ré-exécuter du ML sur les mêmes 8 features est explicitement interdit (`governance/dead_ends_ledger.json:197`).


## 10. Régime funding (l'ancre S2) — le seul survivant sans réserve

**Ce que c'est** : `funding_squeeze_continuation` — z-score du funding rate perp sur fenêtre 180 barres, seuil |z|≥2, hold 24 barres, entrée décalée d'1 barre (causal), coût taker 15 bps RT (`docs/RESTRUCTURATION_2026-07.md:186-188`).

**Les chiffres exacts, post-fix look-ahead (`docs/RESTRUCTURATION_2026-07.md:300-305`)** :
- Joint-null panier basket (max sur 5 seeds) : **p=0.0005** (avant fix : 0.0235 — le fix a RENFORCÉ le résultat, pas affaibli).
- Seul contributeur positif au LOO-Δ du livre : **+0.015 Sharpe**, bear-positif.
- Per-trade Sharpe 0.26–0.37 ; OOS-40 ≈ 0.28–0.33 (`docs/RESTRUCTURATION_2026-07.md:237`).
- Fréquence : **~13 événements/an** sur panier BTC/ETH/SOL dé-chevauché (115 événements / 8.85 ans) (`docs/RESTRUCTURATION_2026-07.md:240`).
- Rendement espéré annuel : ≈+2.1% au sizing 0.5%/trade, ≈+8.3% au sizing 2%/trade (jamais atteint, stage_0) (`docs/RESTRUCTURATION_2026-07.md:241-243`).
- Variance : ~13% de probabilité qu'une année soit négative (modèle gaussien — sous-estimé, SOL bear −397bps/évt sur 5 événements) (`docs/RESTRUCTURATION_2026-07.md:244-245`).
- N registre crypto au moment du dernier calcul : 1910 essais (`governance/trials_by_book.json:4`) — la déflation DSR est calculée contre ce N réel, pas contre le sous-ensemble de la stratégie.

**Statut** : VIVANT — le SEUL pilier du livre sans réserve. Toute expansion prévue (métaux XAU/XAG, top_lsr, cross-venue) ajoute de la FRÉQUENCE d'événements sur ce même axe funding, pas un nouvel edge (`docs/RESTRUCTURATION_2026-07.md:248-250`).


## 11. Event-driven (unlocks, listings, macro FF/SEP) — les seams non câblés

**Unlocks (déblocages de tokens)** : `research_breadth/SCOUT_token_unlocks_feasibility.md:21-23,106-115` — verdict : "AUCUNE source unique gratuite avec profondeur suffisante" ; The Tie et les fournisseurs équivalents (100k+ événements) bloquent l'historique derrière un **paywall** — changement de posture commerciale généralisé de l'industrie, pas une lacune ponctuelle. Statut = PORTE payante, décision opérateur ($/mois) en attente, pas un mort de la thèse.

**Listings (nouvelles cotations)** : AUCUN fichier de recherche dédié trouvé dans le repo (`find -iname "*listing*"` ne retourne que des artefacts de dépendances Python, rien de métier) — **jamais testé, aucune trace d'un scout ou d'un pré-enregistrement**. Contrairement aux familles ci-dessus, ce n'est ni MORT ni explicitement en PORTE : c'est un blanc de couverture, pas une fermeture actée.

**Macro FF/SEP (ForexFactory economic-surprise / Fed SEP)** : deux échecs distincts déjà actés —
- ForexFactory economic-surprise index : MORT#6 (B4), "DATA-LOCKED, pas falsifié sur le fond : le mirroir gratuit est current-week-only sans champ `actual`, site Cloudflare-403, ALFRED n'a pas de prévision consensuelle" (`governance/dead_ends_ledger.json:82-97`).
- `comex dated contracts / macro SEP-FF` : seams **`NotImplementedError`**, priorité basse (l'axe métaux se joue déjà sur le funding perp, déjà couvert) (`docs/RESTRUCTURATION_2026-07.md:212`) — code stub présent, jamais implémenté, décision de priorisation explicite, pas un échec de test.

**Statut** : unlocks = PORTE payante (paywall, décision $ opérateur) ; listings = JAMAIS-TESTÉ réellement ouvert (aucun obstacle de donnée identifié — Binance annonce ses listings publiquement, gratuitement) ; macro FF = MORT#6 (données), SEP-FF = seam stub non prioritaire (choix, pas un mur).

**Ce qu'il faudrait pour ouvrir listings** : c'est la famille la MOINS chère à ouvrir de toute cette carte — les annonces de listing Binance sont publiques et gratuites (API annonces), aucun paywall, aucune donnée L2 requise. Un scout dédié (mesurer le drift post-annonce, le pump-and-dump, avec un panier d'événements dé-chevauché comme les autres axes) serait un candidat légitime pour la prochaine recherche — actuellement non fait, pas par barrière mais par manque de bande passante de recherche.


**Limites déclarées :**
- Ledger dead_ends_ledger.json contient 38 entrées nommées (vérifié par comptage programmatique) mais certaines regroupent plusieurs essais (ex. entrée #25 regroupe 6 frontières pré-vetées) — le compteur trials_by_book.json (N=1910 pour crypto au 2026-07-10 selon contexte opérateur) est distinct du nombre d'ENTRÉES du ledger et compte les CONFIGURATIONS testées (ex. options_surface_grid=1024, funding_discovery_walkforward_sweep=648), pas les familles.
- Le fichier trials_by_book.json lu ce jour affiche counts.crypto=1910 en cohérence avec le contexte opérateur, mais le detail 'provenance' plafonne encore visiblement à des sommes antérieures (ex. dernier bump taker_perp_prereg n=1) — je n'ai pas recalculé la somme exacte des 13 sources listées pour crypto, je rapporte le compteur top-level tel que lu dans le fichier.
- mean_reversion_regime n'a pas de p-value/joint-null dédiée trouvée dans le repo (contrairement à S1/S2) — son statut VIVANT s'appuie sur sa présence actuelle dans le slate live (docs/RESTRUCTURATION_2026-07.md:54) et son design anti-piège documenté en docstring, pas sur un test statistique séparé que j'ai localisé ; à vérifier si un tel test existe ailleurs (research_crypto2/ non exploré exhaustivement).
- La référence 'Cont et al. OFI 2014' est une référence canonique de littérature citée par la mission ; je n'ai pas trouvé de fichier Hermès la nommant explicitement — le rapprochement avec le MORT#19 (OFI/CVD spot) est mon inférence, pas une citation directe du repo.
- Je n'ai pas ouvert exhaustivement tous les FINDINGS_*.md (S1, S5, S6, S7, S9, S10, ALPHA, BETA, FORWARD) — la carte s'appuie sur RESTRUCTURATION_2026-07.md, dead_ends_ledger.json, trials_by_book.json, S4 et S8 FINDINGS, et les fichiers source cités ; des détails supplémentaires pourraient exister dans les FINDINGS non lus mais ne changeraient pas les verdicts déjà tranchés par le ledger.


---

# PARTIE III — L'AUDIT DE SÛRETÉ (mécanismes prouvés, gaps honnêtes)

**Synthèse :** Audit sûreté du système Hermès (stage_0_paper, $103.74, 2026-07-10). Verdict : l'architecture de sûreté DÉCLARÉE est réelle et en grande partie prouvée par le code + les tests (stage-ladder manuel-only, daily-loss-kill testé RED/GREEN, TRM downgrade-only par construction, S3 structurellement non-nu en paper, chat sans tool-calling). Mais l'audit a trouvé, par EXÉCUTION RÉELLE (pas par lecture seule), un défaut concret non déclaré : le garde-fou « no-pyramid » de guardrails.py bloque TOUTE seconde position sur un symbole déjà tenu — y compris par un AUTRE skill — ce qui contredit et neutralise silencieusement le design « sleeves concurrents » que selective.py/tests déclarent et testent en isolation. Sept gaps honnêtes sont confirmés absents par grep+lecture : aucun blackout macro, aucun circuit-breaker de dérive equity au-delà du kill quotidien, le routage delta-neutre de S3 n'existe qu'en paper (pas sur Hyperliquid/Bybit), le fallback LLM OpenRouter n'est pas vérifiable depuis ce dépôt (vit dans l'UI Dokploy), et le découplage CI/Dokploy a un exemple daté concret (commit rouge ccd741d, 2026-07-10 16:31, corrigé 36s plus tard) où un déploiement Dokploy indépendant du CI aurait pu shipper du code non testé.


## 1. Étages de capital (stage-ladder)

**Mécanisme** : 4 stages (`stage_0_paper→stage_1_micro→stage_2_normal→stage_3_perps`), `HERMES_STAGE` env var, aucune écriture programmatique.
**Code** : `src/hermes/settings.py:28` (`stage: str = Field("stage_0_paper", alias="HERMES_STAGE")`), validé par une liste blanche `src/hermes/settings.py:138-143`. `src/hermes/risk/stage.py:27-57` (`StageManager.evaluate()`) est une fonction PURE — elle retourne un `StageReport.can_advance` (bool) mais **ne mute jamais** `settings.stage` ni aucun état. Vérifié par `grep -rn "HERMES_STAGE\s*=\|settings.stage\s*="` sur tout `src/hermes` : **zéro résultat** — aucun code, même mort, n'écrit la stage. C'est un manuel-only vérifié par absence, pas par déclaration.
**Gates réels** (`config/default.yaml:31-41`) : `edge_paper_trades:40` (compte SEULEMENT les trades clôturés dont le skill est dans le slate courant — `TradeJournal.count_closed_by_skills`, pas tout trade comme l'ancien gate `paper_trades:200`), `walk_forward_sharpe:1.0`, `max_drawdown_pct:15.0` (borne supérieure, `src/hermes/risk/stage.py:64`).
**État prod vérifié** (`curl https://api.betterlifewithmk.com/status`, 2026-07-10) : `"stage":"stage_0_paper"`, `"stage_advance_summary":"BLOCKED: stage_0_paper → stage_1_micro; edge_paper_trades=2/40✗"` — cohérent avec le mandat (gate 2/40).
**Test qui mord** : aucun test dédié à `StageManager` trouvé (`find tests -iname "*stage*"` = vide) — la garantie « aucune auto-promotion » repose sur l'absence de code d'écriture, pas sur un test qui la prouverait par mutation (ex.: appeler `evaluate()` avec tous les gates au vert et vérifier que `settings.stage` ne change PAS). Gap de couverture mineur mais réel (R2/R3).


## 2. Garde-fous par trade

| Mécanisme | Code | Test qui mord | État prod |
|---|---|---|---|
| Risk cap 0.5% (stage_0) | `config/default.yaml:25`, appliqué `src/hermes/risk/guardrails.py:173-175,196-199` | `tests/test_min_notional.py` (notionnel), aucun test dédié au cap 0.5% lui-même | actif (stage_0) |
| Kelly fractionnaire 0.20 | `src/hermes/risk/sizing.py:30,58-67,74-75` (`kelly = max(0, (p*b-q)/b)`, puis `fractional_kelly*kelly*confidence`) | pas de test trouvé sur `PositionSizer.size` avec Kelly actif | **INACTIF EN PROD** — `MIN_TRADES_FOR_KELLY=100` (`sizing.py:45,58`) alors que le journal ne compte que 8 trades clos (`/dashboard/data`, `journal.count:8`) ; le sizing utilise donc la branche cold-start (`sizing.py:76-78`), pas Kelly |
| Sizing ATR | `sizing.py:52-55` (`stop_dist_pct = max(0.3%, 2*ATR/price)`), plancher anti-NaN/ATR=0 explicite | pas de test dédié trouvé | actif |
| Conviction/self-sélection | `src/hermes/risk/selective.py:28` — `_CONVICTION_GATED` est **vide** depuis le retrait de `ts_mom_vol_scaled` (2026-07-02) ; mécanisme câblé mais dormant | `tests/test_selective.py:48-60` (réactive le mécanisme via `monkeypatch` pour le tester) — preuve par mutation correcte | dormant (aucun skill du slate actuel n'a besoin du plancher) |
| No-pyramid | `selective.py:88-89` (per `(symbol, skill_id)`) **ET** `guardrails.py:156-157` (per `symbol` SEUL, tous skills confondus) | `tests/test_selective.py:92-117` prouve le no-pyramid PAR SKILL fonctionne ; **AUCUN test n'exerce `RiskEngine.evaluate()` avec deux positions de skills différents sur le même symbole** | **DÉFAUT CONFIRMÉ PAR EXÉCUTION** (voir section dédiée ci-dessous) |
| Max positions 6 | `config/default.yaml:29` (stage_0 seulement ; 3/5/5 aux stages 1-3) | pas de test trouvé sur `max_open_positions` | actif |
| Corrélation haircut | `sizing.py:9-16` (`1/sqrt(1+n_existing*rho)`, `rho=0.85` statique, `config/default.yaml:90`) appliqué APRÈS le clamp du risk_pct (`guardrails.py:191`, `sizing.py:82-87`) | pas de test unitaire dédié trouvé pour `correlation_haircut()` isolément (formule simple, mathématiquement correcte — diversification-scaling standard) | actif ; `rho` est une constante de config, jamais mesurée en direct sur les rendements réels du book |

**Cold-floor absolu** : `guardrails.py:139-140` — refuse tout trade si `equity < $5` (5% du capital de départ $100). C'est le SEUL frein sur une dérive lente d'equity (voir gap #7).


## DÉFAUT CONFIRMÉ (par exécution) — le no-pyramid cross-skill neutralise le design « sleeves concurrents »

**Constat** : `src/hermes/execution/paper.py:63-65` et `src/hermes/runtime/stop_watcher.py:9-12,32-34` déclarent explicitement (commentaires + code de fermeture) que DEUX stratégies (ex. `funding_squeeze_continuation` et `mean_reversion_regime`) doivent pouvoir tenir CHACUNE une position sur le MÊME symbole en même temps — le book est clé `symbol::skill_id`. `tests/test_selective.py:109-117,192-202` testent et NOMMENT cette invariance (« THE fix: momentum holding BTC must NOT prevent funding from being driven on BTC the same tick »).

Mais `src/hermes/risk/guardrails.py:156-157` — la porte finale avant tout ordre — rejette **toute** proposition sur un symbole dès qu'**une** position quelconque y est déjà ouverte, sans regarder `skill_id` :
```
if any(p.symbol == proposal.symbol for p in positions):
    return self._hard_reject(f"already have open position on {proposal.symbol}")
```
**Preuve par exécution réelle** (script exécuté dans ce dépôt, `research_toplsr/scratch_inventory/repro_pyramid.py`) : une position `ts_mom_vol_scaled` ouverte sur BTC/USDT:USDT, puis une proposition `funding_squeeze_continuation` BUY sur le MÊME symbole →
```
DECISION: REJECTED | already have open position on BTC/USDT:USDT
```
Cela contredit directement l'intention documentée et testée à l'échelon `selective.py` (qui, lui, aurait laissé passer la proposition). Le `drive_signal` gate-aware (`risk_manager.py:47-80`) sélectionne correctement le signal funding quand momentum tient déjà le symbole — mais son travail est ANNULÉ une étape plus loin par le check `guardrails.py:156-157`, qui n'a jamais été mis à jour pour l'invariant per-(symbol, skill).
**Conséquence concrète** : S2 (`funding_squeeze_continuation`, l'ancre validée) et `mean_reversion_regime` partagent l'univers BTC/ETH/SOL/AVAX/SUI/ARB/WIF ; dès que l'un des deux tient un symbole, l'autre ne peut PLUS jamais trader ce symbole tant que la première position n'est pas fermée — réduisant la fréquence/largeur exactement là où le moteur multi-sleeves était censé l'augmenter. Aussi confirmé : les paires carry (S3) apparaissent dans `open_positions()` (`paper.py:170-186`) et bloquent également les deux autres skills sur le même symbole.
**Sévérité** : moyenne-basse pour la sûreté stricte (rien ne peut sur-trader ou pyramider — le bug est TROP conservateur, pas trop laxiste, donc ne menace pas le capital), mais élevée pour la fidélité au backtest : le live ne reproduit PAS le book validé (le comptage `edge_paper_trades:40` en sera ralenti sans qu'aucune alerte ne le signale). Coût de fermeture : un test d'intégration + un patch d'une ligne (`any(p.symbol == proposal.symbol and p.skill_id == proposal.skill_id for p in positions)` avec ajustement du `max_open_positions` en conséquence, car ce champ n'est plus un vrai plafond de risque agrégé sinon).


## 3. Garde-fous par jour

**daily_loss_kill 5%** (stage_0) — `src/hermes/risk/guardrails.py:142-150` : calcule le PnL réalisé depuis minuit UTC (`_realized_pnl_today_usd`, `guardrails.py:69-91`, requête SQL sur `JournalRow.closed_at >= midnight`) et rejette tout nouveau trade si la perte dépasse le seuil de la stage (`config/default.yaml:30,50,63,76` — 5%/2%/4%/5% selon stage).
**Test qui mord (preuve RED/GREEN réelle)** : `tests/test_risk_engine.py:44-61` (`test_daily_kill_switch_fires`) force `realized_pnl_today_usd=-6.0` sur equity=100 (stage_0 cap 5%) et vérifie `decision.decision == "REJECTED"` avec `"kill"` dans la raison — un test qui échouerait si le seuil était mal câblé. C'est un vrai test de mutation, pas juste une assertion triviale.
**stop-watcher 30s** — `src/hermes/scheduler.py:86-92` : `IntervalTrigger(seconds=30)`, `max_instances=1, coalesce=True` (anti-double-exécution). Le tick (`src/hermes/runtime/stop_watcher.py:30-80`) est **entièrement déterministe, zéro appel LLM** — confirmé par lecture complète du fichier — donc il continue de fonctionner même si DeepSeek ET le fallback sont morts en même temps.
**Time-exits par skill** — `config/default.yaml:127-130` (`max_hold_bars_by_skill`: momentum 12×4h≈48h, funding 24×4h≈96h, carry 48×4h≈8j), appliqué dans `guardrails.py:264-269` via `hermes.risk.timeexit`, actif SEULEMENT si `selective_mode` est vrai (l'est en prod).


## 4. Structurels

**selective_mode (le LLM ne peut pas forcer un trade)** : `src/hermes/agents/risk_manager.py:47-80`. Quand `selective_mode=true` (`config/default.yaml:101`, actif en prod), le signal déterministe (`drive_signal`, calculé à partir de `state.skill_signals` — indicateurs bruts, PAS le LLM) ÉCRASE la proposition du trader-LLM (`risk_manager.py:61-65`), puis re-passe par `selective_gate` (`selective.py:50-91`). Si aucun signal ne qualifie, `proposal.side` est forcé à `HOLD` (`risk_manager.py:78-80`) — le LLM ne peut ni initier ni bloquer un trade validé. Ceci est TESTÉ par mutation dans `tests/test_selective.py` (14 tests, dont `test_drive_signal_funding_not_starved_when_momentum_holds_symbol`).
**Corollaire vérifié par lecture du graphe** (`src/hermes/graph/orchestrator.py:24-55`) : `market_analyst.run` (calcule `skill_signals`, `src/hermes/agents/market_analyst.py:78-99`) est PUREMENT déterministe — aucun appel LLM. Mais `trader.run`, `risk_manager.run`, `portfolio_manager.run` (lignes 45-47) ne sont PAS enveloppés dans `_safe()` (contrairement à `market_analyst`/`news_analyst`/`sentiment`/`bull`/`bear`, lignes 32-34,41-44). `src/hermes/agents/trader.py:26-34` appelle `router.chat(...)` SANS try/except propre — si le LLM (DeepSeek + son fallback) est totalement injoignable, `router.chat` lève après 3 tentatives (`src/hermes/llm/router.py:181-217`), l'exception remonte hors de `trader.run`, et TOUT le tick avorte au niveau du scheduler (`src/hermes/scheduler.py:34-55`, try/except par symbole). **Réponse à la question posée : si le LLM est mort au tick, `risk_manager.run` — et donc le driver déterministe `selective_mode` — n'est JAMAIS atteint pour ce tick.** Résultat : fail-closed (zéro trade, y compris zéro trade de l'ancre S2 validée) plutôt que « le déterministe continue de trader sans le LLM » — plus sûr que l'inverse, mais contredit la lecture littérale du commentaire `risk_manager.py:44-46` (« le LLM devient advisory ») : en pratique, le LLM reste une DÉPENDANCE DE DISPONIBILITÉ (pas de jugement) pour que le déterministe s'exécute.
**TRM downgrade-only** : `risk_manager.py:82-136`. Le bloc TRM n'est même exécuté QUE si `side != "HOLD"` (ligne 83) — structurellement, TRM ne peut jamais voir/upgrader un HOLD. La seule mutation de `proposal.side` dans ce bloc est `if verdict.used_trm and refined.side=="HOLD" and confidence>0.7: proposal.side="HOLD"` (lignes 111-122) — aucune branche ne permet un flip BUY↔SELL ni un upgrade HOLD→trade. Confirmé par lecture exhaustive, mais **AUCUN test ne cible `risk_manager.run` ou cette logique précise** (`grep -rl "risk_manager" tests/` = vide) — garantie correcte mais non verrouillée par un test de mutation (R3).
**TRM réellement actif ?** `config/default.yaml:172` (`trm.enabled:true`), mais `src/hermes/trm/verifier.py:159-170` (`get_trm()`) retombe sur `NullTRMVerifier` (passthrough, `confidence=1.0`) si le checkpoint `/data/trm/latest.pt` est absent (`verifier.py:81-82`). Aucun `.pt` dans le dépôt, aucune trace documentaire d'un entraînement livré (`docs/DOSSIER_HERMES.md:1052` mentionne juste le mécanisme). **Vraisemblable que TRM tourne en NullVerifier en prod aujourd'hui** — le plafond de confiance et le downgrade-only sont alors des NO-OP réels (le code est correct mais inactif faute de modèle entraîné). Non vérifiable à distance sans accès au volume `/data` du VPS.
**S3 jamais nu (routage structurel)** : `src/hermes/execution/paper.py:267-271` — `place_order` intercepte `skill_id==CARRY_SKILL` et route SYSTÉMATIQUEMENT vers `_open_carry` (paire short-perp+long-spot, `paper.py:191-211`), quel que soit ce que l'agent a proposé — impossible d'émettre un ordre nu sous ce skill_id en paper. **MAIS** : `grep -n "CARRY_SKILL\|carry" src/hermes/execution/hyperliquid.py src/hermes/execution/bybit.py` = **aucun résultat**. Ce garde-fou structurel **N'EXISTE PAS** sur les venues live (Hyperliquid/Bybit) — s'il fallait avancer en stage_1+ avec `funding_carry_harvest` toujours dans `allowed_skills`, rien n'empêcherait aujourd'hui un ordre nu sur Hyperliquid. Gap réel à fermer AVANT toute promotion de stage avec S3 actif.
**Chat lecture seule** : `src/hermes/api.py:651-675` — `/chat` construit les messages et appelle `router.chat(agent_role="chat", messages=..., temperature=0.4, max_tokens=4096)` **sans paramètre `tools=`** — confirmé par lecture : aucun tool-calling n'est exposé au LLM du chat, donc il ne peut appeler AUCUNE fonction d'exécution (contrairement à `trader.py:32` qui passe `tools=TRADER_TOOLS`). Le endpoint est protégé par `x-api-key` (`api.py:111-115,651`). Le seul endpoint qui DÉCLENCHE une décision manuellement est `/tick/{symbol}` (`api.py:683-705`), séparé du chat, protégé par la même clé API.


## 5. Gouvernance anti-illusion

**Ledger durci** : `src/hermes/research/s10_governance.py:1042-1065` (`DeadEndsLedger._match`) — corrigé le 2026-07-03 (commit `ccd741d`, confirmé par `git log`) : le matching substring naïf (faux BLOCKED) est remplacé par nom-exact OU ≥2 tags OU un tag ultra-spécifique (≥4 mots), avec bornage sur limites de mots (`re.search(r"(?<!\w)...", ...)`). 38 entrées vérifiées (`governance/dead_ends_ledger.json`, `len(d)==38`), registre crypto à 1910 essais (`governance/trials_by_book.json:"crypto":1910`) — chiffres identiques à ceux du mandat.
**MAIS le ledger n'est PAS un gate exécutable** : `src/hermes/forward/prereg.py:151-165` — la docstring dit elle-même : « Informative-ONLY... deliberately NON-BLOCKING: it never raises, never skips a collector, never touches run_forward's control flow. It is a log line for governance review, nothing more. » Et `src/hermes/forward/run.py:257` : « S10's DeadEndsLedger.check() existed but was never called in prod (governance gap). » — même après le câblage (`run.py:262-265`), il ne fait qu'ajouter une clé au manifeste JSON imprimé ; rien dans le code ne refuse d'enregistrer un essai ou de router un skill mort. **C'est une garantie de discipline PROMPT/HUMAINE, pas un mécanisme (R3)** : rien n'empêche mécaniquement un agent (ou un humain pressé) de réintroduire un mort du ledger dans `config/default.yaml:allowed_skills`.
**Porte humaine sur `allowed_skills`** : aucune protection de branche possible (`gh api repos/.../branches/main/protection` → 403, « Upgrade to GitHub Pro or make repository public ») — vérifié en direct. Il existe des tests de verrouillage MOU : `tests/test_selective.py:181-188` (`test_default_config_includes_funding_track`, pin que `funding_squeeze_continuation` reste dans le slate), `tests/test_combination.py:122-135` (`test_combination_slate_matches_live_allowed_skills`, verrouille la parité slate/constantes), `tests/test_newdata_shadow.py:62-67` (assert que les skills shadow restent HORS du slate). Ce sont des trip-wires CI (un changement de `allowed_skills` fait échouer un test si non synchronisé), mais AUCUN n'empêche un seul commit de modifier le YAML ET le test en même temps — pas une vraie porte de revue humaine séparée.
**Déflation au vrai N, PBO, CPCV, DSR, pré-enregistrements** : `src/hermes/research/s10_jewel_gate.py` (280 lignes) implémente rigoureusement ces contrôles pour une future fusion de pillars — mais c'est un SPEC non exécuté sur le book réel (`jewel_gate.py:121-124,145-155` : refuse tant que <2 pillars confirmés, aucun aujourd'hui) — conforme à l'état déclaré, correctement gardé par du code (`pillars_ok = len(pillars)>=2 and set(series).issubset(pillars)`), testé sur fixtures synthétiques uniquement.


## 6. Infra

**Mono-conteneur post-incident** : `docker-compose.yml` confirmé — un seul service `hermes-agent` + `hermes-redis`, commentaire explicite : « NO Tailscale sidecar, NO Mac dependency... The old inter-container netns coupling... was the root cause of the 2026-06-25 outage ». Cohérent avec la mémoire.
**Fallback LLM DeepSeek→cloud, vérifié dans le CODE** : `src/hermes/llm/router.py:72-91` (`resolve_targets`) — si `DEEPSEEK_API_KEY` est présent, DeepSeek devient primary/secondary/escalation et les cibles `LLM_PRIMARY/SECONDARY` deviennent le FALLBACK automatique (`router.py:164-174`, capturé sur `_RETRY_EXCEPTIONS`). Testé par mutation : `tests/test_router_deepseek.py` (clé vide → pas de fallback ; clé présente → DeepSeek actif, local en fallback ; clé blanche → traité comme vide).
**MAIS le défaut de code pour ces cibles fallback est `http://hermes-mac:11434/v1`** (`src/hermes/settings.py:41,44` — un hostname Tailscale-only, précisément la dépendance retirée le 2026-06-25). Le commentaire du même fichier (`settings.py:50-53`) dit explicitement : « pointed in prod at OpenRouter... reached direct over the internet, no Tailscale » — c'est-à-dire que la config PRÉVUE en prod DOIT écraser ces défauts via les variables d'environnement Dokploy. `docker-compose.yml` (commentaires en tête) et `.env.example` (sections `LLM_PRIMARY_*`) documentent cette intention (« OpenRouter serving inclusionai/ring-2.6-1t »). **Ce qui n'est PAS vérifiable depuis ce dépôt** : les variables d'environnement réellement chargées par le conteneur Dokploy en prod vivent dans l'UI Dokploy (hors de ce repo, hors de portée en lecture seule) — je ne peux ni confirmer ni infirmer que `LLM_PRIMARY_API_KEY` contient une vraie clé OpenRouter valide en ce moment. Le `.env` local du dépôt (non déployé, 4 clés de collecteurs de recherche seulement — CoinGlass/Deribit/FRED, toutes vides) n'est PAS le fichier utilisé par Dokploy et ne permet aucune inférence sur l'état réel. C'est un GAP DE VÉRIFIABILITÉ honnête, pas une preuve d'échec — mais la mission demandait explicitement « vérifie ce qui est vérifiable », et la réponse honnête est : le mécanisme de code est correct et testé ; sa configuration prod ne l'est pas depuis ici.
**CI filet + Dokploy découplé — exemple daté concret trouvé** : `gh run list` (2026-07-10) montre le commit `ccd741d` (« fix(gouvernance): matching du DeadEndsLedger durci ») **ÉCHOUÉ** en CI à 16:31:11 UTC (échec rapide, 27s = lint `ruff` I001 tri d'imports, cohérent avec le message du commit suivant `a1585e5` : « style: tri des imports (I001) oublié »), corrigé 36 secondes plus tard par `a1585e5` (16:31:47, succès, 13m49s). `.github/workflows/deploy.yml:9-17` montre que `build-deploy` `needs: test` — DANS ce workflow, un test rouge bloque bien le build-push-image. Mais `deploy.yml:50-53` documente lui-même le vrai risque : « relying on Dokploy's own GitHub auto-deploy webhook to pull the fresh :latest image » si les secrets Dokploy ne sont pas configurés dans CE workflow — c'est-à-dire que Dokploy a SA PROPRE souscription au webhook GitHub, indépendante de ce pipeline de test (confirmé par la mémoire `hermes-ci-dokploy-decouple.md`). Sur la fenêtre `ccd741d`→`a1585e5` (~36s), si Dokploy a réagi à `ccd741d` avant la correction, un commit qui échouait le lint (donc non passé par la suite de tests de CI) aurait pu être construit et déployé en prod via ce chemin indépendant. Je n'ai pas d'accès aux logs Dokploy pour confirmer que cela s'est PRODUIT cette fois précise — mais le mécanisme qui le permettrait est confirmé présent et daté.


## 7. Gaps honnêtes

**a) Aucun blackout news/macro** — Sévérité: MOYENNE. Confirmé par grep exhaustif (`FOMC|CPI|blackout|news_blackout|macro_event|econ_calendar` sur `src/hermes/risk`, `src/hermes/agents`, `src/hermes/scheduler.py`, `src/hermes/graph`) : zéro résultat dans le chemin de décision live. Les seuls hits (`s7_cb_tone.py`, `cbspeak.py`, `collect_macro.py`) sont des SIGNAUX de recherche (ton des banques centrales comme feature), pas un filtre de pause. Quantification (données réelles, `/dashboard/data`, `recent_closed`, 2026-07-10) : les 8 trades clos du book vont du 2026-06-21 03:22 au 2026-07-02 13:25. Calendrier réel vérifié (Fed/BLS, recherche web) : FOMC 16-17 juin 2026 et CPI 10 juin 2026 précèdent la fenêtre des 8 clôtures ; le prochain FOMC est 28-29 juillet 2026 et le prochain CPI le 14 juillet 2026, après la dernière clôture connue. Avec seulement 8 trades clos et sans `opened_at` exposé publiquement (l'API ne donne que `closed_at`), une correspondance trade-par-trade exacte n'est PAS calculable depuis ce point d'observation — mais avec des tenues de 48-96h (`max_hold_bars_by_skill`) et un cycle de décision 4h sur 7 actifs, le fait structurel qui gouverne est : l'exposition croît mécaniquement avec le nombre de trades × la durée de tenue × la fréquence des événements, sans AUCUN filtre codé pour la couper — ce n'est pas un risque mesuré et accepté, c'est un risque non instrumenté. Coût de fermeture: faible (un calendrier statique FOMC/CPI + un flag qui force HOLD ±N heures autour, quelques dizaines de lignes) — recommandation: fermer avant toute mise en stage_1 réelle, car c'est exactement le mode de défaillance ($12k un jour) que le système existe pour éliminer.
**b) Venue unique (concentration)** — Sévérité: MOYENNE. `config/default.yaml:6` (`data_venue: binanceusdm`, données ET paper), mais `stages.stage_1_micro/2/3` (`config/default.yaml:44,57,70`) pointent TOUTES vers `venue: hyperliquid` (pas Binance, pas JustMarkets malgré le nom du mandat) — Bybit apparaît seulement comme `EXCHANGE_BACKUP` optionnel (`.env.example`). Aucun mécanisme de bascule automatique Binance-data→Bybit si Binance est indisponible pour les prix n'a été vérifié activement (hors scope de cette lecture, mais `src/hermes/execution/bybit.py` existe et suggère un adaptateur prêt, non branché par défaut). Recommandation : documenter clairement que le plan live est Hyperliquid (perps 1x puis 2x), pas Binance ni JustMarkets, pour éviter une divergence entre le mandat opérateur et la config réelle.
**c) Pas de circuit-breaker de dérive equity au-delà du daily kill** — Sévérité: MOYENNE-HAUTE. Confirmé par grep (`max_drawdown|circuit_breaker|drawdown_kill` sur `src/hermes/risk`, `src/hermes/agents`) : `max_drawdown_pct` n'existe QUE comme condition de PROMOTION de stage (`stage.py:64`, `config/default.yaml:41`), jamais comme frein de trading intra-stage. Le seul filet est `daily_loss_kill_pct` (reset chaque minuit UTC — une perte de 4.9%/jour pendant 10 jours consécutifs ne déclenche RIEN) et le cold-floor absolu à $5 (95% du capital initial déjà perdu). Coût de fermeture: faible-moyen (un compteur d'equity glissant sur N jours + un seuil, réutilisant `_realized_pnl_today_usd` généralisé à une fenêtre roulante). Recommandation: ajouter un kill-switch sur drawdown roulant (ex. 10-15% sur 30 jours) avant stage_1.
**d) S3 (`funding_carry_harvest`) jamais nu — uniquement prouvé en paper** — Sévérité: HAUTE si le slate reste inchangé en avançant de stage. Voir section 4 : le routage structurel `place_order` existe UNIQUEMENT dans `paper.py`. Aucune trace de logique équivalente dans `hyperliquid.py`/`bybit.py` (grep confirmé vide). Recommandation: bloquant AVANT toute promotion à stage_1 tant que S3 est dans `allowed_skills` — soit porter le routage carry sur l'adaptateur live, soit retirer `funding_carry_harvest` du slate à l'avancement.
**e) Dépendance LLM au moment du tick** — Sévérité: BASSE (le comportement est fail-closed, donc sûr) mais désaligné avec l'intention documentée. Voir section 4 : le fait que `trader.run` ne soit pas enveloppé dans `_safe()` signifie qu'une panne LLM totale (DeepSeek + fallback) supprime le tick ENTIER, y compris l'exécution du signal déterministe S2 validé, plutôt que de laisser le déterministe conduire sans le LLM comme le suggère le commentaire « LLM devient advisory ». Recommandation (non bloquante, amélioration de disponibilité): envelopper `trader.run` dans un `_safe`-équivalent qui, en cas d'échec LLM, pousse un HOLD par défaut à `state.trader_proposal` (comme le fait déjà `trader.py:41-50` pour un `tool_calls` vide) plutôt que de laisser l'exception se propager — cela laisserait `risk_manager.run` s'exécuter et le driver déterministe fonctionner même LLM mort.
**f) Fallback OpenRouter — configuration prod non vérifiable depuis ce dépôt** — voir section 6. Ni confirmé ni infirmé ; nécessite un accès Dokploy (hors du périmètre lecture-seule de cette mission) pour trancher.
**g) CI/Dokploy découplé — fenêtre de risque datée et concrète confirmée** (voir section 6, commit `ccd741d`/`a1585e5`, 2026-07-10) — pas un nouveau risque (déjà en mémoire), mais un exemple réel et récent qui prouve que la fenêtre n'est pas seulement théorique.


**Limites déclarées :**
- Le fallback LLM DeepSeek→OpenRouter est correct et testé au niveau du code (router.py, test_router_deepseek.py) mais sa configuration réelle en production (variables d'environnement Dokploy) n'a pas pu être vérifiée depuis ce dépôt en lecture seule — c'est un vrai angle mort, pas une conclusion.
- L'état réel du TRM en production (checkpoint entraîné présent ou NullTRMVerifier passthrough) n'est pas vérifiable sans accès au volume /data du VPS ; l'absence de tout .pt dans le dépôt et de toute trace de run d'entraînement documenté rend le passthrough l'hypothèse la plus probable, mais ce n'est pas confirmé.
- La quantification de l'exposition macro (item 7a) est bornée par les données publiques disponibles (8 trades clos, closed_at seulement, pas opened_at) — le calendrier FOMC/CPI 2026 a été vérifié par recherche web réelle (sources Fed/BLS), mais la correspondance exacte trade-par-trade n'est pas calculable avec les seules données exposées par /dashboard/data.
- Je n'ai pas eu accès aux logs de déploiement Dokploy pour confirmer si le commit rouge ccd741d (2026-07-10 16:31:11-16:31:47) a réellement été construit/déployé pendant sa fenêtre de 36 secondes de CI rouge — le mécanisme qui le permettrait est confirmé présent (docs du workflow + mémoire), mais l'occurrence réelle cette fois précise n'est pas prouvée.
- Le défaut no-pyramid cross-skill (guardrails.py:156-157) a été confirmé par exécution réelle d'un script de reproduction dans ce dépôt (research_toplsr/scratch_inventory/repro_pyramid.py) — c'est une preuve d'exécution, pas une lecture statique, conformément à la règle R2.


---

# PARTIE IV — LES VENUES (Binance actif, JustMarkets dormant, portes datées)

**Synthèse :** Le rail data/backtest est Binance USDT-M (data_venue: binanceusdm, config/default.yaml:8), mais AUCUN adapter d'exécution Binance n'existe dans src/hermes/execution/ (seuls paper.py, hyperliquid.py, bybit.py) : le compte réel Binance de l'opérateur ne peut aujourd'hui exécuter aucun ordre réel dans ce code, seulement fournir les prix miroir du paper book. Pire, stage_1_micro/2/3 pointent tous venue: hyperliquid (config/default.yaml:34,49,63) alors que l'opérateur possède Binance + JustMarkets, PAS de compte Hyperliquid documenté — incohérence de config confirmée à corriger. JustMarkets reste un compte dormant à bon droit : la thèse de revival FX/métaux a été FALSIFIÉE le 2026-06-24 sur toutes les familles testées (mémoire hermes-fx-metals-expansion.md, hermes-cta-carry.md), le programme commodities/FX retail a été explicitement FERMÉ ("do NOT stand up MT5"), et docs/RESTRUCTURATION_2026-07.md §5 documente une porte d'accès datée 2027-07-02 (aucun perp FX majeur liquide n'existe sur 4 venues vérifiées) — les métaux (XAU/XAG) sont déjà couverts par le rail Binance existant (perpétuels USDT-M, funding 4h), rendant JustMarkets/MT5 inutile même pour cet axe.


## 1. Binance — le rail actif (données + paper), pas encore un rail d'exécution réel

**Ce qui est câblé et vérifié :**
- `config/default.yaml:8` : `data_venue: binanceusdm` — le venue que l'opérateur possède réellement ET le venue sur lequel les stratégies ont été backtestées (klines Binance), selon le commentaire lignes 3-6.
- `config/default.yaml:12` : univers 7 perps USDT-M (`BTC/USDT:USDT ... WIF/USDT:USDT`).
- `src/hermes/settings.py:104-112` (`exchange_primary`) : le yaml `data_venue:` est autoritaire ; `EXCHANGE_PRIMARY` env n'est qu'un fallback si le yaml est absent.
- `src/hermes/settings.py:114-123` (`paper_price_venue`) : par défaut = `exchange_primary` (donc Binance) — le paper book est pricé au MÊME venue que celui qui alimente l'OHLCV.
- `src/hermes/execution/paper.py:56` : `PaperExchange` instancie `get_market(venue_for_prices)` — c'est le marché Binance qui alimente les prix de fill du paper book, pas un exchange réel.
- Funding multi-intervalle et perpétuels métaux (XAUUSDT/XAGUSDT, funding 4h) sont sur le MÊME rail Binance USDT-M, en collecte (docs/RESTRUCTURATION_2026-07.md:19-25) — pas dans `universe:` donc pas encore tradés live, statut collect-now avec couperet 2027-07-02 (docs/RESTRUCTURATION_2026-07.md:127-149).

**Ce que le compte réel Binance permettrait aux étages supérieurs (SI un adapter existait) :**
- USDT-M perp : short natif (nécessaire à mean_reversion_regime et à la jambe short-perp de funding_carry_harvest).
- Spot Binance disponible sur le même compte pour la jambe long-spot de la paire carry delta-neutre (Stratégie #3, docs/DOSSIER_HERMES.md:772-786 : _open_carry() short perp + long spot, frais maker 5bps×2 jambes = 10bps ouverture).
- Limite documentée dans la mission : pas de perps FX sur Binance — confirmé indépendamment par docs/RESTRUCTURATION_2026-07.md:161-165 : interrogation directe de 4 venues perp majeures (Bybit 710 symboles, OKX 409, Hyperliquid 231, dYdX v4 296) — aucun perp FX majeur liquide nulle part, le seul listing nominal (EUR-USD dYdX) affiche $0 volume/24h.

**LE TROU CRITIQUE — vérifié par lecture de code :** src/hermes/execution/ ne contient que 3 modules concrets : paper.py, hyperliquid.py, bybit.py. src/hermes/execution/factory.py:33-59 (get_exchange()) n'a que 3 branches — venue == 'paper', venue == 'hyperliquid', venue == 'bybit' — et raise RuntimeError(f'unknown venue: {venue}') (ligne 59) pour tout le reste, y compris 'binanceusdm' ou 'binance'. Il n'existe AUCUN src/hermes/execution/binance.py. Le seul code Binance sous src/hermes/ est src/hermes/datalake/sources/binance.py (+ binance_extended.py, binance_spot.py, binance_tail.py) — ce sont des COLLECTEURS DE DONNÉES (klines/funding/OI), zéro capacité de placement d'ordre. Conclusion factuelle : le compte Binance réel de l'opérateur ne peut aujourd'hui exécuter aucun ordre réel dans ce code — il ne sert qu'à la donnée de marché et au prix du paper book.


## 2. Hyperliquid / Bybit — adapters réels, testés uniquement en unitaire pur, et une incohérence de stage-ladder confirmée

**État des adapters (lecture complète des fichiers) :**
- src/hermes/execution/hyperliquid.py (237 lignes) : adapter réel via hyperliquid-python-sdk, refuse de démarrer sans credentials (HL_ACCOUNT_ADDRESS/HL_SECRET_KEY, lignes 72-76 — 'refusing live mode'). Implémente equity_usd, open_positions, last_price, place_order (avec cloid idempotent + TP/SL natifs attachés post-fill, lignes 200-210), close_position.
- src/hermes/execution/bybit.py (150 lignes) : adapter réel via CCXT, spot uniquement (options: {defaultType: spot} ligne 37), refuse les SELL sans détenir le sous-jacent (spot ne peut pas shorter, lignes 97-103), reconstruit les positions depuis le journal (jamais depuis le solde brut, pour ne pas s'approprier des coins pré-existants de l'utilisateur, lignes 3-9).

**Statut de test — vérifié par recherche exhaustive dans tests/ :** un seul fichier touche ce code, tests/test_hl_helpers.py (lu intégralement), et il ne teste QUE les deux fonctions pures documentées comme 'testable without the SDK / credentials / network' (derive_cloid, build_tpsl_orders — docstring hyperliquid.py:1-16 et 36). Aucun test n'exerce HyperliquidExchange.place_order/close_position/equity_usd ni quoi que ce soit de BybitExchange (zéro fichier de test pour bybit.py). Les chemins réels d'exécution (appels SDK/CCXT, gestion des erreurs _extract_filled, dédup cloid en conditions réelles) n'ont jamais été exercés en CI ni, selon l'état prod (stage: stage_0_paper, bankroll_usd: 100.0, confirmé via curl https://api.betterlifewithmk.com/dashboard/data), en production — le book n'a jamais quitté le paper.

**L'INCOHÉRENCE DE CONFIG (mission point 2) — confirmée par lecture de config/default.yaml :**
stage_1_micro: venue: hyperliquid (ligne 34)
stage_2_normal: venue: hyperliquid (ligne 49)
stage_3_perps: venue: hyperliquid (ligne 63)
Les 3 étages au-dessus du paper pointent TOUS Hyperliquid. factory.py supporte réellement Hyperliquid et Bybit comme venues d'exécution — mais l'opérateur (mandat de cette mission) possède Binance + JustMarkets, aucun compte Hyperliquid ni Bybit n'est mentionné dans le mandat ni dans les mémoires. Si le stage-ladder avance un jour (edge_paper_trades gate à 2/40, très loin), le code enverrait le premier argent réel vers un exchange où l'opérateur n'a démontrablement pas de compte fondé — c'est une incohérence de restructuration à trancher AVANT tout franchissement de stage, pas après. Deux options honnêtes, à décision opérateur : (a) ouvrir/financer un compte Hyperliquid réel avant stage_1 (ce que le code suppose implicitement), ou (b) écrire un vrai adapter src/hermes/execution/binance.py (ordre USDT-M perp + spot) et repointer stage_1_micro/2/3 sur venue: binanceusdm, ce qui aligne aussi l'exécution réelle sur le venue de backtest (actuellement backtesté sur Binance mais exécuté-réel sur Hyperliquid — deux marchés différents = un edge validé Binance n'est pas garanti identique sur Hyperliquid, funding/spread/liquidité diffèrent). Bybit est câblé comme exchange_backup (settings.py:67) mais n'apparaît dans AUCUN stage de config/default.yaml — code mort de facto au niveau stage-ladder aujourd'hui.


## 3. JustMarkets — le compte dormant, historique complet des verdicts et statut honnête

**Chronologie vérifiée (memory + docs, dates concordantes) :**
- 2026-06-24 — Feasibility GO (hermes-fx-metals-expansion.md, section 'ORIGINAL PLAN') : étude de faisabilité verte — coût 12-15× moins cher que crypto (FX/or RT 1.3-2.1bps vs crypto 20bps), architecture MT5 bridge via Tailscale vers hermes-mac, module src/hermes/execution/mt5.py (jamais créé — confirmé, aucun fichier mt5 dans le repo), venue justmarkets_mt5 (jamais ajouté à factory.py, confirmé par lecture du fichier).
- 2026-06-24, LE MÊME JOUR — thèse de revival FALSIFIÉE (hermes-fx-metals-expansion.md, section 'OUTCOME') : 'revalidation ran the full library on XAU/FX with honest per-asset FX cost... ZERO families survived screen→stress→adversarial-verify.' Cointégration (demi-vies 19.7-98.8 jours, pas d'heures), session/ToD (signal brut zéro-moyenne avant coût), mean-reversion (net médian négatif), breakout (passe le screen puis réfuté par le null MC random-entry 0/3), XSec/lead-lag (artefacts single-asset). Le seul 'bright spot' apparent (or TSMOM +6.5bps OOS) réfuté sur 3 angles indépendants : null MC p=0.358 (le hasard bat le signal), Sharpe -0.11 sur 26 ans avec perte -0.99 en 2011-2015, réplique sur données indépendantes (GLD/IAU/SGOL) mais perd le bear 2011-2019 — conclusion : 'conditional long-gold beta in a TSMOM costume'.
- 2026-06-24 — Programme FERMÉ (hermes-fx-metals-expansion.md, 1ère ligne) : 'PROGRAM CLOSED: commodities/FX has NO retail edge — redirect to crypto.' Carry FX G10 crash-protégé : rank-shuffle null p=0.073-0.276 selon protection, jamais <0.05 (hermes-cta-carry.md : 'Verdict: REFUTED. Deploy nothing; do NOT stand up a live carry sleeve'). Carry cross-sectionnel 33-marchés : p=0.804. Trend diversifié 33-marchés : perd contre le bull-beta séculier du panier (Sharpe 0.151 vs 0.431 drift, 0/112 configs). Roll-yield commodités (Gorton-Rouwenhorst) : signe FAUX sur données WTI ground-truth (IC -0.10 à -0.35 toutes époques). Recommandation explicite : 'close the commodities/FX program, do NOT spend on data, do NOT stand up MT5; reallocate to deepening/hardening the crypto book.'
- docs/RESTRUCTURATION_2026-07.md:177 reconduit l'interdiction : 'Ne PAS re-litiger, ne PAS monter d'infra MT5/JustMarkets pour ce volet' — au sujet des familles directionnelles et du carry FX retail, 'morts par la preuve'.

**À quoi ce compte PEUT servir un jour — conditions exactes et datées (docs/RESTRUCTURATION_2026-07.md:161-184, section 5) :**
- Porte A — ré-audit d'accès, 2027-07-02 : re-interroger les 4 venues perp (Bybit/OKX/Hyperliquid/dYdX) pour un perp FX majeur avec volume 24h ≥ $100M ET ≥ 12 mois de funding continu. Si absent → fermé 12 mois de plus (2028-07-02) sans anticipation motivée par l'envie d'activité.
- Porte B — si un perp qualifiant apparaît : règle S2 GELÉE (z sur 180 barres, seuil z≥2, hold 24 barres, non re-tunable), pré-enregistrement avant tout calcul, ≥30 événements dé-chevauchés + couverture d'un régime bear obligatoire.
- Porte C (secondaire) — stablecoins EUR, 2027-01-02 : un stablecoin EUR dépasse-t-il $100M vol/24h ? Sinon pas de re-test avant 2027-07-02.
- Distinction actée dans le doc (§5) : les familles directionnelles/carry FX retail sont mortes PAR LA PREUVE (ne reviennent jamais sans donnée propriétaire neuve, la porte ne les concerne pas) ; l'axe funding/microstructure — le seul axe vivant d'Hermès — n'a JAMAIS pu être testé sur FX car l'instrument n'existe pas sur rails accessibles : 'ce n'est pas un verdict négatif sur l'axe, c'est une impossibilité d'accès. La porte reste une porte, pas un mur.'
- Les métaux (le second usage envisagé pour JustMarkets) sont déjà couverts SANS JustMarkets : XAUUSDT/XAGUSDT tradent sur le rail Binance USDT-M existant à coût zéro (docs/RESTRUCTURATION_2026-07.md:19-25), rendant un pont MT5 inutile même si l'axe métaux clarifiait plus tard.

**Pourquoi l'activer aujourd'hui serait contraire à l'évidence :** aucune des conditions datées (2027-01, 2027-07-02) n'est atteinte ; le mandat lui-même ('program CLOSED', 'do NOT stand up MT5') a été réaffirmé deux fois à 8 jours d'écart (2026-06-24 puis 2026-07-02) par des sessions de recherche indépendantes ; aucun élément neuf (donnée propriétaire, perp FX qualifiant) n'est apparu depuis. Activer JustMarkets aujourd'hui reviendrait à rouvrir un verdict falsifié sans nouvelle preuve.


## 4. Matrice finale venue x étage x instrument, et restructurations candidates

| Étage (config/default.yaml) | Venue configuré | Instruments | Compte opérateur réel disponible ? | Verdict |
|---|---|---|---|---|
| stage_0_paper | paper (prix miroir = paper_price_venue → Binance) | perp, univers 7 cryptos | N/A (papier) | OK — état live actuel (stage_0_paper, equity $103.74, confirmé dashboard/data) |
| stage_1_micro | hyperliquid (ligne 34) | perp, 1x | NON — aucun compte Hyperliquid dans le mandat opérateur ; adapter réel existe mais jamais exercé en tests au-delà de 2 helpers purs | INCOHÉRENT — restructuration candidate |
| stage_2_normal | hyperliquid (ligne 49) | perp, 1x | NON (même compte manquant) | INCOHÉRENT — restructuration candidate |
| stage_3_perps | hyperliquid (ligne 63) | perp, jusqu'à 2x | NON (même compte manquant) | INCOHÉRENT — restructuration candidate |
| (aucun stage) | bybit | spot uniquement, no-short natif | Compte non confirmé dans le mandat | Code mort au niveau stage-ladder — exchange_backup dans settings.py:67 mais jamais référencé dans stages: |
| (aucun stage — n'existe pas) | binanceusdm en exécution réelle | USDT-M perp + spot (carry) | OUI — c'est le compte réel de l'opérateur, ET le venue de backtest | Adapter manquant : src/hermes/execution/binance.py n'existe pas — le rail le plus cohérent (données=backtest=compte réel) n'a aucune capacité d'exécution réelle |
| Métaux (XAU/XAG) | Binance USDT-M (déjà le même rail crypto) | perp | OUI (même compte) | Collect-now, couperet 2027-07-02 ; JustMarkets superflu pour cet axe |
| FX majeurs | Aucun (docs/RESTRUCTURATION_2026-07.md:161-165 : aucun perp FX liquide sur 4 venues vérifiées) | — | JustMarkets (MT5, dormant) | Porte datée 2027-07-02, PAS de réouverture aujourd'hui |

**Décisions/restructurations candidates à trancher par l'opérateur (aucune n'est exécutée ici — lecture seule) :**
1. Aligner stage_1_micro/stage_2_normal/stage_3_perps sur un venue réellement possédé. Deux voies exclusives : (a) écrire src/hermes/execution/binance.py (implémentant le protocole Exchange de base.py, USDT-M perp + spot pour la jambe carry) et repointer les 3 stages sur venue: binanceusdm — bénéfice additionnel : élimine l'écart backtest-Binance / exécution-Hyperliquid ; (b) documenter explicitement que Hyperliquid est un compte à ouvrir avant tout franchissement de stage réel, avec une date/condition, symétrique aux portes datées déjà utilisées pour JustMarkets/FX/métaux — actuellement AUCUNE des deux n'est faite, c'est un vide de config, pas un choix motivé.
2. Bybit : soit le retirer de factory.py/settings.py comme mort (mandat n'en parle pas), soit documenter son rôle réel (exchange_backup — backup de QUOI exactement, sur quel stage ?).
3. JustMarkets : ne rien faire avant 2027-01-02 (porte C, stablecoins EUR) / 2027-07-02 (porte A, ré-audit perp FX) — statu quo correct, déjà verrouillé par deux verdicts indépendants et un doc de restructuration qui interdit explicitement de relancer MT5.
4. Le gate edge_paper_trades: 40 (config/default.yaml:27-31) est actuellement à 2/40 — la question du venue de stage_1 n'est pas urgente opérationnellement, mais c'est précisément le bon moment pour la trancher (avant que le gate ne s'ouvre), pas après.


**Limites déclarées :**
- Aucun commit ni modification n'a été effectué (lecture seule src/config/docs/governance, conforme au mandat de sécurité).
- Les tests hyperliquid.py ne couvrent que 2 fonctions pures (derive_cloid, build_tpsl_orders) — l'absence de test sur place_order/close_position/equity_usd réels et sur la totalité de bybit.py est un fait de code, pas une évaluation de qualité du SDK tiers lui-même.
- docs/DOSSIER_HERMES.md (section ~3.2.5) liste encore ts_mom_vol_scaled dans allowed_skills — ce document est antérieur au retrait de S1 (2026-07-02) ; le slate actuel confirmé dans config/default.yaml:110 est [mean_reversion_regime, funding_squeeze_continuation, funding_carry_harvest]. Ne pas se fier à DOSSIER_HERMES.md pour l'état du slate, seulement pour l'architecture des adapters.
- L'absence de compte Hyperliquid/Bybit détenu par l'opérateur repose sur l'absence de preuve documentaire dans le repo/mémoires/mandat, pas sur une vérification externe des comptes réels de l'opérateur — mais Binance et JustMarkets SONT explicitement nommés dans le mandat de cette mission, ce qui rend l'asymétrie factuelle.
- L'endpoint public dashboard/data a été interrogé en direct (stage_0_paper, equity $103.74) pour corroborer que le book n'a jamais quitté le paper — cohérent avec le gate edge_paper_trades à 2/40 mentionné dans le contexte opérateur.

---

## Addendum — 2026-07-10 soir : gaps (a) et (c) FERMÉS, mécanismes prouvés

Les deux gaps de sûreté « avant stage_1 » sont fermés (TDD, 28 nouveaux tests) :
- **Blackout macro** (`risk/macro_blackout.py` + `config/macro_calendar.yaml`) : 40 événements
  réels 2026-2027 (16 FOMC, 12 CPI, 12 NFP) aux sources officielles Fed/BLS contre-vérifiées
  (y compris deux anomalies BLS réelles : NFP mercredi 2026-02-11 et jeudi 2026-07-02),
  fuseaux IANA + tzdata (DST-safe), fenêtre T−2h→T+2h configurable, warning si horizon < 60 j
  (horizon actuel : 2026-12-04, BLS n'a pas encore publié 2027).
- **Circuit-breaker de dérive** (`rolling_loss_kill`) : PnL réalisé glissant 30 j ≤ −10 %
  de l'equity → kill, preuve par mutation RED/GREEN.
- **Décision de fidélité** : les stratégies ont été validées AVEC les périodes de news —
  les deux gates tournent donc en SHADOW au stage_0_paper (verdict calculé, loggé, exposé
  au dashboard bloc `protections` — jamais bloquant) et n'APPLIQUENT qu'aux stages réels
  1/2/3. On mesure leur impact sur l'évidence avant d'y soumettre du capital.
- **Statut TRM exposé** (`trm_status` au dashboard) : NullVerifier vs actif — la décision
  « entraîner ou retirer » revient à l'opérateur, désormais les yeux ouverts.

**Gaps pré-stage_1 restants : (d) jambe spot carry réelle (S3), et la décision TRM.**
