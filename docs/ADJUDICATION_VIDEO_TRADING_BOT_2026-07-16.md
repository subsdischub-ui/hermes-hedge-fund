# Adjudication — vidéo « build a trading bot that actually works » + PDF *Miles High Club*
**2026-07-16** · Mandat opérateur : « continue la recherche pour notre rentabilité avec cette
vidéo et toutes les recherches pertinentes associées ».
Méthode : 4 agents (méthode-vs-ledger, venue/tooling, angle-neuf) + **audit adversarial**
(24 vérifications sur code/ledger réels + 1 ré-exécution live indépendante). Registre au moment
du verdict : **1 969 essais** (crypto 1 911 · fx_metals 53 · cross 5) · ledger : **41 entrées**.

> **Ce document existe pour qu'on ne re-dépense JAMAIS ce budget sur cette source.** Le verdict
> est : rien à prendre côté méthode, un seul angle neuf révélé *par contraste*, et deux conseils
> qui détruiraient l'ancre du livre s'ils étaient câblés.

---

## 1. Verdict global : manuel de RÉGRESSION

La vidéo (thèse : « l'humain apporte l'edge, le bot apporte l'exécution sans émotion ») et son
PDF de 6 prompts (bot paper TypeScript/Node, MA-crossover 9/21 sur BTCUSDT 5 m, mémoire
ledger.csv + learnings.md) sont un **excellent onboarding débutant**. Ses règles d'honnêteté sont
du bon instinct et méritent d'être citées : *« ne fabrique pas de fausses bougies, ne truque pas
de trades perdants, rapporte les vrais résultats quels qu'ils soient ; si une commande ne peut
pas tourner, imprime le blocage — ne simule pas le résultat »*. C'est le même **esprit** que la
machine d'honnêteté d'Hermès — sans aucune de ses **machines**.

| Ce que la vidéo/PDF prescrit | Ce qu'Hermès a déjà (fichier) |
|---|---|
| `ledger.csv` (fichier plat) | TradeJournal SQL itemisé — skill, confiance, thèse, état de marché (`memory/journal.py`) |
| `learnings.md` (leçons en anglais) | LessonsStore vectoriel Chroma + retrieval sémantique + réflexion nocturne + **calibration isotonique** (garde mathématique, `risk/calibration.py`) |
| `adaptiveFilter.ts` (skip si perdu avant) | `selective_gate` déterministe validé + TRM downgrade-only + meta-labeler shadow |
| `risk.ts` (qty max, SKIP si dépassement) | risk cap 0.5 %, Kelly fractionnaire 0.20, ATR sizing, no-pyramid (symbole,skill), haircut corrélation, kill quotidien 5 %, **blackout macro**, **circuit-breaker 30 j** (`risk/guardrails.py`) |
| paper only puis « risque 1-3 % » | échelle 4 étages, `HERMES_STAGE` manuel-only, gate 40 événements du slate validé, plancher notional par venue |
| backtest TradingView « et si le RSI… ? » | **1 969 essais comptés**, prereg avant calcul, MC random-entry null 5 seeds, joint-null date-level, OOS-by-date, déflation DSR/PBO, CPCV, **41 morts au ledger** |
| MCP : le LLM dans le chemin d'ordre | CCXT direct ; LLM **advisory** — la règle déterministe l'écrase (`agents/risk_manager.py`) ; refus structurels (pas d'ordre sans stop, carry → `NotImplementedError`) |

**Venue** : Alpaca (l'étape 1 de la vidéo) n'a **ni perpetual ni funding rate** — l'unique edge
validé du livre (S2, joint-null p=0.0005, axé sur le funding des perps) n'y a **mécaniquement
aucun toit**. Un 3ᵉ compte y serait un venue sans edge à y déployer.

**Une affirmation de la vidéo est directement réfutée par l'histoire d'Hermès** : « une fois les
règles dures fixées, l'AI s'y tiendra ». Le chat d'Hermès a inventé **4 trades sur 6** (faux
symboles, faux PnL) faute de contexte itemisé (`tests/test_chat.py`) — c'est précisément cet
incident qui a motivé `selective_mode`, le mécanisme qui **empêche** le LLM de décider seul.

---

## 2. Les deux conseils DANGEREUX, chiffrés

### 2.1 « Skippe un setup où tu as déjà perdu » — détruirait S2 par construction

C'est le cœur du PDF (Prompt 05) et la démo de la vidéo : signal → perte → même signal → perte →
*« c'est parce qu'il n'a pas de mémoire »*.

**Le mécanisme causal, chiffré sur l'ancre réelle** (`_s2_funding_squeeze.json`) :

| | Sharpe/trade | P(trade perdant) | P(2 pertes d'affilée) |
|---|---|---|---|
| panier S2 | 0.427 | **33.5 %** | 11.2 % |
| BTC | 0.324 | **37.3 %** | 13.9 % |
| ETH | 0.372 | **35.5 %** | 12.6 % |
| SOL | 0.261 | **39.7 %** | 15.8 % |

*(approximation normale ; les vraies queues sont plus grosses → pire.)*

Sur les **115 événements** du panier validé : **~38 trades perdants attendus** et **~13
occurrences de deux pertes consécutives** — dans un edge qui **fonctionne** (joint-null p=0.0005).
La démo de la vidéo décrit donc un événement à **11 % de probabilité** : la variance ordinaire
d'un edge sain, pas une leçon.

**Conséquence directe** : sous la règle du PDF, la première fois que S2 perd sur BTC (≈1 fois sur
3), le bot **arrêterait définitivement** de trader S2 sur BTC. Sa « mémoire » tuerait le seul edge
du livre. Ce n'est pas de l'apprentissage — c'est le **sophisme du joueur codé en TypeScript**.
L'edge vit dans l'agrégat, jamais dans le trade individuel.

**Statut épistémique (à ne pas maquiller)** : cette idée n'a **pas** été testée telle quelle chez
Hermès — elle est **réfutée par mécanisme**, ce qui est plus fort qu'un test. Son cousin
statistiquement mieux motivé *a* été testé et est mort : ledger **#32** « S8 regime conditioning of
funding_squeeze (HMM/BOCPD) » — shuffle-p≈0.11, CPCV régime-aware perm-p=0.135 (bruit).
Conditionner sur l'**état de régime** (causal, exogène) est strictement plus défendable que
conditionner sur le **résultat passé** (endogène) ; la version défendable a échoué. *A fortiori*.

### 2.2 Le backtest TradingView sans déflation = machine à faux positifs

La vidéo recommande d'itérer « et si le RSI touchait ce niveau… et si le prix spike de x %… » et
de retenir ce qui « a l'air fiable ». Aucun des 4 garde-fous d'Hermès n'y est : MC random-entry
null 5 seeds, déflation sur grille (DSR/PBO), OOS-by-date, joint-null panier synchronisé.

**Le chiffre qui tranche** : Hermès compte **1 969 essais**. La déflation punit exactement ce
comportement — le même track record voit son DSR s'effondrer à mesure que le nombre d'essais
grimpe. Exemple de faux positif que la méthode de la vidéo aurait retenu : ledger « D1 ML
feature-zoo » — OOS acc 0.518-0.543 ≤ 0.556 (base rate), mais **combined MC-null p=1.000**,
invisible sans le null. Appliquée aux données d'Hermès, la méthode de la vidéo reproduirait une
fraction des 41 morts du ledger **déguisés en découvertes**.

### 2.3 Bonus : la stratégie-exemple est morte deux fois

Croisement de moyennes 9/21 sur BTCUSDT **5 minutes** — deux morts indépendantes :
- **cadence rapide** : ~11 déclinaisons de familles OHLCV single-asset meurent net de coût taker
  honnête (15-20 bps RT) = whipsaw (ledger #15) ;
- **famille momentum** : `ts_mom_vol_scaled` retiré du slate (LOO-Δ Sharpe **−0.146**, joint-null
  p=0.507) et « Gold trend-following (TSMOM) » au ledger (battu par le MC null, net-négatif
  sur 26 ans). Morte sur deux marchés distincts.

---

## 3. Le seul angle neuf : encoder les setups DISCRÉTIONNAIRES de l'opérateur

### 3.1 Le trou est RÉEL — vérifié, pas allégué

Grep exhaustif du repo (code + docs + gouvernance) : **ZÉRO** fichier de skill cite une provenance
« expérience opérateur ». Toute la bibliothèque vient de la littérature — `mean_reversion_regime`
cite Connors/Bollinger, `ts_mom` cite la littérature TSMOM, `signals.py` cite Lo-MacKinlay. Sur
**1 969 essais** tracés, **aucun** n'est nommé d'après l'opérateur.

**La thèse de la vidéo est l'inverse de celle d'Hermès** — et c'est ce contraste qui révèle
l'angle. Les $12 k perdus en un jour ne venaient pas d'un mauvais signal : c'était une **faille
d'exécution sous émotion**. Or `risk/macro_blackout.py` dit littéralement : *« the exact failure
mode that cost the operator $12k on a manual account »*. **La machine d'exécution sans émotion que
la vidéo réclame existe déjà, et elle a été construite en nommant cet incident.** Ce qui n'a
jamais existé, c'est l'autre moitié : **l'edge du trader**.

### 3.2 Verdict adversarial : l'angle est réel mais MINCE

Les familles qu'un trader discrétionnaire FX/matières premières utiliserait sont **déjà toutes
mortes**, avec preuve exécutée :

| Famille discrétionnaire | Statut | Preuve |
|---|---|---|
| Breakout de canal (Donchian) | **MORT** ×2 | crypto #15 ; **FX ré-exécuté le 2026-07-16** : MC-null 0/3, death cost min 4.87 bps vs stress 3.20-4.20 bps → *SURVIVOR REFUTED* (ledger #40) |
| Session London/NY | **MORT** | ledger #17 — zero-mean pre-cost |
| Cointégration / paires | **MORT** | ledger #16 — demi-vies 20-99 jours |
| Momentum cross-sectionnel / lead-lag | **MORT** | ledger #18 — artefacts single-asset ou queue mince |
| **Niveaux S/R FIXES (swing highs/lows, ronds, pivots, retests)** | **JAMAIS TESTÉ** | grep exhaustif : **0 occurrence** dans tout le repo |

**Différence structurelle qui justifie l'exception** : tout ce qui est mort utilise une fenêtre
**glissante** (Donchian, cointégration). Un niveau S/R discrétionnaire est un prix **FIXE qui
persiste** dans le temps et se **re-teste** — mécanisme jamais implémenté ici. C'est une famille
réelle, pas un territoire : **une seule survit à l'élimination**.

### 3.3 La réserve que l'audit a posée, et que je transmets telle quelle

> **$65 → $18 500 = petit compte + levier + résultat extrême.** C'est la signature statistique
> classique de la **variance**, pas nécessairement de la compétence (un seul tirage, pas de
> distribution). Aucun protocole statistique, si rigoureux soit-il, ne peut réparer ça *après
> coup* : il testerait une **règle reconstruite de mémoire**, c'est-à-dire une **histoire**, pas
> l'edge de l'époque.

D'où la question **préalable** à tout calcul, et dont la réponse change le statut du test :
**ces règles étaient-elles écrites à l'époque, ou sont-elles reconstruites aujourd'hui ?**

### 3.4 Protocole exact (à n'engager qu'après la description opérateur)

1. **PREREG écrit AVANT tout calcul** : la définition du niveau (swing high/low sur fenêtre W
   confirmée par K barres ; tolérance de retest en % ou ATR), le sens (rejet vs cassure), le
   stop, la sortie, l'univers exact (klines 4 h/1 j déjà au lake), le seuil de succès. Registre
   bumpé **via `BookTrialRegistry.bump()`** (jamais à la main — défaut du 2026-07-16), source
   `operator_discretionary_sr_prereg`, **avant** de lire quoi que ce soit.
2. **Implémentation causale** `fn(df) -> SkillSignal` (pattern `research/harness.py`), zéro
   paramètre optimisé post-hoc.
3. **MC null au MÊME mécanisme** — pas un simple random-timing : le null doit lui aussi chercher
   des niveaux **aléatoires** avec le même nombre de retests (leçon `refute_tier2_breakout.py` :
   sign-flip ET random-entry block-bootstrap).
4. **OOS-by-date** (split chronologique strict).
5. **Déflation au N réel** (1 969 + 1) via `cpcv_sharpe_distribution`, comparée au grid-max.
6. **Bear-check explicite** — au moins un régime baissier complet couvert.
7. **Orthogonalité vs S2** : |ρ| PnL quotidien ≤ 0.30 (gate déjà codé, `harness.py`).
8. **Si l'opérateur raffine sa règle après avoir vu un résultat : c'est un NOUVEL essai**, compté.
   C'est ce qui contient le p-hacking d'un espace conjonctif (« breakout + daily en tendance +
   après un sweep raté + 2 premières heures de NY » : assez de conditions et on fitte n'importe
   quoi).

**Ce que ce test ne prétendra pas** : même un PASS complet ne rend pas l'entreprise rentable — il
ajouterait une famille au livre. La variable dominante du revenu reste le **capital**.

---

## 4. Décisions

- **Rien à prendre de la vidéo/PDF côté méthode, venue ou architecture.** Aucune modification du
  code n'en découle. Ne pas re-adjuger cette source.
- **Aucune entrée ledger créée pour la vidéo** : ses claims mappent sur des entrées existantes
  (#32 pour le conditionnement, #15/#24 + gold TSMOM pour le MA-crossover). Le « skip du setup
  perdant » est **réfuté par mécanisme** (§2.1) et non « testé et mort » — l'écrire au ledger
  serait mentir sur son statut épistémique. Ce document porte le mécanisme ; #32 porte le test.
- **Angle discrétionnaire : OUVERT, en attente de l'opérateur.** Sans description écrite et
  précise de ses règles, le trou reste théorique et **non actionnable** — on sait qu'il n'a
  jamais été testé, pas encore **quoi** tester.

## 5. Trous de gouvernance relevés par l'audit — tous FERMÉS le jour même

| Trou | Fermeture |
|---|---|
| Verdicts perps actions/tokenisées en prose seulement | ledger #39 (commit `127ec67`) |
| Verdict FX breakout en prose seulement | ledger #40 (commit `127ec67`) |
| Registre bumpé à la main : `counts` sans provenance | réparé + invariant testé (`925fee1`) |
| `governance/` **jamais embarqué dans l'image Docker** → registre re-semé à 1892 à chaque boot | `COPY` + test de garde (`a482cbd`) — **prod vérifiée à 1911** |

---

*Sources : workflow `w2hr290in` (4 agents, 418 k tokens, 128 appels d'outils) ; chiffres S2
re-calculés localement ; grep de provenance et grep S/R re-vérifiés par l'orchestrateur ;
`research_fx/refute_tier2_breakout.py` ré-exécuté indépendamment (sortie bit-à-bit identique).*
