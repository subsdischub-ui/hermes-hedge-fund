# STREAM 10 v-next — GOUVERNANCE (déflation durcie + registres partagés) — FINDINGS

**Branche `round2/s10-gov`. Hub de gouvernance — aucune stratégie créée, rien en live, gate du
joyau NON lancé (pas de pilier).** Le mandat : (1) durcir les règles méthodologiques en code testé,
(2) rendre les registres PARTAGÉS (trackés, pas gitignorés), mis à jour avec la réalité post-S0/S2,
(3) ajouter l'OOS-forward comme précondition dure de promotion, (4) spécifier (sans le lancer) le
gate du joyau, (5) ajouter des garde-fous de provenance.

Étiquettes de provenance : **[VÉRIFIÉ]** = confirmé contre la FINDINGS source ; **[PROBABLE]** =
inféré d'une source fiable ; **[PRÉSUMÉ]** = interprétation/jugement, non cité littéralement.

**Suite : 80 tests S10 v-next verts** (`test_s10_governance.py` 42, `test_s10_rules.py` 19,
`test_s10_provenance.py` 10, `test_s10_jewel_gate.py` 9), ruff clean. Repo : 656 passants ;
les 2 seuls échecs sont environnementaux et **PRÉ-EXISTANTS, sans rapport avec S10** (`test_s6_*`
importent `fxgate` du worktree `research_fx/` gitignoré ; `test_datalake_gold` exige `openpyxl` non
installé) — aucun fichier que j'ai touché n'est en cause.

---

## 0. ÉTAT DU MERGE — DIFFÉRÉ (aucun pilier market-neutral livré)

S2 v-next a rendu un **négatif honnête sur toute la famille cross-sectionnelle** (2A/2B/2D/2.3) et
`funding_squeeze` reste **faiblement-supporté / non-confirmé**. **Il n'existe donc AUCUN sleeve
candidat à la promotion aujourd'hui.** Le gate du joyau (Task 5) est une **spec en code**, testée
sur fixtures synthétiques uniquement, prête pour le jour où un pilier survivra. La porte de merge
reste humaine (Malachie).

---

## 1. TASK 1 — REGISTRES PARTAGÉS : migration `research_streams/S10/` → `governance/` (TRACKÉ)

**Le problème (réel).** Les registres Round 1 vivaient en `research_streams/S10/*.json` — or
`/research_*/` est **gitignoré**, donc chaque worktree portait SA copie divergente du compte de
déflation. Un compte gitignoré, per-worktree, ne peut pas être l'état autoritaire du labo.

**La correction.** Nouvel emplacement **`governance/` à la racine, TRACKÉ** (committé sur
`round2/s10-gov` → identique partout via git). Code :
- `governance_registry_dir()` → `<repo>/governance` (override `HERMES_GOVERNANCE_DIR` pour tests).
- `BookTrialRegistry()` (no-arg) pointe désormais sur ce dir (et non `settings.data_dir`,
  per-worktree). [VÉRIFIÉ : aucun code live n'importe `BookTrialRegistry` — `grep` propre.]
- `governance/README.md` documente le contrat : toute mise à jour future passe par
  `BookTrialRegistry().bump()` + commit ; les snapshots `dead_ends`/`acquisition` sont régénérés
  depuis les constantes embarquées (code-as-truth), avec un **test de sync anti-drift**.

`governance/` contient : `trials_by_book.json` (crypto **1891** / fx 53 / cross 5),
`dead_ends_ledger.json` (36 entrées), `acquisition_map.json` (13 sources), `README.md`.

---

## 2. TASK 2 — RÈGLES DURCIES EN CODE TESTÉ (`src/hermes/research/s10_rules.py`, 16 tests)

Cinq leçons adversariales que les streams ont payées, transformées en helpers falsifiables que les
autres streams importent :

| Règle | Helper | Leçon (stream) | Test-clé qui l'encode |
|------|--------|----------------|----------------------|
| **R1** Bannir l'IC-en-niveau | `robust_ic` / `ic_significance` / `effective_sample_size` | S6 : IC-en-niveau anti-conservateur (~16% FPR à N=60) | Monte-Carlo 500× : FPR niveau >0.12 vs FPR delta ≤0.10 ([VÉRIFIÉ S6]) ; `mode="level"` lève une erreur |
| **R2** Chance-correction obligatoire | `chance_corrected_agreement` (κ) / `state_stability_excess` / `excess_over_chance` | S8 : 0.88 brut = chance (un HMM collapsé score 1.0) | prédicteur constant → κ=0 ; série mono-état → excess=0 |
| **R3** Orthogonalité sur semaines co-actives | `coactive_correlation` (+ participation ratio) | S5 : juger sur co-actives, pas la grille zéro-remplie | sleeves corrélés sur 30 semaines co-actives mais grille pleine `corr≈0` → co-active `>0.9` |
| **R4** Injection-de-fuite (driver + survivant) | `future_perturbation_leak_test` / `assert_no_lookahead` / `replay_leak_audits` | AM-4 : les bugs de composition fabriquent des positifs | builder qui lit `data[t+1]` → flag ; survivant sans replay → bloqué au merge |
| **R5** Nulls cluster-aware | `block_events_by_time` / `effective_independent_events` / `cluster_block_bootstrap_pvalue` | S2 : les squeezes co-firent sur les alts → null i.i.d. anti-conservateur | données clusterisées : `p_iid<0.05` (sur-rejette) vs `p_cluster>0.05` (honnête) |

`★` Le test R1 calcule LITTÉRALEMENT le FPR du faux-IC-niveau (à la main, puisque le helper le
refuse) pour DÉMONTRER pourquoi il est banni — c'est une garde falsifiable, pas un slogan.

---

## 3. TASK 3 — REGISTRES À JOUR (faits VÉRIFIÉS contre les FINDINGS sources)

### 3.1 DeadEndsLedger += 4 négatifs S2 (32 → 36 entrées) [VÉRIFIÉ `../hermes-s2-xsec/.../S2/FINDINGS.md` + `results_s2_xsec.json`]

| Sleeve | Mécanisme de mort | Kill-metric (vérifié) |
|--------|-------------------|-----------------------|
| **2A** funding-XS continuation | **Signe wrong-way** : le funding cross-sectionnel **MEAN-REVERTS**, il ne continue pas ; ET cost-killed (rotation ~100%/j) | gross ann Sharpe **−1.38** ; coût **−16.85 bps/j** ; MC-sign p=1.0 |
| **2D** vol-dispersion XS | Pas d'edge : le **gross price-spread** montre un whiff +0.20 mais NS, mangé par le coût | gross price-spread +0.20 Sharpe (NS, MC-p **0.311**) ; net −1.84 bps |
| **2B** OI-growth XS | Data-gated / under-powered : 7 coins, survivorship | 7 coins (ARB/AVAX/BTC/ETH/SOL/SUI/WIF) ; gross +0.13 NS ; coût −26 bps/j |
| **2.3** cross-venue dispersion | Pas d'edge, 5 coins, thin | ann Sharpe **−0.44** ; MC-p 0.05 ; 5 coins × 4 venues |

### 3.2 BookTrialRegistry : crypto 1869 → **1891** (+22)

[VÉRIFIÉ] 22 = **4 pré-enregistrés (2A/2B/2D/2.3) + 18 exploratoires**. **Honnêteté importante :**
le chiffre **1891 est MON arithmétique** (1869 + 22) — **S2 ne l'écrit JAMAIS** (il laisse la somme
non-calculée « 1869 + 4 + 18 » ; `grep 1891` dans S2 = 0 hit). Le commentaire de provenance dans
`DEFAULT_TRIAL_SEED` le dit explicitement. C'est le nouveau **compte de déflation crypto**.

### 3.3 AcquisitionMap — corrections [VÉRIFIÉ `../hermes-hedge-fund/.../S0/COVERAGE.md`]

- **CoinGlass multi-venue = PROFOND/INGÉRÉ** : Binance 2020-03-30 · OKX/Bybit 2020-10-30 · Bitget
  2021-08-12 (OI encore plus profond : OKX 2020-04-17, Bybit 2020-05-08). **Corrige l'ancienne
  hypothèse « lake Binance-only »** qui gatait S2.3/S9 (`coinglass_multi_{funding_rate,open_interest}`).
- **Amberdata = NON** (confirmé par S4) ; **Databento = NE PAS dépenser** ; alt-data = DO-NOT-SPEND.

### 3.4 Statut `funding_squeeze` : `validated survivor` → **WEAKLY_SUPPORTED, non robustement confirmé**

`funding_squeeze_status()` porte l'évidence convergente [VÉRIFIÉ S2/FINDINGS.md] :
- **Ma propre déflation** le re-gate **REJECT** au compte vrai (1869, désormais 1891).
- **S2 4h cluster-aware** : bull **macro-event t=1.50 (NS)**, médiane +25bps seulement (le i.i.d.
  p=1e-4 est anti-conservateur car les squeezes co-firent → null cluster-aware obligatoire) ; bear
  35 trades → **21 macro-events, t=1.23 (NS)** → non-établi en bear.
- **Le signe ne flippe pas** bull↔bear → **pas** pure bull-beta, mais **pas** robustement confirmé.
- Sharpe par trade **0.16 (all-23) / 0.24 (deep)** = bande **real-but-weak** (une fuite lirait ≫1).

→ **Ne PAS traiter le `funding_squeeze` live comme un edge confirmé.** Shadow size 0.

---

## 4. TASK 4 — OOS-FORWARD = PRÉCONDITION DURE DE PROMOTION (la leçon structurante)

**Pourquoi.** S0 (les 23 paires retenues sont toutes vivantes — que S2 lit comme du survivorship,
voir §8.2) et S2 (lake gelé unique) établissent qu'**aucun résultat in-sample, même propre, ne peut
CONFIRMER un edge**. Le split in-sample (derniers 40% par date) est nécessaire mais **pas suffisant**.

**Code.**
- `ForwardOOSEvidence(forward_sharpe, n_trades, window_days, regime_real, leak_replayed)` — l'OOS
  accumulé en **régime réel** (sortie de Forward-Shadow), avec l'audit de fuite AM-4 **rejoué sur le
  survivant mergé** (`leak_replayed`).
- `PromotionBar.evaluate_with_forward(evidence, forward)` — le gate de promotion COMPLET : le screen
  in-sample PUIS la précondition forward. **Strictement additif** : il ne peut JAMAIS racheter un
  sleeve qui rate la barre de déflation (testé : `test_forward_does_not_rescue_a_failing_in_sample_sleeve`),
  et un sleeve qui passe in-sample est REJETÉ tant que l'OOS-forward propre n'a pas été accumulé.
- `evaluate()` (le screen in-sample seul) reste inchangé → contrat Round 1 préservé.

**gp_lock** : 4e condition ajoutée — `forward_validated_candidate_exists` — **UNMET aujourd'hui**.
`gp_lock_status()` retourne `candidate_sleeves_today: []` et embarque le statut downgradé de
`funding_squeeze`. Le verrou GP reste donc engagé, et reflète qu'**aucun sleeve n'est candidat**.

### Le lead REVERSION = PRÉ-REGISTRATION EN ATTENTE (PAS un résultat)

S2 a noté un lead exploratoire (XS funding **inversé**, donc mean-reversion). Loggé ici comme
**pré-registration en attente, à valider en OOS-forward** — *ce n'est pas un résultat* :
- **Cellule exacte** : signe **inversé** (long low-funding / short high-funding), cadence **bi-hebdo**,
  **dollar-neutre**, **23 perps**, **terciles**, fenêtre **180 settlements**.
- **Compte de déflation à battre : 1891** (livre crypto).
- **Validation requise** : barre déflatée @1891 **ET** OOS-forward propre (Task 4) **ET** null
  cluster-aware (R5, car le funding co-fire). Tant que non validé → reste idée, taille 0.

---

## 5. TASK 5 — SPEC DU GATE DU JOYAU (`s10_jewel_gate.py`, 7 tests — CODÉ, NON LANCÉ)

`jewel_combination_gate(...)` : la combinaison = **UN essai pré-enregistré** (`n_trials = book+1`).
Contrôles sur la série combinée :
1. **Null combiné** : random-sign **par sleeve**, **autocorr préservée** (flip du signe de la série
   entière), **énuméré exactement sur les 2^K patterns** quand K petit → p déterministe. NB : le livre
   observé EST le pattern all-+1 (compté), donc **p min = 2/(2^K+1)** → il faut **K≥6 sleeves** pour
   battre le plancher 0.05 — *par design*, sous K=6 la chance-de-signe domine (test dédié).
2. **CPCV / PBO / DSR** sur la série combinée, DSR déflaté à `book_trial_count + 1`.
3. **LOO-Δ-Sharpe > 0 PAR sleeve** (l'unité est le livre, jamais le solo).
4. **Orthogonalité sur semaines co-actives** (réutilise R3).
5. **Précondition OOS-forward** (Task 4).
- **Sécurité** : REJETTE sauf si ≥2 piliers confirmés sont nommés ET tous les sleeves en sont →
  **aucun aujourd'hui**, donc jamais atteint sur un vrai livre. **Tests sur fixtures synthétiques
  uniquement** (incl. un happy-path PROMOTE à 6 piliers distinguables). **NON exécuté sur un pilier.**

Réutilise la machinerie committée (`validation/book.py`, `cpcv.py`, `pbo.py`, `learning/stats.py`) —
ne la reconstruit pas.

---

## 6. TASK 6 — GARDE-FOUS DE PROVENANCE (`s10_provenance.py`, 10 tests)

- **P1 — vue funding** : `check_funding_provenance(sql)` / `assert_deep_funding(sql)` flaggent tout
  backtest lisant le **stub `binance_funding`** (~200 lignes/sym, forward-only depuis 2026-04 — le
  clamp `startTime=0` de `/fapi/v1/fundingRate`) au lieu de **`binance_funding_native`** (profond,
  depuis l'inception) [VÉRIFIÉ S0 ; root cause `FUNDING_DEEP_START_MS=1`]. Regex
  `\bbinance_funding\b(?!_)` : matche le token nu, JAMAIS les vues profondes qui le contiennent en
  préfixe (`_native`/`_on_4h`/`_info`).
- **P2 — format de symbole** : `detect_symbol_format` / `validate_symbol_join` / `normalize_symbol` /
  `assert_joinable` pour les 3 orthographes du lake [VÉRIFIÉ S0] : `BTC/USDT`
  (binance_klines_4h, binance_funding_native), bare `BTC` (coinglass_multi_*), `BTCUSDT`
  (coinglass_funding_rate, binance_funding_info). Une jointure cross-format donne 0 ligne en silence.

---

## 7. ⚠ DÉCISION REQUISE DE MALACHIE — les constantes de la barre

[VÉRIFIÉ contre `STREAM_10_honesty.md` + `STREAM_COMMON.md`] **La doctrine ne fixe que des ancres
lâches** : `MC p<0.05` (plancher), `DSR>0`, `PBO<0.5`, `|ρ|<0.3`, et « **FX plus strict** » de façon
**qualitative** (aucune magnitude). Les valeurs strictes ci-dessous sont **MA reco Round 1
(head-of-research), PAS de la doctrine** — surchargeables via `CRYPTO_BAR`/`FX_BAR`/`CROSS_BAR` :

| Livre | PBO max | DSR min | Statut |
|------|---------|---------|--------|
| crypto | **0.50** | **0.95** | PBO = doctrine ; **DSR 0.95** = barre-probabilité (P(Sharpe vrai > benchmark)) — a un précédent labo (DOSSIER_HERMES) mais reste mon choix de promotion |
| fx_metals | **0.40** | **0.975** | **les deux = mon choix** (« plus strict » sans chiffre dans la doctrine) |
| cross | **0.40** | **0.975** | mon choix |

NB : `min_dsr` est une **probabilité** (≥0.95 = 95% de confiance que le Sharpe vrai > le benchmark de
déflation), pas un Sharpe brut. La doctrine ne fixe que `DSR>0`. **Les chiffres 0.40 et 0.975
n'apparaissent dans AUCUN fichier de doctrine.**

→ **À trancher par Malachie**, pas par moi. Je recommande de les garder (le coût FX bas déplace la
contrainte du coût vers l'overfitting → il faut serrer la déflation côté FX), mais c'est sa décision.

---

## 8. HONNÊTETÉ / PROVENANCE — les 3 nuances que je signale explicitement

1. **1891** = mon arithmétique (1869+22) ; **S2 ne calcule jamais la somme**. [encadré §3.2]
2. **23/23 instruments vivants** = **DONNÉE [VÉRIFIÉ S0/COVERAGE.md]** (23 symboles distincts retenus,
   aucun délisté ; MATIC/POL retiré pour instabilité). L'**interprétation** « survivorship / la queue
   perdante d'un long-only de continuation est absente par construction » est un **propos de S2**
   (**[VÉRIFIÉ S2/FINDINGS.md:138]**), **PAS** une citation de COVERAGE.md (qui présente les 23 comme
   un fait de rétention neutre). N'attribuez pas le caveat survivorship à S0. La conclusion
   (l'OOS-forward est requis) tient sur la donnée, pas sur l'étiquette.
3. **Constantes de la barre** = **mon choix [PRÉSUMÉ doctrine→non]**, pas de la doctrine (§7).

---

## 9. INVARIANTS TENUS

- TDD strict (rouge d'abord) sur chaque helper. 73 tests verts, ruff clean. Aucun fichier live touché
  (config / allowed_skills / market_analyst / pulse jamais effleurés ; le fix des labels live reste sur
  `feat/newdata-shadow`, non dupliqué).
- MC-null = plancher, jamais la barre. LOO = livre, jamais solo. DSR au vrai compte.
- Gate du joyau **NON lancé** (pas de pilier). Rien en live. Shadow size 0.
- Un négatif propre est un succès du labo : S2 v-next l'est, et toute la famille est au ledger.
