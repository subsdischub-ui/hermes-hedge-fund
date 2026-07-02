# Restructuration d'Hermès — 2 juillet 2026

> Mandat opérateur : « Notre agent doit trader cryptos, matières premières et devises, constituant
> ainsi une source de passive income qui leverage les capacités d'auto-improvement de Hermès
> combinées à la puissance de LLM + une approche algo appropriée. »
>
> Méthode : relecture intégrale de la séance (47 messages, artefacts, mémoire), 6 agents d'angle
> Sonnet indépendants + 2 attaquants adversariaux (intégrité anti-piège-activité, statistique
> anti-double-comptage), corrections exigées par les attaquants appliquées avant publication.
> Chaque affirmation chiffrée ci-dessous est sourcée d'un fichier du repo ou d'une API publique
> interrogée le 2026-07-02.

---

## 1. La réponse directe au mandat

**Cryptos — OUI, durci.** Le livre live reste le cœur. Le ré-audit (`research_crypto2/_book.json`)
impose une décision sur S1 (section 3) : le seul edge réel est S2 `funding_squeeze_continuation`
(joint-null p=0.017, seul contributeur positif LOO-Δ +0.015, bear-positive).

**Matières premières — OUI, par la seule voie honnête.** Binance USDT-M liste depuis fin 2025 des
perpétuels métaux (XAUUSDT or, XAGUSDT argent) portant des funding rates à intervalle 4h — le même
axe de microstructure (demande de levier retail) que S2 exploite, sur les rails déjà en production,
à coût zéro. Le funding y est VIVANT (std 4.3–5.5 bps) et arme le signal S2 gelé 10 fois (XAU) et
7 fois (XAG) sur l'historique disponible. C'est ~7–10× trop peu pour un verdict (S2 crypto a été
validé sur 73/77/47 événements par actif, multi-cycles) : statut **collect-now avec re-test à date
fixe et couperet d'échec daté** (section 4) — exactement le statut de top_lsr et du skew 25Δ.
Ceci ne re-litige PAS le programme commodities/FX retail fermé le 2026-06-24 : ce programme a tué
les familles trend/carry/roll-yield/saisonnalité sur données CFD/futures ; l'axe funding-perp
n'existait pas sur ces instruments et n'a jamais été testé — c'est un axe nouveau, pas une réouverture.

**Devises — NON aujourd'hui, porte datée.** (Section 5 — vérification 2026 des venues, conditions
de réouverture pré-enregistrées avec critère d'échec.)

**« Passive income » — vocabulaire abandonné, remplacé par les chiffres réels.** Section 7. Le flux
espéré de S2 seul est de ~13 événements/an, soit un rendement espéré d'environ +2 % à +8 %/an du
capital selon le sizing, avec ~13 % de probabilité (estimation gaussienne, donc optimiste) qu'une
année soit négative. Un revenu mensuel type SMIC exigerait ~$224k–897k de capital. L'entreprise
restructurée est un **pipeline de validation patient à coûts bornés**, pas une machine à revenu à
court terme — la trajectoire vers du capital réel se mesure en années, par construction.

**Auto-amélioration — la version réelle, formalisée et partiellement câblée aujourd'hui.** Pas le
« Hermès Researcher » auto-générateur (tué 5/5 par stress-test) : la boucle gouvernée
collecte → accrual shadow size-0 → re-gate calendaire → barre de promotion → porte humaine.
Deux trous fermés ce jour (collecteur top_lsr, verdict DeadEndsLedger dans le manifest du cron),
calendrier de re-test posé (section 6).

---

## 2. L'architecture restructurée — trois étages

```
ÉTAGE A — LIVRE LIVE (exécute, capital paper)
  mean_reversion_regime + funding_squeeze_continuation + funding_carry_harvest (HOLD-only)
  [S1 ts_mom_vol_scaled : décision opérateur en attente — section 3]
  Rails : Binance USDT-M paper, 7 perps crypto, cadence 4h, selective gate déterministe,
  LLM advisory (DeepSeek V4 primaire — décision gelée ; fallback OpenRouter ring-2.6-1t).

ÉTAGE B — EXPANSION GOUVERNÉE SUR L'AXE VALIDÉ (collecte + shadow, capital zéro)
  Le seul axe porteur d'information = funding. La seule direction d'expansion honnête = plus
  d'ACTIFS sur cet axe, jamais plus de cadence.
  - XAUUSDT / XAGUSDT : collect-now, re-test pré-enregistré (section 4)
  - top_lsr (7 cryptos + XAU/XAG si dispo) : collecte quotidienne câblée ce jour, re-gate ≥ 2027-07
  - skew 25Δ Deribit : collecte active, re-gate 2027-01 → 2027-07
  - Écartés sur preuve : PAXG/XAUT (funding peg-suppressed, std 0.27–0.47 bps, 1 et 0 armements),
    SPX (floor administratif +0.5 bps sur 88,5 % des points), TSLA/NVDA (funding=0 exact chaque
    week-end : « marché fermé » qu'un z-score lirait comme « neutre »), COIN/MSTR (ρ 90j vs BTC
    0.71–0.78 : des paris bitcoin déguisés, zéro pari nouveau).

ÉTAGE C — GOUVERNANCE (ce qui empêche l'étage B de mentir)
  trials_by_book (déflation au vrai N), DeadEndsLedger (désormais visible dans le manifest du cron),
  pré-enregistrements datés avec couperets d'échec, porte humaine sur tout changement du live.
```

---

## 3. Décision n°1 (opérateur) — retirer S1 du slate live, et restructurer le stage-gate

### Les chiffres (vérifiés dans `research_crypto2/`, imputation confirmée par lecture d'`audit_book.py`)
- Le sleeve S1 de l'audit est construit **exclusivement** sur `tsmom_score_vectorized`
  (ts_mom_vol_scaled) — `mean_reversion_regime` n'y entre pas et **n'est pas concerné** par le retrait.
- LOO-Δ Sharpe = **−0.146** : le livre entier gagnerait +0.146 de Sharpe (vs 0.074 actuel, soit plus
  que doubler) en retirant S1 seul. Joint-null du panier S1 : **p = 0.693** (69 % des tirages
  aléatoires font aussi bien). ρ(S1,S2) = 0.348 : même son risque n'est pas diversifiant.
- Fenêtre live : 7 trades clos (+$4.15). Seulement **2** sont du S1 strict (+$2.74, tous deux le
  2026-06-24, fenêtre haussière) ; 4 viennent de skills legacy déjà retirés ; 1 de mean_reversion_regime.
  n=2 ne contredit pas p=0.693 sur 645 événements de backtest.
- La « dé-pondération » n'existe pas techniquement : `risk/sizing.py` ignore skill_id ; le seul
  levier live est le gate binaire `allowed_skills`. Retrait sec ou statu quo.

### La tension que le retrait crée (exigence des attaquants : la poser explicitement)
Le stage-gate actuel exige **200 paper trades**. Au rythme observé (0.66 trade/jour, porté par la
famille S1/legacy), le gate serait atteint en ~10 mois. Sans S1, le flux retombe à
mean_reversion (rare) + S2 (~13 évts/an) + S3 (HOLD-only, 0 trade à ce jour) : ordre de grandeur
**10–15 ans**. Retirer S1 sans toucher le gate rend le plan de paliers inatteignable en silence.

### La résolution honnête : le gate lui-même était mal calibré
Un gate à 200 trades nourri par un skill mesuré comme un drag ne valide rien : le franchir sur du
bruit bull-beta, puis monter du capital dessus, c'est précisément scaler du bruit — le mécanisme
que toute la machine d'honnêteté existe pour empêcher. Le critère `walk_forward_sharpe: 1.0` ne
passerait de toute façon jamais avec S1 dedans (Sharpe book 0.074).

**Recommandation (à trancher par l'opérateur, les deux gestes ensemble) :**
1. Retrait sec de S1 — diff en 3 fichiers, prêt à appliquer :
   `config/default.yaml:100` (retirer ts_mom_vol_scaled d'allowed_skills),
   `combination.py:37` (TREND_SKILLS = []), `risk/selective.py:25` (_CONVICTION_GATED = set()).
   Le test `test_combination_slate_matches_live_allowed_skills` valide la synchronisation ;
   `pytest` complet obligatoire avant commit.
2. Remplacer le critère `paper_trades: 200` par un critère d'évidence adapté à un livre
   d'événements rares — proposition : « ≥ 40 événements S2 armés en paper OU confirmation forward
   (accrual shadow positif sur ≥ 12 mois) + aucun bris de guardrail + verdict book (DSR/joint-null)
   maintenu au re-audit ». Un gate honnête sur un livre à un pilier prend des années quel que soit
   son libellé — le dire est plus honnête que compter des trades de bruit.
3. Pré-enregistré (exigence attaquant) : si 6 mois après le retrait la cadence paper est
   < 2 trades/mois, re-consultation opérateur sur le calendrier — pas de retour silencieux de S1.

Conséquence à accepter les yeux ouverts : le dashboard deviendra **calme** — des semaines sans
trade clos. « Les sleeves au repos dans le chop » est une protection du capital, pas une panne.

---

## 4. Matières premières — pré-enregistrement XAU/XAG (écrit AVANT tout backtest)

### Faits vérifiés (2026-07-02, API Binance fapi, scripts sous `research_restructure/`)
| Perp | Funding depuis | Barres 4h | Événements z≥2 (no-overlap) | Funding std | Vol 24h |
|---|---|---|---|---|---|
| XAUUSDT | 2025-12-11 | 1 219 | **10** | 4.30 bps | $1.85B |
| XAGUSDT | 2026-01-07 | 1 056 | **7** | 5.47 bps | $1.46B |
| PAXGUSDT | 2025-03-27 | 2 772 | 1 | 0.47 bps | $125M |
| XAUTUSDT | 2026-03-26 | 587 | 0 | 0.27 bps | $42.8M |

Spreads mesurés 0.02–1.63 bps — l'exécution n'est pas le blocage. Le blocage est la puissance
statistique : à N=10, l'écart-type de l'estimateur de Sharpe par trade (~0.32× l'écart-type par
trade) rend toute affirmation directionnelle infalsifiable.

### Le pré-enregistrement (les couperets exigés par les attaquants)
1. **Règle gelée.** La règle S2 exacte (`signals.py` : fenêtre z = 180 barres, seuil z ≥ 2.0,
   hold = 24 barres, long-only) est déclarée **non re-tunable sur métaux**. Tout retuning = essai
   séparé, compté au registre `fx_metals` (déjà à 53).
2. **Un sous-jacent = un pari.** XAU+PAXG+XAUT = UN pari (or) ; XAG = un second (argent). Le budget
   d'essais du futur test = 2, jamais 3–4. Avant de garder les deux métaux au livre :
   ρ(résidus S2 XAU, résidus S2 XAG) < 0.5 exigé, sinon un seul.
3. **Le tri d'aujourd'hui est un filtre de faisabilité, pas un essai.** Retenir XAU/XAG et écarter
   PAXG/XAUT se fonde sur la variance du funding (vivant vs peg-suppressed), constatée avant tout
   calcul de rendement — déclaré ici pour interdire toute requalification post-hoc.
4. **Couperet d'échec daté.** Re-test au plus tôt quand n ≥ 40 événements dé-chevauchés par
   sous-jacent ET couverture d'au moins un cycle baissier. Échéance ferme : **2027-07-02**. Si à
   cette date n < 30 OU aucun régime bear couvert → **axe mort**, aucune re-proposition avant
   2028-07 minimum. (Au rythme observé ~1.4–1.7 évt/mois sur XAU, n≈30–40 vers 2027-08 : l'échéance
   est réaliste, pas complaisante.)
5. **Orthogonalité à prouver, pas à supposer.** ρ(XAU, BTC) mesuré : 0.256 sur 459 j mais 0.52–0.68
   sur les fenêtres 30–90 j récentes — instable. La fenêtre de jugement est gelée à : ρ all-history
   quotidien ET stabilité exigée sur 30/90 j/all (les trois < 0.30 pour le label « orthogonal »).
   Interdiction de choisir la fenêtre après coup.
6. **Barres complètes au re-test.** Joint-null date-level dé-chevauché p < 0.05 stable 5 seeds,
   OOS-by-date, déflation contre le N réel du registre fx_metals, coût taker 15 bps RT, et
   incrément du registre AVANT le calcul.
7. **Aucune donnée n'est perdue en attendant** : le funding Binance est intégralement
   re-téléchargeable depuis le listing — contrairement à top_lsr, aucune collecte d'urgence n'est
   nécessaire pour les métaux.

---

## 5. Devises — NON aujourd'hui, porte datée (vérifié le 2026-07-02)

**L'accès n'existe pas.** Interrogation en direct des 4 venues perp majeures : Bybit (710 symboles
linear), OKX (409 swaps), Hyperliquid (231 perps), dYdX v4 (296 marchés) — **aucun perp FX majeur
(EUR/GBP/JPY/AUD/CAD/CHF/NZD) liquide n'existe nulle part**. Le seul listing nominal, EUR-USD sur
dYdX, affiche $0 de volume 24h et $2 651 d'OI : un listing fantôme. Les stablecoins EUR sur Binance
plafonnent à $38.8M/24h (EURUSDT) — très sous le seuil de $100M — et ne portent aucun axe
exploitable (flux trop fins, peg sans tension informative).

**Deux fermetures de nature différente, à ne pas confondre :**
- (a) Les familles directionnelles et le carry FX retail sont **morts par la preuve**
  (carry G10 rank-shuffle p=0.28 même crash-protégé, carry cross-sectionnel p=0.804, trend/session/
  mean-reversion/breakout tous réfutés) — ne reviennent pas sans donnée propriétaire neuve. Ne PAS
  re-litiger, ne PAS monter d'infra MT5/JustMarkets pour ce volet.
- (b) L'axe funding/microstructure — le seul axe vivant d'Hermès — n'a **jamais pu être testé** sur
  FX : l'instrument n'existe pas sur des rails accessibles. Ce n'est pas un verdict négatif sur
  l'axe, c'est une impossibilité d'accès. La porte reste une porte, pas un mur.

**Conditions de réouverture pré-enregistrées (datées, avec critère d'échec) :**
- **A — Ré-audit d'accès : 2027-07-02.** Re-interroger les venues pour un perp FX majeur avec
  vol 24h ≥ $100M ET ≥ 12 mois de funding continu. Si absent → fermé 12 mois de plus (2028-07-02),
  aucune re-vérification anticipée motivée par l'envie d'activité.
- **B — Si un perp qualifiant apparaît :** règle S2 GELÉE (z 180 barres, z ≥ 2, hold 24 barres,
  non re-tunable), pré-enregistrement avant tout calcul, ≥ 30 événements dé-chevauchés + couverture
  d'un régime bear pour tout verdict — sinon collect-now.
- **C — Stablecoins EUR (secondaire) : 2027-01-02.** Un stablecoin EUR dépasse-t-il $100M vol/24h ?
  Sinon, pas de re-test avant 2027-07-02.

**Exposition indirecte déjà présente :** l'or (XAUUSDT, section 4) est corrélé négativement au
dollar (ρ retours gold/DXY = −0.355 à −0.384, stable sur 2/3/5 ans) — le livre acquiert donc, via
les métaux, une exposition macro-devises partielle **anti-USD à une dimension**. C'est un
sous-produit réel mais ce n'est pas un volet devises multi-paires, et ce dossier ne le présente
pas comme tel.

---

## 6. La boucle d'auto-amélioration — formalisée, et deux trous fermés ce jour

La version honnête de « l'auto-improvement » n'est pas un agent qui s'auto-modifie (Hermès
Researcher : tué 5/5 — gouvernance non câblée, hallucinations, pulsion d'activité rhabillée).
C'est ceci, et ça tourne :

| Étage | Quoi | État |
|---|---|---|
| 1. Collecte | skew 25Δ Deribit (`collect_skew`) | ✅ câblé, accumule |
| | funding_squeeze shadow size-0 (`shadow_funding_squeeze`) | ✅ câblé (0 lignes = pas de squeeze, normal) |
| | xs_reversion shadow | ✅ câblé, gate OFF volontaire (fragile in-sample) — ne pas activer |
| | **top_lsr** (rétention Binance ~30 j : perte irréversible chaque jour sans collecte) | ✅ **câblé ce jour** (voir ci-dessous) |
| | comex dated contracts / macro SEP-FF | ⚠️ seams NotImplementedError — priorité basse (l'axe métaux se joue sur le funding perp, déjà couvert) |
| 2. Accrual shadow | positions théoriques size-0, log PIT idempotent (`logio.append_snapshot`) | ✅ pattern éprouvé |
| 3. Re-gate calendaire | top_lsr : ≥ 2027-07 (12 mois de collecte), point d'étape 2027-01 sans verdict. skew : 2027-01→2027-07. XAU/XAG : couperet section 4 (2027-07-02). | 📅 posé ici |
| 4. Barre de promotion | s10_governance (joint-null 5 seeds, OOS-by-date, déflation, orthogonalité) | ✅ existe, testé |
| 5. Porte humaine | rien n'entre dans `allowed_skills` sans merge opérateur | ✅ la norme, inchangée |

**Câblé ce jour** (TDD, aucune modification du chemin de trading live) :
- `src/hermes/forward/collect_top_lsr.py` : collecteur quotidien des deux ratios top-trader
  (account + position) pour les 7 perps live **+ XAUUSDT + XAGUSDT** — les 18 paires
  (9 symboles × 2 ratios) vérifiées non-vides sur l'API avant écriture du code. Fail-soft par
  paire, écriture PIT idempotente (`logio.append_snapshot`, dédup par (symbol, ratio_type)),
  enregistré dans `default_collector_names()` → le cron Dokploy le ramasse au prochain déploiement.
  Le module documente explicitement que le DeadEndsLedger n'interdit que l'usage DIRECTIONNEL de
  top-LSR (le contrarian S2.1, mirage 30 jours) — la collecte pure ne le ressuscite pas, elle rend
  seulement testable un futur axe positionnement sur historique profond au lieu de zéro.
- `prereg.dead_ends_verdicts()` + manifest `dead_ends_check` dans `run.py main()` : le verdict
  `DeadEndsLedger.check()` (BLOCKED / ALLOWED-NEW-AXIS / NOT-A-DEAD-END) est désormais imprimé
  pour chaque pré-enregistrement à chaque run du cron — informatif, non-bloquant, fail-soft.
  Ferme le trou de gouvernance « le ledger existe mais personne ne le consulte ».
- Tests : 13 nouveaux (61 verts sur tests/forward/), ruff propre sur les 6 fichiers touchés.

---

## 7. L'arithmétique honnête (ex-« passive income »)

Chiffres sourcés de `_s2_funding_squeeze.json` (per-trade Sharpe 0.26–0.37, OOS-40 ≈ 0.28–0.33 —
jamais le « +1.1 » annualisé que le ré-audit a fait restater) :

- S2 = ~13 événements/an sur le panier BTC/ETH/SOL dé-chevauché (115 événements / 8.85 ans).
- Rendement espéré annuel de S2 seul : **≈ +2.1 %** du capital au sizing actuel (0.5 %/trade),
  **≈ +8.3 %** au sizing des stages 2–3 (2 %/trade, jamais atteints).
  En dollars : $100 → $2–8/an ; $1 000 → $21–83/an ; $10 000 → $207–829/an.
- Variance : ~13 % de probabilité qu'une année soit négative (modèle gaussien, donc **sous-estimé** —
  les vraies queues sont plus lourdes : SOL bear −397 bps/évt sur 5 événements).
- Capital pour que l'espérance mensuelle égale un SMIC (~$1 550/mois) : **~$897k** au sizing 0.5 %,
  **~$224k** au sizing 2 %. Hors de portée du stage-gate actuel (equity floor $250).
- L'extension métaux, si elle passe un jour la barre, ajoute de la **fréquence** d'événements,
  pas de l'edge par événement — elle améliore la cadence de test et la diversification, pas
  l'ordre de grandeur du calcul ci-dessus.
- Coûts d'infra : aucun chiffre dans le repo. **Action opérateur : relever les factures réelles
  (VPS Dokploy + DeepSeek + OpenRouter)** — les ordres de grandeur de marché (~$10–35/mois)
  suggèrent que l'infra coûte aujourd'hui plus que l'espérance de S2 à capital ≤ $1 000.

**Couperet global pré-enregistré (exigence attaquant) :** si au **2028-01-02** (18 mois) le livre
combiné n'a ni un second pilier ayant passé la barre complète (métaux, top_lsr, skew ou autre),
ni une confirmation forward de S2, le narratif « revenu par trading » de ce véhicule est
officiellement abandonné — l'entreprise reste un véhicule de R&D/preuve de capacité, et la
conversation redevient celle du 2026-07-01 (l'actif = la capacité démontrée, pas le P&L).

---

## 8. Gouvernance — réconciliation des registres (constatée ce jour)

Trois compteurs coexistaient pour le livre crypto : `governance/trials_by_book.json` = **1906**
(canonique, committé), volume prod `/data/governance` = 1892 (en retard des 14 essais des sessions
de recherche locales : alpha_carry +8, beta_illiquid +6), `data/trials_by_book.json` = 1870
(fossile local gitignoré du bug corrigé en 47abf72, **sous-ensemble strict vérifié** de
governance/ — supprimé du disque ce jour ; son homologue sur le volume prod `/data/` est un
fossile équivalent, inoffensif, plus rien ne le lit). Règle : `governance/` (repo) est LA vérité
pour tout calcul de déflation ; tout pré-enregistrement fait côté prod doit être miroité en commit
(comme 3fc6dbf l'a fait). Item d'ingénierie ouvert (petit) : seed/merge idempotent du volume
depuis l'image au déploiement.

---

## 9. Récapitulatif des décisions

**Décidé et appliqué ce jour (additif, hors chemin live) :**
- Collecteur top_lsr quotidien + verdict DeadEndsLedger dans le manifest du cron (TDD).
- Pré-enregistrements datés : métaux (section 4), FX (section 5), couperet global 2028-01 (section 7).
- Suppression du registre fossile local `data/trials_by_book.json` (sous-ensemble strict vérifié).

**En attente de l'opérateur :**
1. Retrait de S1 + restructuration du critère de stage-gate (section 3 — recommandé, diff prêt).
2. Relevé des coûts d'infra réels (factures) pour ancrer la section 7.
3. Le choix du vocabulaire : ce dossier remplace « passive income » par « rendement espéré positif
   à variance élevée, en validation » dans toute communication interne — à entériner.

---

## Addendum — 2026-07-02 soir : le 5e mensonge attrapé, et le test de largeur

**Un bug de look-ahead a été découvert dans la machinerie d'honnêteté elle-même** — par son
propre processus adversarial, pendant le test de largeur S2. `research_fx/fxgate.py::
_basket_event_return` faisait « entrer » les instruments pas encore listés à la date d'un
événement sur leur toute première barre historique (des mois/années plus tard). Corrigé
(fix symétrique jambe réelle/null), puis TOUS les chiffres fondateurs ont été re-calculés :

| Chiffre décisionnel | Avant fix | Après fix | Décision |
|---|---|---|---|
| S2 basket joint-null p (max 5 seeds) | 0.0235 | **0.0005** | S2 seul edge réel — **renforcé** |
| S1 basket joint-null p | 0.693 | 0.507 | S1 indiscernable du hasard — **confirmé** |
| S1 sleeve mean | +275.8 bps | +64.6 bps | S1 encore plus faible qu'affiché |
| Book LOO-Δ (S1/S2/S3), Sharpe, DSR, PBO | — | **bit-à-bit identiques** | jamais touchés par le bug |

Aucune décision de la restructuration n'est inversée ; deux sont renforcées. Les chiffres
affichés (dashboard, chat, config) ont été mis à jour — y compris les deux notes de cartes
stratégies qui portaient encore les grandeurs gonflées pré-ré-audit (« +1.1 », « +2.4..+4.5 »).

**Test S2-largeur (PREREG `research_breadth/`, 2 essais comptés au registre → crypto 1908) :**
univers figé de 12 perps (7 référence + ZEC/XRP/BNB/DOGE/TLM, critères écrits avant lecture).
Résultat post-fix : **+75.8 % d'événements marginaux** (91 événements date-level dé-chevauchés),
joint-null p_max5 = 0.003, edge marginal +211 bps/évt (unité panier-synchronisé, NON comparable
au per-trade). **Verdict mécanique à la lettre du PREREG : ÉCHEC** — une fenêtre bear (n=4,
2024-07→09) reste négative, et la règle à zéro tolérance s'applique telle qu'écrite. Deux
réserves structurelles relevées par l'audit : (1) **biais de survie non testé** — l'univers ne
contient que des survivants, signe défavorable au chiffre publié ; (2) **2022 : zéro événement
z≥2 sur les 12 symboles** — le bear-check est structurellement aveugle au cœur du crash (S2
est un phénomène de rebond, pas de chute). Suite honnête : soit UN test durci pré-enregistré
(univers reconstruit AVEC les perps délistés — un test plus dur, pas plus facile), soit
clôture de la piste. Pas de re-découpage des fenêtres, pas de 3e chance.

**Éclaireurs hors-repo :** token unlocks = axe verrouillé par les données (toute l'industrie a
mis l'historique derrière un paywall ; le pattern J-30 est déjà publié par Keyrock sur 16k
événements — valeur différentielle douteuse ; option : CryptoRank Basic $19/mois, décision
opérateur, ou collecte gratuite quotidienne). Cross-venue = collect-now (Bybit : 6,3 ans de
funding vérifiés — piste de largeur supplémentaire documentée ; Hyperliquid : funding horaire,
la règle gelée ne s'y transpose pas sans requalification ; re-test ≥ 2026-11).

**Réponse à la question opérateur « comment le système génère d'ici les échéances » :** la
largeur était le levier le plus court et il vient d'être testé — il ne passe pas la barre à la
lettre aujourd'hui. Le flux du livre reste celui du slate validé. La variable dominante du
revenu reste le capital, pas le signal ; le chemin par paliers ($100 paper → gates → réel)
reste le seul qui ne scale pas du bruit.
