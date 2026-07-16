# JOURNAL — Étape 2/3 : gate S2-en-largeur (pré-enregistré, un seul essai)

Date : 2026-07-02. Source de vérité : `research_breadth/PREREG.md` (figé avant tout calcul).
Script : `research_breadth/gate_s2_breadth.py`. Résultats bruts : `research_breadth/_breadth_results.json`.

> **AVIS — le run ci-dessous (§0-7) est CONTAMINÉ par un bug de look-ahead dans
> `research_fx/fxgate.py::_basket_event_return`** : un instrument dont l'historique commence
> APRÈS la date d'un événement était inclus quand même (searchsorted retourne l'indice 0, donc
> le calcul "entre" sur la toute première barre historique de l'instrument, des mois/années
> après l'événement réel — un rendement hors-contexte, aléatoire en signe). Bug trouvé et
> corrigé par un audit adversarial le 2026-07-02 (commentaire `LOOK-AHEAD FIX` dans le code).
> Ce run historique est **conservé tel quel ci-dessous pour la trace** (ne pas supprimer/éditer
> les chiffres buggés) — le verdict et les chiffres à utiliser sont ceux du
> **§8 "Re-run post-fix"** plus bas, qui est le run officiel sur le fxgate corrigé.
> Sauvegarde brute du run buggé : `research_breadth/scratch_crossvenue/_archive_buggy_run/_breadth_results_BUGGY_prebugfix.json`.

## 0. Registre de déflation (bump AVANT lecture des résultats)

`governance/trials_by_book.json`, book `crypto`, +1, source `s2_breadth_prereg`, via
`BookTrialRegistry` (`HERMES_GOVERNANCE_DIR=governance`).

- **N crypto avant bump : 1906**
- **N crypto après bump : 1907** (vérifié en relisant le fichier après écriture)
- Fichier modifié mais **NON committé** (laissé à l'orchestrateur, conformément à la mission).

## 1. Règle S2 gelée — appliquée telle quelle

Copiée verbatim de `research_crypto2/audit_s2_funding_squeeze.py` (`funding_z`, `long_events`) :
fenêtre z = 180 barres 4h, seuil z ≥ 2.0, long-only, hold = 24 barres (~96h), warmup = 200
barres (aligné shadow-live, >= la fenêtre 180), coût taker 15 bps round-trip. Loaders 4h +
funding as-of repris de `research_crypto2/cryptolib.py` (`resample_4h`, `attach_funding`),
repointés sur `research_breadth/data/` (marché **FUTURES fapi**, comme établi à l'étape 1 —
et non `data/intraday/` qui est SPOT). Gate honnête (`event_basket_gate`, no-overlap au
niveau DATE, `oos_sharpe_by_date`) importé verbatim de `research_fx/fxgate.py`.

Aucun paramètre modifié, aucune grille, aucun retrait de symbole.

## 2. Univers

- Panier référence (7, figé) : BTC, ETH, SOL, AVAX, SUI, ARB, WIF.
- Univers élargi (12, figé à l'étape 1) : + ZEC, XRP, BNB, DOGE, TLM.

Événements bruts par symbole (avant dé-chevauchement panier, warmup=200) :

| Symbole | n événements bruts | span (années) |
|---|---|---|
| BTC | 472 | 6.81 |
| ETH | 500 | 6.60 |
| SOL | 279 | 5.80 |
| SUI | 149 | 3.16 |
| AVAX | 356 | 5.77 |
| ARB | 209 | 3.28 |
| WIF | 67 | 2.45 |
| ZEC | 491 | 6.40 |
| XRP | 471 | 6.49 |
| BNB | 415 | 6.39 |
| DOGE | 389 | 5.98 |
| TLM | 132 | 3.26 |

## 3. Paniers dé-chevauchés au niveau DATE (`event_basket_gate`, `no_overlap=True`)

| | Référence (7) | Élargi (12) |
|---|---|---|
| n événements (dé-chevauchés) | 120 | 167 |
| span (années, calendrier commun) | 6.81 | 6.81 |
| n événements / an | **17.61** | **24.51** |
| edge net moyen / événement | **-174.0 bps** | **-39.1 bps** |
| Sharpe par événement | -0.211 | -0.052 |
| OOS-Sharpe (40% par date) | +0.275 | +0.252 |
| MC-p joint (seed 0) | 0.0008 | 0.0014 |

**Événements marginaux** (dates armées par l'univers élargi et NON armées par le panier de
référence, au niveau date après dédup) :
- **n = 91 dates marginales**, soit **13.35 évts/an marginaux**, **+75.8 %** vs les 120
  événements du panier référence — **franchit le seuil pré-enregistré de +50 %**.
- Edge net moyen des événements marginaux (pricé sur le panier élargi complet) :
  **-59.2 bps/évt** (n=91).
- Répartition par année (mono-régime ?) : 2020=19, 2021=14, 2023=18, 2024=15, 2025=13,
  2026=12 (partiel, à mi-2026). Répartition étalée sur 6 années civiles distinctes — **pas
  mono-régime** au sens temporel.

## 4. Barres sur le panier ÉLARGI (12 symboles)

### 4.1 Joint-null MC date-level, 5 seeds
`mc_p_5seeds = [0.0014, 0.0006, 0.0010, 0.0014, 0.0004]` → **max = 0.0014 < 0.05, stable
sur les 5 seeds**. Le panier élargi bat largement le null random-entry date-level joint.

### 4.2 OOS-by-date (moitiés calendaires strictes, panier élargi dé-chevauché)
- Première moitié (in-sample) : mean net = **-234.5 bps/évt**, Sharpe = -0.364, n=79.
- Seconde moitié (OOS) : mean net = **+136.4 bps/évt**, Sharpe = +0.170, n=88.
- (Le chiffre officiel `oos_sharpe_40pct` du gate = +0.252, cohérent en signe.)

### 4.3 Bear-window check (BTC drawdown, sous-fenêtres macro fusionnées gap≤60j)
Note méthodologique : un premier calcul naïf du "drawdown>25%" sur un running-max brut
fragmentait le signal en 47 micro-segments techniques (faux positifs dès qu'un rebond
franchissait temporairement le seuil). Corrigé en fusionnant les segments contigus séparés
de moins de 60 jours — ceci est une correction de **détection de fenêtre**, PAS un
re-tuning de la règle S2 (qui reste z=180/2.0/24/200/15bps intacte).

| Fenêtre | Dates | n événements (panier élargi, non dédupliqué) | edge net moyen |
|---|---|---|---|
| 2022 complet (calendaire) | 2022-01-01 → 2022-12-31 | **0** | n/a |
| drawdown25_seg0 | 2019-09-30 → 2020-04-28 | 127 | **-470.0 bps** |
| drawdown25_seg1 | 2021-01-21 → 2021-10-04 | 680 | +155.0 bps |
| drawdown25_seg2 | 2021-12-04 → 2024-02-26 (LUNA/FTX + suite) | 1017 | **-58.1 bps** |
| drawdown25_seg3 | 2024-07-05 → 2024-09-09 | 4 | **-591.2 bps** |
| drawdown25_seg4 | 2025-02-28 → 2025-04-11 | 7 | +779.6 bps |
| drawdown25_seg5 | 2025-11-16 → 2026-07-02 | 104 | +49.3 bps |

Constat factuel important : **aucun des 12 symboles n'a un seul événement z≥2.0 en 2022**
(vérifié individuellement) — un funding-squeeze extrême haussier est structurellement un
phénomène de bull/rebond, pas de bear complet ; 2022 est absent du signal par construction,
pas par un artefact de calcul. Le grand segment 2021-12→2024-02 (qui contient 2022 dans son
étendue calendaire) tire donc ses 1017 événements presque entièrement de la période
2023-2024 post-LUNA/FTX, pas de 2022 lui-même.

**Bear-check : 3 des 6 fenêtres bear/drawdown avec des événements réels sont NÉGATIVES**
(seg0 -470bps, seg2 -58bps, seg3 -591bps) → **le panier élargi ÉCHOUE le bear-window check**
pré-enregistré ("le panier élargi doit rester positif dans les sous-fenêtres bear comme S2
l'est").

### 4.4 DSR contre le N réel du registre (1907)
DSR (panier élargi, rendements bruts des 167 événements dé-chevauchés, variance de Sharpe
estimée sur les Sharpe per-asset + per-basket disponibles, n_trials=1907) ≈ **3.7e-16**
(effectivement nul). Sans surprise étant donné que le sleeve_mean net du panier élargi lui-même
est négatif (-39.1 bps/évt) — un edge dont la moyenne brute est négative ne peut pas avoir de
Sharpe déflaté significatif.

## 5. VERDICT MÉCANIQUE (contre les critères d'échec écrits AVANT le calcul)

Critères du PREREG :
1. Marginaux < +50 % → échec largeur. **Résultat : +75.8 % → ce critère est PASSÉ** (la
   largeur ajoute bien plus d'événements indépendants qu'exigé).
2. Joint-null / bear-check ratés → échec. **Joint-null PASSÉ** (p max 5-seeds = 0.0014 < 0.05).
   **Bear-check ÉCHOUÉ** (3/6 fenêtres bear négatives, dont deux fortement : -470bps, -591bps).

**VERDICT : ÉCHEC.**

Un seul critère d'échec suffit à tuer le test ("Si le panier élargi échoue... le bear-check :
la largeur DILUE l'edge — verdict négatif propre"). C'est le cas ici : le débit d'événements
augmente bien comme espéré (+75.8% marginaux, 24.5 évts/an au lieu de 17.6), et le joint-null
passe très confortablement — mais l'edge net moyen du panier élargi est **négatif** en absolu
(-39.1 bps/évt, à comparer à l'edge lui-même négatif du panier référence recalculé ici sur
cette base -174.0 bps/évt, cf. §6), et le panier élargi perd franchement dans 2 des 3 fenêtres
bear/drawdown documentées qui contiennent des événements réels.

## 6. Réserve méthodologique majeure — à porter à l'attention de l'opérateur

Le panier de RÉFÉRENCE (7 symboles), recalculé ICI avec le gate `event_basket_gate` de
`fxgate.py` (moyenne PAR ÉVÉNEMENT à travers tous les instruments simultanément armés,
dé-chevauché au niveau date), affiche un edge net moyen de **-174.0 bps/évt** — très
différent des chiffres de validation antérieurs de S2 (+30 à +70 bps/évt, joint-null p=0.017,
mémoire `hermes-strategy2-funding.md`). Deux différences honnêtes à noter, sans qu'aucune
n'ait été retouchée pour "sauver" le test (le PREREG l'interdit) :
  - **Source de données** : ce test utilise le marché FUTURES (fapi, `research_breadth/data/`)
    découvert par l'agent de l'étape 1, alors que les audits antérieurs (dont
    `audit_s2_funding_squeeze.py` original) tournent sur `data/intraday/` qui est du marché
    SPOT. Un écart de prix systématique (médiane ~$16 sur BTC, jusqu'à ~$937) a été documenté
    à l'étape 1 entre les deux sources — cela peut suffire à inverser le signe d'un edge mince.
  - **Unité de mesure** : l'audit original mesure le edge **par instrument** (BTC seul, ETH
    seul, SOL seul) puis regarde le "basket joint null" séparément comme un test de robustesse
    statistique — il ne calcule PAS le "mean edge per event" comme moyenne simultanée
    cross-instruments à chaque date armée (ce que fait `event_basket_gate.sleeve_mean`, exigé
    ICI par le PREREG comme l'unité honnête du panier élargi). Ce sont deux quantités
    différentes : le premier peut être positif par instrument tandis que le second (qui moyenne
    les instruments qui tirent dans des sens/forces différentes le même jour) peut être
    négatif si les instruments corrélés au signal amplifient les faux positifs synchronisés.

Ce test a suivi le PREREG à la lettre (même méthode `event_basket_gate` appliquée
identiquement au panier référence ET au panier élargi, mêmes données futures pour les deux,
aucune différence de traitement entre les deux paniers) — la comparabilité référence-vs-élargi
est donc honnête et interne au test. Mais la divergence externe avec les chiffres antérieurs de
S2 signale que le passage SPOT→FUTURES et/ou la bascule vers l'unité "moyenne panier
inter-instruments" changent substantiellement le signe de l'edge mesuré. **Ceci n'invalide pas
le verdict ÉCHEC de ce test** (les deux paniers ont été traités à l'identique), mais cela
suggère que le chiffre "+30-70bps/évt" cité dans le PREREG comme référence de S2 pourrait
lui-même reposer sur SPOT plutôt que sur FUTURES, et mérite d'être ré-audité séparément (hors
mandat de cette mission — aucun retrait/re-tuning n'a été fait ici).

## 7. Conclusion

La largeur d'univers déclenche bien un débit d'événements plus élevé et statistiquement
propre (joint-null passé, marginaux +75.8% bien au-delà du seuil), mais l'edge net du panier
élargi reste négatif en absolu et échoue le bear-window check pré-enregistré. Conformément au
PREREG : **verdict ÉCHEC — la largeur ne débloque pas de flux honnête ; S2 reste sur son
panier validé (dans sa mesure d'origine) ; fin de cette piste telle que testée.**

Aucun symbole n'a été retiré, aucun paramètre n'a été re-tuné, aucune fenêtre OOS n'a été
changée après avoir vu les résultats. Le registre de déflation a été incrémenté AVANT tout
calcul (N crypto 1906→1907).

---

## 8. Re-run post-fix (2026-07-02) — gate officiel sur `fxgate.py` corrigé

### 8.0 Contexte : le bug et sa correction

Un audit adversarial a découvert, le 2026-07-02, un bug de **look-ahead** dans la fonction
partagée `research_fx/fxgate.py::_basket_event_return` : lorsqu'un instrument du panier n'avait
pas encore d'historique à la date d'un événement (listing plus tardif que l'événement — cas
fréquent pour SUI/ARB/WIF/ZEC/TLM dont le span est 2.4–3.3 ans vs 5.8–6.8 ans pour BTC/ETH/SOL),
`np.searchsorted(ts, target, side="left")` retournait l'indice `0` et le code "entrait" la
position sur la **toute première barre historique de l'instrument** — des mois ou des années
après la vraie date de l'événement, dans un régime de marché sans rapport. Ce rendement
garbage polluait la moyenne du panier à chaque événement où au moins un instrument pré-listing
était concerné. Le fix (déjà en place dans le fichier, commentaire `LOOK-AHEAD FIX`) exclut
proprement l'instrument de la moyenne de cet événement si `target < ts[0]` de l'instrument.

Ce fix de méthode de mesure compte comme un essai au sens du registre de déflation (une
correction qui change matériellement le chiffre mesuré, même si elle corrige un bug plutôt que
d'ajuster un paramètre).

### 8.1 Registre de déflation — bump #2

`governance/trials_by_book.json`, book `crypto`, +1, source
**`s2_breadth_prereg_v2_gatefix`**, via `BookTrialRegistry`
(`HERMES_GOVERNANCE_DIR=governance`).

- **N crypto avant ce bump : 1907** (déjà incrémenté une fois pour `s2_breadth_prereg`, §0)
- **N crypto après ce bump : 1908** (vérifié en relisant le fichier après écriture)
- Fichier modifié mais **NON committé** (laissé à l'orchestrateur, conformément à la mission).
- Provenance crypto complète après bump : ... `s2_breadth_prereg` (n=1), `s2_breadth_prereg_v2_gatefix` (n=1).

### 8.2 Ré-exécution

`research_breadth/gate_s2_breadth.py` relancé **tel quel, aucun changement de script** — le
fxgate corrigé est ramassé automatiquement via l'import (`from fxgate import event_basket_gate,
...`). `__pycache__` purgé avant le run (`research_breadth/`, `research_crypto2/`,
`research_fx/`) pour garantir qu'aucun bytecode compilé de l'ancienne version de `fxgate.py`
n'était réutilisé. Aucun cache de résultats interne au script (le fichier `_breadth_results.json`
est réécrit intégralement à chaque exécution — pas de logique de cache à purger côté script).

### 8.3 Chiffres officiels (post-fix)

**n_events et cadence :**

| | Référence (7) | Élargi (12) |
|---|---|---|
| n événements bruts par symbole | cf. §2 (inchangé — le bug est dans le gate, pas dans la détection d'événements) | idem |
| n événements dé-chevauchés (panier) | 120 | 167 |
| span (années, calendrier commun) | 6.8145 | 6.8145 |
| **n événements / an** | **17.609** | **24.507** |
| n dates marginales (élargi \ référence) | — | **91** |
| n marginaux / an | — | **13.354** |
| % marginaux vs référence | — | **+75.83 %** |

**sleeve_mean net (bps/évt) et Sharpe :**

| | Référence (7) | Élargi (12) |
|---|---|---|
| **sleeve_mean_net_bps** | **+433.60** | **+270.33** |
| sleeve_sharpe (par événement) | +0.4178 | +0.3039 |
| oos_sharpe (40 % par date, officiel du gate) | +0.4780 | +0.3162 |
| mc_p (seed 0) | 0.0000 | 0.0030 |

**mc_p — 5 seeds, panier ÉLARGI (joint-null date-level) :**

| seed | mc_p |
|---|---|
| 0 | 0.0030 |
| 1 | 0.0022 |
| 2 | 0.0020 |
| 3 | 0.0024 |
| 4 | 0.0016 |

max(5 seeds) = **0.0030 < 0.05** → **stable, critère joint-null PASSÉ** (marge large — la
seed la plus défavorable reste ~17× sous le seuil).

**OOS-by-date, moitiés calendaires strictes (panier élargi dé-chevauché, 167 événements) :**

| | n | mean_net_bps | Sharpe |
|---|---|---|---|
| Première moitié (in-sample) | 79 | +319.60 | +0.3235 |
| Seconde moitié (OOS) | 88 | +226.10 | +0.2847 |

L'edge OOS reste positif et du même ordre de grandeur que l'in-sample (pas de décroissance
qualitative dans le temps, contrairement au run buggé où l'IS était négatif et l'OOS positif —
signe supplémentaire que le signe négatif du run buggé était un artefact).

**Événements marginaux (dates armées par l'élargi, pas par la référence) :**
- n = 91 (identique au run buggé — la détection des dates marginales ne dépend pas du fix, qui
  n'affecte que le calcul du RENDEMENT de chaque événement).
- **mean_net_bps recalculé (post-fix) = +211.28 bps/évt** (n=91) — contre -59.2 bps dans le run
  buggé. Répartition par année inchangée : 2020=19, 2021=14, 2023=18, 2024=15, 2025=13, 2026=12.

**Bear-windows, segment par segment (panier élargi, post-fix) :**

| Fenêtre | Dates | n événements | mean_net_bps |
|---|---|---|---|
| 2022_full | 2022-01-01 → 2022-12-31 | 0 | n/a (toujours aucun événement z≥2.0 en 2022, cf. §4.3) |
| drawdown25_seg0 | 2019-09-30 → 2020-04-28 | 127 | **+205.39** |
| drawdown25_seg1 | 2021-01-21 → 2021-10-04 | 680 | **+829.91** |
| drawdown25_seg2 | 2021-12-04 → 2024-02-26 (LUNA/FTX + suite) | 1017 | **+169.24** |
| drawdown25_seg3 | 2024-07-05 → 2024-09-09 | 4 | **-591.19** |
| drawdown25_seg4 | 2025-02-28 → 2025-04-11 | 7 | **+779.60** |
| drawdown25_seg5 | 2025-11-16 → 2026-07-02 | 104 | **+49.32** |

**5 des 6 fenêtres bear/drawdown avec des événements réels sont désormais POSITIVES** (seule
`drawdown25_seg3` reste négative, et c'est la fenêtre la plus fine — n=4 événements sur ~2 mois,
donc la plus sensible au bruit d'échantillon). C'est un renversement complet par rapport au run
buggé (qui avait 3/6 négatives, dont deux fortement).

**DSR contre le N réel du registre (1908, post-bump) :**
DSR (panier élargi, 167 événements dé-chevauchés, variance de Sharpe estimée sur les Sharpe
per-asset + per-basket disponibles, n_trials=**1908**) = **0.1475**.

### 8.4 VERDICT MÉCANIQUE (contre les critères d'échec écrits AVANT le calcul, PREREG §"Critère d'échec honnête")

1. **Marginaux ≥ +50 % du panier référence ?** +75.83 % (91 évts marginaux / 120 évts
   référence) → **PASSÉ**.
2. **Joint-null MC date-level < 0.05, stable 5 seeds, sur le panier ÉLARGI complet ?**
   max(5 seeds) = 0.0030 < 0.05 → **PASSÉ**.
3. **OOS-by-date positif ?** Seconde moitié calendaire = +226.10 bps/évt, Sharpe +0.2847 (et
   le `oos_sharpe_40pct` officiel du gate = +0.3162) → **PASSÉ**.
4. **Bear-window check (le panier élargi doit rester positif dans les sous-fenêtres bear comme
   S2 l'est) ?** 5 des 6 fenêtres avec données sont positives ; **`drawdown25_seg3` reste
   négative (-591.19 bps, n=4)** → un seul échec suffit selon la règle du PREREG ("Si le panier
   élargi échoue... le bear-check : la largeur DILUE l'edge — verdict négatif propre") →
   **CE CRITÈRE ÉCHOUE**.

**VERDICT MÉCANIQUE : ÉCHEC** — au sens strict et littéral du PREREG (règle "un seul critère
d'échec suffit", zéro tolérance, zéro pondération). Le fait que la fenêtre en échec ne
contienne que 4 événements sur ~2 mois (2024-07-05 → 2024-09-09) — l'échantillon bear le plus
fin des 6 — est une observation honnête à consigner, mais le PREREG interdit explicitement de
repondérer/retirer un sous-résultat après l'avoir vu ("Interdits : ... retrait post-hoc de
symboles perdants, changement de fenêtre OOS. Tout écart = essai supplémentaire compté au
registre"). Le verdict mécanique reste donc ÉCHEC, malgré une amélioration massive de tous les
autres chiffres par rapport au run buggé.

**Contraste avec le run buggé (§0-7) :** le fix ne change PAS le verdict formel (ÉCHEC dans les
deux cas), mais change complètement le RAISONNEMENT qui y mène. Le run buggé échouait
franchement (edge net négatif en absolu sur les deux paniers, 3/6 fenêtres bear négatives dont
deux à -470/-591bps). Le run corrigé échoue sur un point unique et marginal (1/6 fenêtre, la
plus petite en échantillon, à -591bps sur seulement 4 trades) alors que tout le reste — edge
absolu, joint-null, OOS, 5/6 bear-windows — est fortement positif. C'est la différence entre
« cette piste est morte » et « cette piste est presque validée mais bute sur une règle de
zéro-tolérance appliquée à un échantillon de 4 trades ». Les deux verdicts sont "ÉCHEC" au sens
mécanique du PREREG ; leur substance opérationnelle est très différente.

### 8.5 Traduction opérationnelle (§7 du dossier de restructuration — espérance conditionnelle, pas de chiffres annualisés gonflés)

Méthode identique à `docs/RESTRUCTURATION_2026-07.md` §7 : `rendement_annuel_espéré =
edge_net_bps/évt × sizing_%/trade × événements/an` (non composé — espérance simple, pas de
capitalisation).

**Panier référence (7 symboles) : 17.609 évts/an, edge net = +433.60 bps/évt**

| Sizing | Rendement annuel espéré | $100 | $1 000 | $10 000 |
|---|---|---|---|---|
| 0.5 %/trade | **+0.382 % /an** | +$0.38/an | +$3.82/an | +$38.18/an |
| 2 %/trade | **+1.527 % /an** | +$1.53/an | +$15.27/an | +$152.71/an |

**Panier élargi (12 symboles) : 24.507 évts/an, edge net = +270.33 bps/évt**

| Sizing | Rendement annuel espéré | $100 | $1 000 | $10 000 |
|---|---|---|---|---|
| 0.5 %/trade | **+0.331 % /an** | +$0.33/an | +$3.31/an | +$33.12/an |
| 2 %/trade | **+1.325 % /an** | +$1.33/an | +$13.25/an | +$132.50/an |

**Lecture honnête :** la largeur ajoute bien de la CADENCE (24.5 évts/an vs 17.6, +39 % de
débit brut, +75.8 % marginal net) mais l'edge par événement est plus fin sur le panier élargi
(+270 bps vs +434 bps) — les deux effets se compensent presque exactement en espérance annuelle
(+0.331 %/an élargi vs +0.382 %/an référence à 0.5 % sizing ; quasiment identique à 2 %). Le
panier élargi ne change donc PAS l'ordre de grandeur du revenu espéré — il change la VITESSE À
LAQUELLE l'évidence statistique s'accumule (24.5 évts/an de preuve au lieu de 17.6), ce qui est
la promesse explicite du PREREG ("ce que la largeur achète d'abord, c'est la VITESSE
D'ÉVIDENCE"), pas un gain de rendement.

Ce chiffre reste **très inférieur** au SMIC de référence (~$1 550/mois, cf. dossier
`RESTRUCTURATION_2026-07.md` §7) à tout capital raisonnable pour ce véhicule : au sizing 2 %,
il faudrait un capital de l'ordre de plusieurs centaines de milliers de dollars pour que
l'espérance mensuelle atteigne un SMIC — dans le même ordre de grandeur que le calcul déjà fait
pour S2 seul (~$224k–897k), la largeur ne change pas ce constat.

### 8.6 Conclusion du re-run post-fix

Le bug de look-ahead corrigé dans `fxgate.py` a une conséquence FACTUELLE majeure : les
paniers référence et élargi, mesurés honnêtement, ont un edge net **positif** (+433.6 bps et
+270.3 bps respectivement), pas négatif comme le run buggé le montrait. Le joint-null passe
très confortablement (5 seeds, max p=0.0030), l'OOS-by-date est positif et du même ordre que
l'in-sample, et 5 des 6 fenêtres bear/drawdown documentées sont positives.

Le VERDICT MÉCANIQUE du PREREG reste néanmoins **ÉCHEC**, strictement à cause d'une seule
fenêtre bear (`drawdown25_seg3`, 2024-07-05→2024-09-09, n=4 événements, -591.19 bps) — la règle
de zéro-tolérance du PREREG ne permet aucune exception, même pour un échantillon de 4 trades.
Aucun paramètre n'a été re-tuné, aucun symbole retiré, aucune fenêtre OOS changée après avoir vu
les résultats ; le registre de déflation a été incrémenté AVANT la lecture des résultats
(N crypto 1907→1908 pour ce bump).

**Recommandation à l'opérateur (hors mandat mécanique du PREREG, à traiter séparément) :** vu
l'écart entre "édge massivement positif sur tous les autres critères" et "1 échec sur une
fenêtre de 4 trades", un ré-essai pré-enregistré séparé qui teste spécifiquement la robustesse
du bear-check (par ex. sur un découpage de fenêtres bear différent, décidé AVANT de regarder
les rendements) serait légitime et honnête — mais constituerait un nouvel essai à part entière
au registre, pas une réouverture de celui-ci.

---

## §9 — Test DURCI (PREREG_HARDENED.md, 2026-07-16) : ÉCHEC → PISTE LARGEUR CLOSE

**Verdict à la lettre du PREREG_HARDENED : FAIL, sur DEUX critères indépendants.**
Conformément à la clause écrite avant tout calcul (« FAIL sinon — et la piste largeur est
CLOSE définitivement »), la piste est fermée et mécanisée au `DeadEndsLedger`
(entrée « S2 funding-squeeze BREADTH », 41e), prouvée mordante par mutation.

| Critère (lettre du prereg) | Résultat durci | Verdict |
|---|---|---|
| Événements marginaux ≥ +50 % | **+15.0 %** (25 dates) | **ÉCHEC** |
| Joint-null MC p<0.05 stable 5 seeds | max 0.0072 | passe |
| OOS-by-date positif | +166.05 bps (Sharpe 0.222) | passe |
| Toutes fenêtres bear positives | **seg3 = -622.8 bps (n=4)** | **ÉCHEC** |

**Le biais de survie est MESURÉ — et l'audit avait raison.** C'était la raison d'être du
durcissement : l'univers point-in-time incluant les délistés (4 qualifiés — MKR, BLZ, FTM,
REEF — sur 131 candidats testés) fait tomber les chiffres publiés sur survivants :

| Métrique | Survivants seuls (12) | Durci (12+4) | Δ |
|---|---|---|---|
| edge net / évt | +270.3 bps | +236.6 bps | **−12.5 %** |
| Sharpe panier | 0.304 | 0.273 | −10 % |
| OOS-by-date (40 %) | 0.316 | 0.245 | **−23 %** |
| marginaux vs panier 7 | +75.8 % | +15.0 % | — |
| DSR au N registre (1911) | — | 0.00027 | — |

**Réserve structurelle NON résolue (et non résoluble par cette voie) :** zéro événement z≥2 en
2022 sur l'univers durci — les 4 morts qualifiés n'entrent qu'en 2023-2024 (aucun n'atteint le
seuil vol30j + funding-3-ans avant 2022). Le bear-check reste aveugle au cœur d'un crash : S2
est un phénomène de REBOND, pas de chute. Ce trou ne peut pas être comblé en élargissant
l'univers Binance.

**Vérification (R2, par exécution, pas par lecture de rapport) :** (1) l'instrument est sain —
le gate ORIGINAL ré-exécuté reproduit ses chiffres publiés (91 marginaux, +75.83 %) ; (2) le
gate durci a été ré-exécuté INDÉPENDAMMENT par l'orchestrateur après purge des caches et
suppression du JSON : sortie **bit-à-bit identique** à celle de l'exécutant. Incident
d'instrument à signaler (leçon R2) : une première tentative de vérification utilisait `timeout`
(absent de macOS) — le script n'a jamais tourné et le diff comparait le fichier à sa propre
copie, renvoyant un faux « IDENTIQUE ». L'instrument de vérification doit lui-même être
vérifié : la re-exécution n'a été crue qu'après contrôle du mtime du fichier régénéré.

**Défaut trouvé et corrigé par l'exécutant pendant l'implémentation (déclaré comme tel) :** la
1re version incluait la jambe d'un mort dans la moyenne-panier dès l'existence de ses données
de prix (2020-21) au lieu de sa vraie date d'entrée point-in-time (2023+) — un mort « existait »
dans le panier des années avant d'y entrer. Corrigé par un garde `entry_ts_by_coin` symétrique
au garde look-ahead de `fxgate.py` ; vérifié par (a) retour bit-identique des fenêtres seg0/seg1
(avant toute entrée) et seg5 (après tous les délistings) au panier survivants, (b) mutation on/off
du coût de sortie forcé (30 bps) déplaçant la moyenne dans le sens attendu.

**Découverte d'instrument réutilisable :** `/fapi/v1/klines` sert des barres jusqu'à « aujourd'hui »
pour les symboles délistés, mais **gelées** (volume 0, prix figé) ; `deliveryDate` d'exchangeInfo
est **non fiable** (OMGUSDT : métadonnée 2025-01-31 vs funding réel jusqu'à 2026-06-18, ~16 mois
d'écart). La date de mort réelle se recalcule empiriquement : max(dernière barre à volume non nul,
dernière entrée funding réelle).

**Registre :** +1 essai (`s2_breadth_hardened_prereg`, crypto 1910→1911) bumpé et committé AVANT
lecture des résultats (commit `004529e`). La piste largeur aura coûté 3 essais au total.
**Réveil autorisé :** UNIQUEMENT une largeur d'une autre nature, avec son PROPRE prereg —
cross-venue Bybit (≥2026-11) ou cadence Hyperliquid requalifiée. Jamais un ré-élargissement de
l'univers Binance, jamais un re-découpage de fenêtres, jamais un z abaissé.
