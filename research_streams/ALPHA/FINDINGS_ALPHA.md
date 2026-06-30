# FINDINGS — Boucle α, Carry-Harvest (itération 0)

> Branche `round3/alpha-carry` · backtest + shadow size-0 ONLY, rien live · porte de merge humaine (Malachie).
> Étiquettes : `[VÉRIFIÉ]` (requêté/testé directement), `[PROBABLE]` (inféré de mesures solides), `[PRÉSUMÉ]` (jugement).
> **Garde-fou §0 respecté : ce livrable est un NÉGATIF PROPRE de déploiement — le résultat le plus sain du labo.**

---

## 0. VERDICT (rangé)

**Aucune variante (CC ni XV) n'est robustement déployable sur le régime forward-pertinent (2025–2026).** `[VÉRIFIÉ]`

| Rang | Variante | Net forward (2025 / 2026-H1, @20 bps) | Déployable ? |
|---|---|---|---|
| 1 | **CC — always-on-when-positive** (naïve, §4a) | **+2.87 % / +0.45 %** | **NON** — vit du bull 2021/2024 ; ~0 aujourd'hui |
| 2 | CC — τ=5 % mensuel | +3.42 % / −0.21 % | NON — le seul à battre la naïve en 2025, perd en 2026 |
| 3 | XV — \|apr_diff\|≥10 % hebdo | +1.77 % / +1.02 % | NON — minuscule, données fines (2024-03+), meurt au coût taker |
| — | CC — τ=10/15 % | +0.5–0.7 % / 0.00 % | NON — le seuil haut NUIT (sélection adverse sur funding mean-reverting) |
| — | CC/XV buy&hold (sans règle) | +1.25 % / −0.71 % (CC) · −4.5 %/an (XV) | NON |

**La structure est saine, l'edge a été arbitré.** Le reframe delta-neutre TIENT `[VÉRIFIÉ]` : la position long-spot/short-perp
n'a **pas** de bull-beta (maxDD livre ~−4.6 % sur 7 ans vs ~−80 % directionnel ; corr spot/perp ~0.999, terme de base
≈0 en moyenne). **C'est bien la première structure du labo qui ne meurt pas de bull-beta.** Mais le **net récoltable**
a compressé sous le seuil déployable : headline ~12 %/an = **artefact 2021 (+38 %) + 2024 (+13 %)** ; le forward
(2025 +0.5–3 %, **2026-H1 ≈ 0 %**) ne passe pas §4(b) « net MEANINGFULLY positif en 2025–2026 ».

### Ce qui ROUVRIRAIT l'axe (données/régime — §6)
1. **Un régime de funding élevé revient** (bull) → mais « ça revient » n'est pas « déployable aujourd'hui ».
2. **La couche de contrôle d'itération 1** (le croisement promis §7) — mais iter-0 + red-team la dégradent :
   (a) **filtre de sélection de coins = ~ZÉRO valeur forward** `[VÉRIFIÉ red-team]`. Le LOO montre des coins à carry
   net-NÉGATIF qui diluent le livre (SOL −0.53, TRX −0.35, BNB −0.23, BTC −0.19 pp d'ann%), MAIS les exclure est un
   **nettoyage d'ère bull, pas forward** : oracle look-ahead (drop-6) lève le headline 12.66→14.55 % mais **2026 −0.09 pp
   (PIRE)**, 2025 +0.41 pp ; aucune version *causale* ne bat la naïve (2025-26 +2.63 % vs +2.68 %). → à rétrograder de
   « encourageant » à « pas de lift forward ». (b) **timing d'entrée** `funding_squeeze` (non-testé iter-0). (c) **moteur
   de régime** S6/S8 pour sizer-down avant les flips (non-testé). La règle on/off dodge déjà les négatifs (+1.2 pp vs
   buy&hold en 2026) — mais c'est *l'essentiel* du levier déployable, et il plafonne à ~0 forward.
3. **XV** : profondeur HL klines pré-2024 (collecte forward / source payante) + positionnement **bi-directionnel**
   (iter-0 ne récolte qu'un sens, short-Binance/long-HL).

---

## 1. CE QUI A ÉTÉ CONSTRUIT (réutilisé vs neuf)

- **Réutilisé (non reconstruit)** : `carry_join` (alignement + identité `delta_neutral_pnl` + `assert_no_lookahead`),
  les stats (`sharpe_ratio`, `deflated_sharpe_ratio`, `max_drawdown`), `BookTrialRegistry`. Le backtester
  `validation/backtest.py` (single-OHLCV, prix-only, **pas** d'accumulation de funding) n'est **PAS touché** ; la cible
  de déploiement S3 `paper.py` (short-perp+long-spot, accrue/8h, 5 bps maker × 4 fills) n'est **PAS touchée** (miroir seulement).
- **Neuf, TDD (rouge d'abord)** : `src/hermes/research/carry_harvest.py` (+ 18 tests `tests/research/test_carry_harvest.py`,
  ruff clean, **0 régression** — 807 passed ; 1 échec + 4 erreurs PRÉ-EXISTANTS et hors-sujet : dép. `openpyxl` manquante
  sur un test gold xlsx ; `import fxgate` (research_fx path) sur 4 tests S6). Un **accumulateur de règlements de funding**,
  PAS un moteur de rendement-prix.

### Découverte de correction LOAD-BEARING (la donnée qui aurait fabriqué un faux gagnant) `[VÉRIFIÉ]`
`binance_funding_on_4h.funding_rate` **reporte en arrière** (backward-carry) le taux réglé 8h sur un nombre **variable**
de barres 4h (l'échantillon BTC montre un taux porté sur 3 barres), et **~43 %** des règlements natifs n'ont pas de
ts-match (3223/7453). Donc **« sommer par barre » double-compte (~2×), « diviser par 2 » est faux (longueur variable),
« keep-on-change » perd les règlements de taux égal**. → **Le funding vient de `binance_funding_native`** (1 ligne = 1 règlement
réel = exactement ce qu'accrue S3 paper). Test #2 PROUVE `sum(attribué) == sum(natif)` et l'ancre réalité :
funding accumulé BTC-2024 = **+11.96 %**, identique à la mesure indépendante. Pas de double-compte, pas de perte. Aucun leak forward (`match_neither=0`).

---

## 2. PROFIL NET-CARRY — CC (cash-and-carry, Binance spot+perp, univers=livre 23 coins)

**Grille pré-enregistrée (8 cellules, registre crypto 1892→1900, source `alpha_carry_harvest_prereg_grid`)** `[VÉRIFIÉ]`
CC : τ∈{5,10,15} % APR × cadence∈{hebdo(42 barres), mensuel(180)} = 6 ; XV : \|apr_diff\|∈{10,20} % = 2.
Référence (non comptée) : always-on-when-positive + buy&hold.

### 2.1 Net annualisé full-sample + queue, par modèle de coût `[VÉRIFIÉ]`
Coût round-trip = open(spot+perp) + close(spot+perp). Maker 11 bps / S3-paper 20 bps / taker+slippage 35 bps.

| Variante | ann% (11/20/35 bps) | Sharpe* | util% | maxDD% | pire barre% |
|---|---|---|---|---|---|
| always-on-pos | 12.66 / 12.31 / 11.74 | 4.2 | 99 | −4.6 | −2.9 |
| τ5 hebdo | 13.27 / 12.54 / 11.33 | 4.2 | 96 | −2.9 | −2.9 |
| τ5 mensuel | 12.70 / 12.49 / 12.13 | 5.1 | 91 | −1.9 | −1.4 |
| τ10 hebdo | 12.51 / 11.85 / 10.75 | 4.1 | 63 | −3.0 | −2.9 |
| τ10 mensuel | 11.41 / 11.22 / 10.90 | 4.9 | 54 | −2.0 | −1.5 |
| τ15 hebdo | 11.06 / 10.67 / 10.04 | 4.4 | 33 | −2.3 | −2.0 |
| τ15 mensuel | 10.89 / 10.73 / 10.45 | 5.0 | 39 | −2.0 | −1.3 |

\* **Le Sharpe ~4–5 est un ARTEFACT de funding lisse, PAS une mesure de risque** — sanity-check seulement (mandate §3A).
Une série dominée par du funding positif lisse a une vol minuscule → Sharpe explose, mais sous-pèse la queue gauche
(blowout de base, flip de funding). **DSR = mauvais instrument ici** (la significativité d'une récolte mécanique n'a
jamais été la question) ; sa valeur est sensible aux unités — en unités cohérentes per-barre **DSR=1.0** (le carry
« passe » trivialement), en unités mixtes 0.0 (artefact, corrigé dans le driver post-red-team). Le point juste : le
Sharpe ~4–5 est dénué de sens — le funding lisse écrase la vol au dénominateur en sous-pesant la queue gauche. Le juge
reste le **profil forward + queue**, pas le DSR.

### 2.2 Net PAR ANNÉE @20 bps — le cœur forward-pertinent `[VÉRIFIÉ]`

| Variante | 2020 | 2021 | 2022 | 2023 | **2024** | **2025** | **2026-H1** |
|---|---|---|---|---|---|---|---|
| always-on-pos | 21.3 | **38.1** | −1.4 | 7.3 | **13.1** | **2.9** | **+0.45** |
| τ5 mensuel | 19.1 | 38.1 | +2.3 | 7.6 | 13.1 | **3.4** | −0.21 |
| τ10 mensuel | 19.6 | 37.6 | +2.1 | 4.0 | 12.1 | 0.5 | 0.00 |
| τ15 hebdo | 22.9 | 36.4 | 0.0 | 3.8 | 9.4 | 0.0 | 0.00 |
| buy&hold | 16.8 | 36.7 | −4.4 | 4.0 | 11.9 | 1.3 | **−0.71** |

→ **Monotone compression.** Headline porté par 2021/2024. 2025 = 0.5–3.4 %. La règle on/off *fonctionne* (évite le
saignement buy&hold) mais le net forward reste sous le seuil déployable.

> **Nota (red-team) — ces chiffres par-année sont des SOMMES de période brutes, pas annualisées.** 2026-H1 (~0.49 an,
> 1078 barres) s'annualise à ~2× l'affiché : always-on 2026-H1 brut 0.76 % → **run-rate annualisé ~1.55 % APR** (maker ;
> trailing 6–18 mo ~1.5–2.7 %). Donc « 2026 ≈ 0 % » est le ½-an brut ; le **run-rate forward honnête est ~1.5–2.7 % APR**
> — toujours **sous le seuil déployable** après coûts honnêtes + frictions non-modélisées (cf. §3 Axe B, §7). **Attribution
> par ère** (always-on, maker, cum 86.2 %) : **2019–2021 = 72 % du cumul**, 2024 = 15 %, **forward 2025–2026H1 = 5 % du
> cumul (~4 pp / 1.5 an ≈ 2.7 % APR)**. Le headline EST un artefact bull, chiffré. `[VÉRIFIÉ red-team rerun]`

---

## 3. DÉCOMPOSITION DEUX AXES

### Axe A — OVERFITTING (le risque familier) `[VÉRIFIÉ]`
- **Plateau** : les 6 cellules CC sont cohérentes en headline (Sharpe 4.05–5.06, spread 1.0 ; ann% 10.7–13.3). Pas de
  spike isolé → **pas de best-cell overfit sur le headline**. MAIS le plateau forward est uniformément **compressé à ~0**
  (toutes les cellules à ~0 en 2026), donc le plateau ne sauve pas la déployabilité.
- **DSR** : mauvais instrument (cf. §2.1) — sanity-check, pas le juge ; valeur sensible aux unités (per-barre 1.0 / mixte 0.0).
- **Sous-périodes** : NON stable — concentré 2020–2021/2024 ; 2022/2025/2026 plats-à-négatifs. **Attribution : 72 % du cumul
  vient de 2019–2021, 5 % du forward 2025–2026H1** `[VÉRIFIÉ red-team]`.
- **LOO par coin** (always-on @20 bps, ann% base 12.31) : contributions petites/distribuées (max ETH +0.26 pp) → **pas
  porté par 1–2 coins** (sain). 4 coins à contribution NÉGATIVE (SOL −0.53, TRX −0.35, BNB −0.23, BTC −0.19). MAIS
  **les exclure ne sauve PAS le forward** (oracle look-ahead drop-6 : 2026 −0.09 pp, aucune version causale ne bat la naïve)
  — la dilution est un phénomène d'ère bull, pas forward. Sélection ⇒ **PAS un levier déployable** (cf. §0).
- **Seuil ne bat pas robustement la naïve (§4a)** : en 2026 la naïve (+0.45) bat toutes les cellules à seuil ; le seuil
  haut NUIT (sélection adverse — le funding **mean-reverte**, arming sur funding récent élevé attrape le pic avant reversion ;
  écho de l'alpha-mort S2.2A).

### Axe B — STRUCTUREL (le nouveau, décisif) `[VÉRIFIÉ]`
- **Net par RÉGIME DE FUNDING** (mean per-bar bps, @20 bps) : always-on `mean_pos=+0.82`, `mean_neg=−0.97` bps ;
  τ15 hebdo `mean_pos=+1.71`, **`mean_neg=−4.21`** bps. → quand le seuil haut arme puis le funding flippe négatif, le
  saignement est sévère. **fraction de barres à funding positif** chute avec le seuil (always-on 0.53 → τ15 0.18) :
  chasser le funding élevé = sélection adverse.
- **Régime forward 2026-H1** `[VÉRIFIÉ]` : **13/23 coins à carry NÉGATIF** (INJ −9 %, ATOM −6 %, APT/WIF/DOT/TRX ~−4.5 %) ;
  meilleur +1.8 % ; médiane −0.29 %. Le funding est **majoritairement négatif** dans le forward → la jambe short-perp PAIE.
- **Risque de QUEUE — basis blowout RÉEL** `[VÉRIFIÉ]` : pires barres 4h par coin = vrais évènements —
  **SOL −11.9 % (FTX 2022-11-10)**, FIL −11.2 %, DOGE −10.1 %, et un **cluster CORRÉLÉ 2023-03-24** (depeg USDC :
  ARB/APT/ETH/LTC/OP/DOT tous −1.5 à −5 % la même barre). p01 par barre −7 à −27 bps. Diversifié à ~−4.6 % maxDD livre,
  MAIS les jours de blowout corrélés (FTX, depeg) frappent plusieurs coins ensemble. **La jambe « delta-neutre » n'est pas
  sans risque** — un déploiement DOIT sizer pour ça. **Queue SOUS-ESTIMÉE** `[VÉRIFIÉ red-team]` : mesurée close-to-close,
  mais une position margée subit l'intra-barre (SOL FTX **−28.6 % intra** vs −19.9 % close) → l'appel de marge / la queue
  de sizing sont PIRES que le maxDD −4.6 % affiché. Le tail n'est pas un artefact de pondération : le livre equal-NOTIONAL
  1/23 donne maxDD −5.79 % (similaire).
- **Frictions NON modélisées (rendent le négatif PLUS robuste)** `[VÉRIFIÉ red-team]` : (1) le livre equal-weight-among-armed
  reweight les survivants à chaque changement de `n_armed` — **~0.65 %/an de turnover de re-pondération non chargé** (push
  2025/2026 encore plus vers 0). (2) Pas de coût de funding/borrow manquant : long-spot (no borrow) + short-perp ; les
  épisodes de funding négatif (le short PAIE) passent déjà dans la série `funding` (cf. `mean_neg_bps`). Annualisation
  additive = conservatrice vs composée.
- **CAPACITÉ / forward-relevance** (le check décisif §3C) : **le net 2026 n'est PAS positif après coûts honnêtes**
  (~0 % gated, négatif buy&hold ; 13/23 coins négatifs). → arbitré à ~0 → **axe fermé pour déploiement aujourd'hui**.

---

## 4. PROFIL XV (cross-venue : short Binance perp / long HL perp, harvest apr_diff) `[VÉRIFIÉ]`

Fenêtre limitée par profondeur HL klines : **2024-03-18 → 2026-06-29 (~2.25 ans)** — tout le forward compressé.

| Variante | ann% (11/20/35 bps) | Sharpe | util% | maxDD% | 2024 / 2025 / 2026 @20bps |
|---|---|---|---|---|---|
| XV buy&hold | −4.47 / −4.52 / −4.60 | −2.6 | 99 | −11.2 | −6.0 / −4.0 / −0.3 |
| **XV \|d\|≥10 % hebdo** | 2.31 / **1.58** / 0.36 | 0.6 | 41 | −1.1 | **+0.82 / +1.77 / +1.02** |
| XV \|d\|≥20 % hebdo | 0.61 / 0.38 / −0.02 | 0.3 | 12 | −0.8 | −0.03 / +1.34 / −0.44 |

→ Le **buy&hold XV est NÉGATIF** (−4.5 %/an : le funding HL > funding Binance en moyenne + drague de base inter-venue,
maxDD −11 %). La cellule gated `\|d\|≥10 %` est **marginalement positive (~1.6 %/an, et +1.0 % en 2026** — le seul truc
positif dans le forward), MAIS : Sharpe <1, 2.25 ans seulement, **meurt au coût taker (+0.36 %)**, uni-directionnel, **et
c'est un ARTEFACT MONO-COIN** `[VÉRIFIÉ red-team]` : le LOO 2026 montre **OP seul = +0.95 pp des +1.44 pp** bruts (ADA +0.26,
ATOM +0.20, le reste ~0 ou négatif INJ/ARB) → drop OP ⇒ 2026 s'effondre à ~+0.49 %. **Pas un edge mécanique diversifié.
Pas déployable.** NB : l'alpha-mort S2.3 (dispersion funding cross-venue, Sharpe −0.44) concernait un signal *prédictif* ;
XV ici est une récolte *mécanique* — distinct, mais le drapeau jaune (cross-venue funding ténu) se confirme.

---

## 5. BASELINE HONNÊTE (§4)
- **(a) battre la naïve always-on-when-positive** : ÉCHEC robuste — la naïve est ~aussi bonne que tout dans le forward
  (2025 +2.9, 2026 +0.45) ; le raffinement par seuil ne la bat pas robustement (le bat en 2025 mensuel-τ5, perd en 2026).
- **(b) net MEANINGFULLY positif en 2025–2026 après TOUS coûts** : ÉCHEC — 2025 ~0.5–3 %, 2026 ~0 %. Ne vit qu'en 2021/2024.

---

## 6. PROVENANCE & DISCIPLINE
- **Causal/PIT** : funding réglé natif (lag 0, no forward leak prouvé `match_neither=0`) ; signaux trailing backward-window
  (L0 `assert_no_lookahead` test #1) ; décision d'arming à la frontière de cadence gouverne le bloc SUIVANT (lag 1).
- **Coûts dans le backtest** (open+close, 2 jambes), 3 modèles ; funding/coût jamais par-barre erroné (test : 1 round-trip/épisode).
- **Lake** `data/hermes_datalake.duckdb` (read-only, gelé). Univers 23 coins `[VÉRIFIÉ]` ; intervalles funding 22×8h + WIF 4h.
- **Pré-enregistrement** : grille gelée AVANT run, comptée +8 au `BookTrialRegistry` crypto (1900). Test anti-drift mis à jour
  (`test_committed_canonical_count_is_1900`).
- **Frontière** : rien live ; S3/paper/hyperliquid non touchés ; câblage = porte humaine.

**Garde-fou §0 : NÉGATIF PROPRE = SUCCÈS.** On n'a pas fabriqué de gagnant ; on n'a pas cherché la moins-mauvaise cellule.
La structure (delta-neutre, sans bull-beta) est validée et la machinerie livrée ; l'edge net forward est ~0 → axe à la marge
de déployabilité, rouvrable par la couche de contrôle (iter-1) ou un régime de funding plus élevé.

---

## 7. VÉRIFICATION ADVERSARIALE (red-team, 3 agents indépendants)

Le verdict a été red-teamé par 3 agents indépendants (audit code-correctness, steelman-déployabilité, critique méthodo).
**Les 3 concluent `negative_is_robust`** (confiance 0.88 / 0.82 / audit-clean). Aucun n'a trouvé de bug changeant le verdict.

- **Code-correctness (indépendant)** : les 5 invariants vérifiés corrects par trace — (1) `attribute_funding` lossless
  (`searchsorted(side=right)-1` = sémantique `[t,t+1)`, no double-count ; le funding ne passe JAMAIS par `funding_on_4h`
  forward-carry, seulement par `native`) ; (2) `arm_schedule` causal lag-1 ; (3) coût 1×open+1×close (même le cas 1-barre) ;
  (4) `build_book` union-grid exact, pré-listing plat ; (5) signe XV correct. « Le négatif n'est l'artefact d'aucune mécanique. »
- **Steelman (tente de RÉFUTER le négatif)** : même en empilant maker-only + sélection oracle look-ahead + best-gate
  cherry-pické SUR le forward, le plafond 2025-26 = **+3.97 % APR** (3 couches de biais) et ne bat jamais la naïve en
  causal. Le négatif tient.
- **Critique méthodo** : 3 corrections appliquées ci-dessus — (a) bug d'unités DSR (corrigé driver + §2.1) ; (b) turnover
  de re-pondération ~0.65 %/an non chargé (négatif PLUS robuste) ; (c) queue close-to-close sous-estimée vs intra-barre
  (négatif PLUS robuste). Confirme : la fenêtre forward 2025-26 est le bon juge (pas un cherry-pick — la compression du
  funding est un fait de marché exogène : gross livre 2021 +624 % → 2024 +282 % → 2025 +66 % → 2026-H1 +8 %).

**Bilan red-team : le verdict est correct en direction ET magnitude. Les corrections RENFORCENT le négatif. Ne rien déployer.**
