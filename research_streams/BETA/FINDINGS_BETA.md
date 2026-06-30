# FINDINGS — Boucle β, Illiquid-Niche Carry (itération 0)

> Branche `round3/beta-illiquid`. Backtest + shadow size 0 SEULEMENT — **rien live**. Le câblage d'une
> variante gagnante = **porte de merge humaine** (Malachie). Étiquettes : `[VÉRIFIÉ]` lu/exécuté ·
> `[PROBABLE]` inférence solide · `[PRÉSUMÉ]` jugement. Le funding de toute ACCUMULATION vient de
> `*_funding_native` (1 ligne = 1 règlement), jamais du backward-carry `*_on_4h` (piège α du double-compte ~2×).

---

## 0. VERDICT (rangé)

**NÉGATIF PROPRE DE DÉPLOIEMENT — l'axe « petit-livre exploite l'illiquide » est FERMÉ.** L'hypothèse de
migration du carry est **réfutée** : le carry n'a pas migré vers la queue illiquide non-arbitrable ; **la
queue ne porte simplement pas de prime de carry positive**, et le mince brut résiduel est **mangé en net dans
le régime forward, la bande illiquide étant la PIRE** (pas un sweet spot). C'est l'**issue #3 du §0 du mandat
(H1 plat/réfuté → le carry est arbitré/mort partout sur le côté récoltable)**, *renforcée* par le test tueur H2.

| Question | Résultat `[VÉRIFIÉ]` | Verdict |
|---|---|---|
| **H1** : funding-APR ↓ quand liquidité ↓ ? | **SIGNÉ** (récoltable cash-and-carry) : pente forward 2025+ **+0.047 (t=+2.7)**, POSITIVE → la récolte monte AVEC la liquidité (per-an native-mean **positif chaque année**). **\|MAGNITUDE\|** : pente forward **−0.045 (t=−3.1)** → la queue a **2–3× plus** de funding (médiane \|APR\| major 3.3 % / tail 8.6 % / exo 10.9 %) MAIS **négatif-signé** (territoire reverse-carry, borrow-locké §3.4). Absolus forward : majors ~0 %, queue Binance **−6.6 %**, exo HL méd **−0.6 %** / moy **−10.7 %**. | **RÉFUTÉ** (signé) ; magnitude réelle mais **non-récoltable** |
| **H2** : le NET survit-il à la slippage + queue illiquide ? | Forward 2025+ net **POOLÉ NÉGATIF** : −0.6 % (τ=5 %) / −0.2 % (τ=10 %). Bande **illiquide la pire** : net **−2.1 % / −1.1 %** vs mid +1.2 %/+0.5 % vs upper −0.7 %/−0.1 %. Le brut est **déjà ~0 à coût ZÉRO** (+0.7 %/+0.4 %) → négatif **piloté par le brut**, pas par le coût. | **PAS DE SWEET SPOT** (mangé ; illiquide = pire) |
| Sweet spot existe-t-il ? | Aucune bande illiquide n'est net-positive forward après coûts honnêtes. La seule cellule marginalement positive (mid, +1.2 %) est **l'extrémité la PLUS liquide** de la queue, sous toute barre déployable, biaisée survivor, et basculée par 1 marché. | **NON** |
| Reverse-carry (récolter le funding NÉGATIF) ? | Long-perp/short-spot sur la queue à funding négatif = **+15 % brut** forward (séduisant) MAIS = la **prime de borrow** short (no-arb : funding négatif ≡ coût de borrow d'un token hard-to-borrow). Charger le borrow no-arb (λ=1.0) → **−1.9 %** (0 % des marchés positifs). Le lake ne voit pas le borrow → **non-revendicable**. | **FERMÉ** (§3.4) |

**Le négatif est précieux** (garde-fou §0) : il ferme proprement et de façon datée l'axe « avantage petit-livre
sur l'illiquide ». Le carry cash-and-carry est **arbitré à ~0 sur les majors (α) ET absent/négatif sur la queue
illiquide (β)** → la récolte de carry directionnellement-neutre est morte partout sur son côté récoltable.

### Ce qui ROUVRIRAIT l'axe (donnée / structure)
1. **Un régime de funding structurellement plus élevé** (vrai bull persistant) — mais « ça revient » ≠ « déployable
   aujourd'hui » (leçon α). La structure delta-neutre reste saine ; c'est la **prime** qui est absente.
2. **Une donnée hors-survivor-set** : une liste externe d'historique de listing Binance (vendeur tiers) pour
   mesurer le carry des **délistés** — précisément là où il a explosé. Sans elle, tout « positif » sur survivants
   est un plafond optimiste (cf. §4).
3. **Une jambe spot co-localisée pour les exotiques HL** (Binance-spot des exotiques, ou spot HL natif) : la queue
   la PLUS illiquide ($17 k–$130 k/barre) n'est **pas récoltable** en cash-and-carry depuis le lake gelé (pas de
   jambe spot) — elle n'est mesurée qu'en H1. Le cross-venue est aussi **fermé par la donnée** : seul HYPE est à
   la fois sur Binance-tail ET HL-exotic (les exotiques à funding négatif ME/KAITO/POLYX/… ne sont **pas** sur
   Binance) → pas de construction perp-vs-perp pour récolter le funding négatif HL sans shorter le spot.
4. **Le bi-directionnel — TESTÉ et FERMÉ (§3.4)** : un « reverse carry » (long perp / short spot quand funding < 0)
   sur la queue à funding négatif rend **+15 % brut** forward, MAIS c'est la **prime de borrow short** (no-arb :
   le funding négatif EST le coût de borrow d'un token hard-to-borrow). Au borrow no-arb (λ=1.0) → **−1.9 %**, 0 %
   des marchés positifs. **Donnée de borrow observable** (lending rates par token) requise pour rouvrir — le lake
   ne l'a pas → l'edge ne peut être revendiqué honnêtement.

---

## 1. CE QUI A ÉTÉ CONSTRUIT (réutilisé vs neuf)

- **Réutilisé sans reconstruire** : la mécanique carry α (`carry_join` / `carry_harvest`) et l'axe liquidité
  S0-β (`datalake/liquidity.py`). La machinerie carry est **agnostique au symbole** — elle tourne sur la queue
  inchangée.
- **§2 — extension d'une ligne (vues)** `[VÉRIFIÉ]` : ajout de **paramètres de nom de vue** (keyword-only,
  **défaut = vues majors → rétro-compat stricte, 0 régression**) à `cash_and_carry_from_lake`,
  `cross_venue_from_lake`, `_native_funding_raw`, `cash_and_carry_stream`, `cross_venue_stream`. Les conveniences
  pointent désormais sur `binance_tail_*` / `hyperliquid_exotic_*`. (42 tests α carry **inchangés verts**.)
- **Neuf — `src/hermes/research/illiquid_carry.py`** `[VÉRIFIÉ]` : (H1) `assemble_h1_frame` +
  `liquidity_funding_regression` (OLS funding_apr ~ log10($-vol), par-marché/poolé) +
  `infer_funding_interval_hours` (intervalle de funding **dérivé de la donnée** — correction red-team) ;
  (H2) `liquidity_slippage` (le modèle de coût √-impact pré-enregistré) + `gated_net_returns_variable_cost`
  (coût par-barre à la liquidité prévalente). **21 tests TDD**, L0 leak-injection en garde, ruff clean.
- **Drivers** `research_streams/BETA/run_h1.py` / `run_h2.py` / `run_h2_reverse.py` (glue d'analyse sur le lake
  gelé, mirroir des scripts eval α) + `H2_PREREG.md` (pré-enregistrement gelé) + `register_h2_grid.py`.

### La contrainte structurelle LOAD-BEARING (à ne pas sur-interpréter) `[VÉRIFIÉ]`
Les exotiques HL — la queue **la plus illiquide** ($17 k–$130 k/barre, exactement la zone que β visait — sont
**perp-only sur HL** et leur spot n'est **pas dans le lake gelé** → le cash-and-carry n'y est **pas
constructible**. H2 (récolte nette) est donc fondamentalement un test sur les **13 marchés Binance-tail à jambe
spot** ($190 k–$11 M/barre). Les exotiques HL n'informent que **H1** (le niveau de funding vs liquidité à
l'extrême). C'est une **limite de donnée honnête**, pas un choix — et elle **borne H2 à la zone « mid-illiquide »**.

---

## 2. H1 — LA RÉGRESSION carry-vs-liquidité (la mesure) `[VÉRIFIÉ]`

Spectre complet : **23 majors + 18 Binance-tail + 18 exotiques HL** ; variance de liquidité par-barre = **~94 000×**
($17 k POLYX → $1.59 Md BTC). Régression de `funding_apr` (réalisé) sur `log10(dollar_volume)`. **Correction
red-team appliquée** : l'intervalle de funding est **dérivé de la donnée** par symbole (13/18 tail + WIF règlent
en **4h**, pas 8h — l'assumer à 8h sous-estimait leur APR 2×), et le funding est **fenêtre-apparié** à la fenêtre
de liquidité (perp) par symbole (un coin relisté ne paire pas son funding profond à une liquidité courte).

### 2.1 SIGNÉ (récoltable cash-and-carry) vs |MAGNITUDE| — les deux moitiés de l'histoire
- **Pente SIGNÉE** (le carry qu'un cash-and-carry long-spot/short-perp encaisse) : full **+0.0562 (t=+3.46)** ;
  **forward 2025+ +0.0470 (t=+2.72)** ; Binance seul forward +0.0507 (t=+2.38). **POSITIVE et significative** →
  le carry **récoltable** monte AVEC la liquidité, **l'inverse de H1**. → **H1 réfuté.**
- **Pente |MAGNITUDE| `|APR|`** (la quantité récoltable si on était sign-agnostique) : **forward 2025+ −0.0449
  (t=−3.07)** ; la queue porte **2–3× plus** de funding en valeur absolue (médiane |APR| major **3.3 %** / tail
  **8.6 %** / exo **10.9 %**). → β avait **à moitié raison** : la queue a plus d'**activité** de funding ; mais
  cette magnitude est **négative-signée** (territoire reverse-carry), donc **non-récoltable** sans shorter le spot
  — et ce reverse-carry est **borrow-locké** (§3.4).

### 2.2 Robustesse (par venue / par an / outliers) `[VÉRIFIÉ]`
- **Par venue (signé, forward)** : within-major +0.0294 (t=+2.26) ; **within-tail +0.0308 (t=+0.34, NS)** ;
  within-exotic +0.1131 (t=+1.51, NS). Aucune pente intra-venue n'est négative — la queue ne porte nulle part une
  prime de carry positive plus forte que les majors.
- **Par an — native MEAN (carry réalisé, corr 0.997)** : **POSITIF chaque année** (2021 +0.060, 2023 +0.178,
  **2025 +0.044 (t=+2.25)**, **2026 +0.043 (t=+2.40)**). ⚠ **Correction red-team** : une version antérieure
  utilisait la *médiane* du funding-NIVEAU `_on_4h` backward-carrié, qui donnait des pentes 2025/2026 *négatives*
  (−0.011/−0.013, t≈−2.5) — **artefact de clamp modal** (13/18 exotiques HL ont une médiane figée à +0.110 ; la
  médiane ne corrèle que **0.28** au carry réalisé vs **0.997** pour la moyenne). Le seul « soutien » apparent de
  H1 était cet artefact → **éliminé**, la réfutation est nette sur tous les axes.

### 2.3 Les NIVEAUX ABSOLUS forward — le cœur du verdict `[VÉRIFIÉ]`
Dans le régime déployable (2025+), le funding-APR réalisé (signé) est :
- **Majors liquides** ($10 M–$1.6 Md/barre) : **médiane +1.3 %, moyenne ~0 %** (arbitré à zéro — constat α).
- **Queue Binance** ($190 k–$11 M/barre) : **médiane −2.3 %, moyenne −10.8 %** (33 % positifs) — NÉGATIF.
- **Exotiques HL** (le plus illiquide) : **médiane −0.6 %, moyenne −10.7 %, 44 % positifs** (pires : ME −60 %,
  KAITO −60 %, POLYX −24 %).

→ **Il n'y a pas de prime de carry positive (récoltable) dans la queue illiquide.** Un funding négatif signifie
que la jambe short-perp d'un cash-and-carry **PAIE** au lieu d'encaisser. **H1 réfuté (signé).**

---

## 3. H2 — LE TEST TUEUR : le net survit-il à la queue illiquide ? `[VÉRIFIÉ]`

Récolte cash-and-carry (mécanique α, inchangée) sur les **13 marchés Binance-tail à jambe spot**, avec la slippage
**dépendante de la liquidité** facturée à la liquidité **trailing** prévalente (causale, leak-gardée), demi à
l'open + demi au close. Pré-enregistrement gelé : `research_streams/BETA/H2_PREREG.md`.

### 3.1 Le modèle de coût PRÉ-ENREGISTRÉ (√-impact + spread de base) — gelé AVANT de tourner
`round_trip_cost(dv) = clip( taker_floor,  taker_floor + k·√(Q / daily_dv),  cap )`, avec **taker_floor = 35 bps**
(taker α, l'ancre major-liquide), **Q = $50 k** (petit livre = steelman de la thèse petit-livre), **k = 0.025**
(unique, central), **cap = 250 bps**. Garde-fous Malachie respectés : major liquide → ~35 bps ; illiquide paie plus
via √-impact ; plancher = frais taker ; plafond explicite ; **un seul k** (H1 a tranché ; H2 est confirmatoire).
Slippage moyenne armée mesurée : **~76–95 bps** (bande illiquide) → **~45 bps** (upper-tail) `[VÉRIFIÉ]`.

### 3.2 Net par bande de liquidité — l'EXHIBIT (≤0 partout forward, pire en bas)
**Forward 2025+** (régime déployable §4b), net annualisé moyen par bande :

| bande (médiane $-vol/barre) | τ=5 % gross→net | τ=10 % gross→net | net>0 marchés |
|---|---|---|---|
| **illiquide** ($190 k–$420 k) | −0.7 → **−2.1 %** | −0.3 → **−1.1 %** | la pire |
| mid ($490 k–$780 k) | +2.6 → **+1.2 %** | +1.2 → **+0.5 %** | marginale |
| upper-tail ($680 k–$2.8 M) | +0.4 → **−0.7 %** | +0.3 → **−0.1 %** | ~0 |
| **POOLÉ** | +0.7 → **−0.6 %** | +0.4 → **−0.2 %** | **46 % / 15 %** |

La relation est l'**INVERSE du sweet-spot** : plus illiquide = net PIRE (slippage la plus haute, brut le plus
mince). La seule cellule marginalement positive (mid +1.2 %) est l'**extrémité la plus liquide** de la queue,
**sous la barre déployable**, **biaisée survivor**, et **basculée par un seul marché** (ANKR mid = −2.5 % à lui
seul). **Pas de bande déployable.**

**Le négatif est piloté par le BRUT, pas par le coût** `[VÉRIFIÉ]` : à coût **ZÉRO**, le brut forward poolé est
déjà seulement **+0.7 % (τ5) / +0.4 % (τ10)** — économiquement mort *avant* toute slippage. Le modèle de coût
n'est **PAS load-bearing** : le cap 250 bps ne mord jamais (le marché le moins liquide de H2, CVC $190 k/barre,
paie ~88 bps non-cappé), et la slippage est facturée sur la liquidité *trailing* (si tant est, **clémente**).
La conclusion ne dépend donc pas d'un coût « inventé » dur — c'est le brut qui est absent.

### 3.3 La QUEUE (worst-bar, maxDD) — réelle et SOUS-estimée `[VÉRIFIÉ]`
Pires barres net **−3 % à −10 %** ; maxDD jusqu'à **−68 % (LPT, full history)** = blowout de basis ; forward,
maxDD de la bande illiquide jusqu'à **−9 %**. Et ce sont des mesures **close-to-close** → la vraie queue
intra-barre / margin-call est **PIRE** (leçon α : SOL/FTX −28.6 % intra vs −11.9 % close), *a fortiori* sur
l'illiquide (gaps, halts, rug-pulls). La queue de l'illiquide est plus brutale ET le brut plus mince : double peine.

### 3.4 Le REVERSE-CARRY (récolter le funding NÉGATIF de la queue) — testé et FERMÉ `[VÉRIFIÉ]`
Puisque la queue porte surtout du funding **négatif** (§2.1), le steelman évident est le **reverse carry** :
**short spot + long perp** pour encaisser la magnitude du funding négatif. Reproduit (`run_h2_reverse.py`) sur les
13 marchés à jambe spot, gaté |trailing-APR| ≥ 5 %, **slippage √-impact pré-enregistrée** : forward 2025+ =
**+15.0 % net poolé brut-de-borrow** (LPT +77 %, CVC +45 %, NMR +42 %, CYBER +27 % — les noms en death-spiral).
**MAIS** c'est la **prime de borrow short**, pas du carry déployable : par no-arbitrage perp-spot, le funding
négatif d'un token illiquide **EST** son coût de borrow (même demande short). Le funding capté annualise à
**52–94 %** sur ces noms — *exactement* l'ordre du borrow qui le compense. En chargeant le borrow no-arb
(borrow = λ × funding-capté) : **λ=0.5 → +6.5 %, λ=0.7 → +3.2 %, λ=1.0 → −1.9 % (0 % des marchés positifs)**. Le
lake **ne voit pas** le borrow → l'edge **ne peut être revendiqué honnêtement** ; au borrow plein il est négatif.
**Le bi-directionnel est donc FERMÉ**, pas seulement non-testé. (Limite connue : le moteur de coût ne facture pas
un round-trip aux flips de direction intra-armés — immatériel ici, 1 flip ; pertinent seulement si on poursuivait
le reverse-carry, qui est fermé.)

---

## 4. SURVIVORSHIP — le piège #1, et pourquoi il rend le négatif CONSERVATEUR `[VÉRIFIÉ]`

- **Binance tail = survivor-set PUR** : `/fapi/v1/exchangeInfo` ne liste que les contrats TRADING actuels ; les
  délistés sont **non-énumérables**. Toute conclusion H1/H2 est **bornée au survivor-set**. Le net **full-history
  mildement positif (+1.4 %, τ=5 %)** est précisément un **plafond survivor** (il vit dans l'ère 2021-2024, comme
  le carry majors d'α vit dans 2019-2021) ; il **ne survit pas forward**.
- **HL exotique = survivorship OBSERVABLE** : **55/207 délistés (26.6 % brut)** présents dans le snapshot ctx →
  borne basse mesurable du biais. Nuance honnête (red-team) : la liste inclut ~3–5 **coquilles de migration de
  ticker** (MATIC→POL, RNDR→RENDER, FTM→S — l'actif survit sous un nouveau ticker) → la vraie **mortalité
  économique ≈ 24–25 %**. HL zéroise les champs des délistés → leur funding-à-la-mort n'est **pas récupérable**
  (le signal survivorship est le **compte + les noms**, pas leurs niveaux).
- **Conséquence — le négatif est CONSERVATEUR sur tous les axes** `[VÉRIFIÉ red-team]` : (a) survivorship :
  les 18 coins Binance-tail sont tous des survivants continus sélectionnés en 2026-06 ; les délistés-avant-
  sélection sont non-énumérables → le net full-history +1.4 % est un **plafond survivor** (vrai carry ex-ante
  pire) ; (b) les exotiques HL exclus de H2 ont un funding forward **encore plus négatif** (moy −10.7 % vs queue
  −5.7 %) → les inclure rendrait H2 *plus* négatif ; (c) le basis forward est ~0 (+4e−6/barre) → le net est piloté
  par le funding, pas un artefact de basis ; (d) funding correctement natif (somme stream == native, ratio 1.000 ;
  le piège on_4h gonflerait 2–3×). Le survivorship **sur-estime** le carry (délistings = là où il s'est retourné
  en catastrophe) → un net déjà ~0/négatif sur survivants est, en vérité, **encore pire**.

---

## 5. BASELINE & GATE (la barre déployable, héritée d'α)

- **§4(a)** battre la naïve always-on-when-positive : non pertinent — il n'y a pas de brut positif à battre dans la
  queue forward. **§4(b)** net **meaningfully positif en 2025-2026 après TOUS coûts** : **ÉCHEC** (poolé négatif ;
  illiquide la pire). La friction de re-pondération (~0.65 %/an, non chargée, leçon α) rendrait le net **encore
  plus négatif**.
- **Gate machinerie (`validation/book.py`, DSR, PBO, CPCV)** : disponible et câblable, MAIS — comme α l'a établi —
  **le DSR/Sharpe est le MAUVAIS juge** d'une récolte mécanique de funding (DSR=1.0 trivial en unités par-barre,
  artefact ; Sharpe 4-5 = artefact de funding lisse qui écrase la vol et sous-pondère la queue gauche). Le juge
  correct est le **profil forward + la queue** — et il dit NON. Aucune promotion : `evaluate_with_forward`
  échouerait (net forward < 0 → `min_forward_sharpe`/`min_oos_sharpe` non franchis). Rien promu.

---

## 6. PROVENANCE & DISCIPLINE `[VÉRIFIÉ]`

- **Funding causal, pas de double-compte** : accumulation via `binance_tail_funding_native` → `attribute_funding`
  (chaque règlement une fois). `_on_4h` utilisé UNIQUEMENT pour l'axe liquidité/niveau H1 (jamais sommé).
- **L0 leak-injection = garde** : `liquidity_slippage` sur liquidité **trailing backward** est causale
  (`assert_no_lookahead` passe le builder causal ET attrape un builder `shift(-1)`).
- **Format symbole gardé** : tail = slash, exotic = bare ; `assert_joinable` lève sur mismatch (join silencieux 0-ligne).
- **Registre** : grille H2 pré-enregistrée = **3 bandes × 2 τ = 6 variantes déployables** (1 seul modèle de coût)
  → `BookTrialRegistry` crypto **1900 → 1906** (`bump` idempotent-gardé, fichier committé). Robustesse-à-travers-
  bandes (pas best-cell). La régression H1 descriptive **ne consomme PAS** d'essai. Anti-drift test mis à jour (1906).
- **Tests** : **21 TDD** (`tests/research/test_illiquid_carry.py`), ruff clean, **0 régression** (862 passés ;
  hors 2 cassures pré-existantes d'environnement : `fxgate` gitigné [4 modules S6] + `openpyxl` absent [1 test gold]).
- **Frontière** : rien live, aucun câblage exécution/ordre. S3 `funding_carry_harvest` non touché. Câblage = porte
  de merge humaine.

---

## 7. VÉRIFICATION ADVERSARIALE (red-team, 3 agents indépendants)

Trois agents indépendants (lentilles distinctes), chacun avec accès au code ET au lake (read-only) pour recalculer.
**Conclusion unanime : `verdict_robust = true`, AUCUN bug changeant le verdict** (confiance 0.90 / 0.86 / 0.90). Le
négatif survit au steelman. Les corrections trouvées **renforcent** ou **complètent** le négatif :

- **Code-correctness (0.90)** : a trouvé le bug d'intervalle 4h/8h (13/18 tail + WIF règlent en 4h → APR
  sous-estimé 2×). **Corrigé** (intervalle dérivé par symbole) → la pente reste positive, le funding queue devient
  **plus négatif** → **renforce** le négatif. A *vérifié* que H2 est immune (funding via `attribute_funding` natif,
  cadence-agnostique ; reproduit la table forward byte-for-byte) et que le piège du double-compte `_on_4h` est
  bien évité (somme native 0.180 vs on_4h 0.569 = ×3.16 évité).
- **Steelman-deployability (0.86)** : a construit le **reverse-carry** (+15 % brut forward) — le seul angle non
  fermé — puis l'a **lui-même fermé** par le no-arbitrage du borrow (§3.4 ; λ=1.0 → −1.9 %). A signalé la nuance
  **|magnitude|** de H1 (intégrée §2.1) et confirmé que le **cross-venue est data-closed** (seul HYPE sur les deux
  venues). Conclusion : *« even the steelman's best honest forward net-APR is ~0 to slightly negative once
  short-borrow is charged… the clean deployment-negative verdict survives. »*
- **Methodology + survivorship (0.90)** : a identifié que les pentes par-an *négatives* (le seul soutien apparent
  de H1) étaient un **artefact de clamp modal** de la médiane `_on_4h` (corr 0.28 vs 0.997 pour la moyenne) →
  **corrigé** (per-an sur native MEAN → positif partout). A confirmé le verdict **conservateur sur tous les axes**
  (§4) et que le **coût n'est pas load-bearing** (brut ~0 à coût zéro, §3.2). A noté la mortalité HL ~24–25 % nette
  (intégré §4) et le window-mismatch des relistés (**corrigé** par fenêtre-appariement §2).

**Bilan** : le red-team a transformé un négatif déjà propre en un négatif **complet et durci** — H1 réfuté sur tous
les axes (l'artefact de soutien éliminé), H2 négatif piloté par le brut (coût non load-bearing), et le
bi-directionnel **fermé** par le no-arb borrow plutôt que laissé non-testé. Aucune des trois lentilles n'a trouvé
de configuration honnête déployable.
