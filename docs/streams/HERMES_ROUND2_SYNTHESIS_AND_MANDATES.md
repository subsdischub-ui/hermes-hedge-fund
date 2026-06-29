# HERMÈS — ROUND 2 : SYNTHÈSE CROISÉE DES 10 STREAMS + MANDATS v-NEXT
### Ce qui n'apparaît PAS stream par stream — seulement en les croisant. Daté 2026-06-27.

---

## 0. LE MÉTA-VERDICT (à lire avant les mandats)

Le programme est, à ce stade, un **négatif honnête quasi-total** — et **c'est un vrai résultat**, pas un
échec : il a **épuisé proprement la frontière directionnelle/mono-sleeve en données gratuites** et prouvé que
le **même mécanisme de mort (bull-beta déguisé) est universel** (crypto, or, FX, ton CB). Un labo qui sait ça
est beaucoup plus intelligent qu'un labo avec un faux positif.

Trois choses, et trois seulement, empêchent de conclure « on n'a rien trouvé » :
1. **Un edge réel mais décayé** : `fx_aud_cad` (convergence AUD/CAD), MC-null battu, sans look-ahead (attaqué 5×)
   — mais solo, décayé ~5× depuis 2016 (un trade du supercycle commodities 2008-16).
2. **Plusieurs signaux « bon signe mais sous-puissants »** : S6 CB-demand-**accélération** (bon signe, OOS+,
   p=0.14 à N≈60), S6 spread reported-unreported (IC+0.21, p=0.067), S7 ton-comme-**identifieur-de-régime**
   (label-shuffle p=0.013), `funding_squeeze` (modeste, LOO +0.015).
3. **Le test fondateur n'a JAMAIS été fait** : la thèse Renaissance est que *l'edge est la COMBINAISON* de
   nombreux paris faibles décorrélés — or la structure parallèle a fait que **chaque stream a tué son propre
   sleeve EN SOLO**. Les survivants à-bon-signe n'ont jamais été **assemblés en un seul livre combiné gaté au
   niveau de la combinaison**. *(Honnêteté : ce test est probablement négatif lui aussi — ces signaux sont très
   marginaux. Mais c'est le test qu'on DOIT à la thèse, il est peu coûteux, et son résultat — positif OU négatif
   propre — est le vrai end-state du programme.)*

---

## 1. LES 7 ANGLES MORTS (croisés) — par valeur décroissante

> Chacun indique **quels streams il croise** et **pourquoi il était invisible singulièrement**.

### AM-1 — Le livre combiné n'a jamais été testé (la thèse Renaissance elle-même)
**Croise : TOUS les streams + la prémisse fondatrice.** Chaque stream a gaté son sleeve **par-sleeve** et l'a
tué. Mais l'edge Medallion est la *combinaison* de paris faibles, **aucun individuellement significatif**. Les
survivants à-bon-signe-OOS (fx_aud_cad, CB-accel, dollar-decoupling, funding_squeeze, le cross-sectionnel si
AM-3 livre, le ton-comme-conditionneur) **n'ont jamais été poolés** en un livre et gatés **au niveau de la
combinaison** (la combinaison = **UN** test pré-enregistré, battant un null combiné, au vrai trial count).
**Invisible singulièrement** parce que le parallélisme = chaque stream n'a vu que son propre sleeve. La
machinerie existe (S1 `combine_book`, S9 cross-book) ; ce qui manque, c'est **la composition des nouveaux
survivants**. → **C'est le joyau du merge.** Danger à cadrer brutalement (S10) : c'est *aussi* la machine à
overfit (« combiner le bruit ») — d'où la discipline mécanisme-a-priori + bon-signe-OOS + orthogonal +
poids non-optimisés + la combinaison comptée comme un essai + battre un null combiné, sinon **négatif propre**.

### AM-2 — Le gap data n°1 (4h) est GRATUIT et n'appartient à personne
**Croise : S2 + S3 + S9** (les trois citent « funding/prix **4h** profond multi-cycles » comme LA donnée
manquante qui bloque le livre crypto cœur). **Aucun stream ne possède son acquisition.** **Vérifié (juin 2026) :**
les klines Binance **4h** sont **gratuites et profondes** (depuis le listing) ; l'historique de funding natif
Binance est **gratuit et profond** (`/fapi/v1/fundingRate`, depuis l'inception ; intervalle 8h, et **4h pour de
nombreuses paires depuis 2023-10-12**, géré via `/fapi/v1/fundingInfo`). Le lake n'a pris que 6h **uniquement
pour matcher la fenêtre 360j de CoinGlass** — il a **laissé la donnée 4h Binance gratuite sur la table**. Un
re-freeze Stream 0 ajoutant 4h klines + funding natif profond peut **ressusciter le test cross-régime de S2 et
la généralisation de S3 — pour $0**. **Invisible singulièrement** parce que chaque stream a noté « manque 4h »
comme *sa* limite, sans voir que c'est **la même** limite partagée et qu'elle est **closable gratuitement**.

### AM-3 — Le cross-sectionnel crypto market-neutral n'a jamais été testé
**Croise : S2 + S3 + la leçon de S5.** Tout est mort de bull-beta parce que **chaque sleeve crypto était
long-only/directionnel** (`funding_squeeze` s'arme au-dessus de la MA200 = long-en-bull). S5 a **prouvé** que la
seule façon de tuer la bull-beta est la **construction relative-value/market-neutral**. Le signal funding exprimé
en **cross-sectionnel long-short** (long les perps à plus haut funding_z, short les plus bas, **dollar-neutre**)
**retire le beta marché par construction** — et **n'a jamais été testé**. *(Vérifié : c'est une construction
réelle et distincte du cash-and-carry que `funding_carry_harvest` couvre déjà — ici on price la dispersion
cross-sectionnelle du funding, pas la collecte du paiement.)* C'est **le sleeve crypto inexploré le plus
prometteur**. **Invisible singulièrement** parce que S2 possède le signal, S3 possède l'univers, et **personne ne
possède la jonction** (le signal de S2 exprimé sur l'univers de S3, en neutre). *(Caveat coût : 2 jambes à 15bps
→ doit être lent — rebalance sur settlement funding — et gaté au coût 2-jambes honnête.)*

### AM-4 — Une contagion méthodologique qui a pu fabriquer des FAUX résultats ailleurs
**Croise : S6 + S8 + S10 + S2.** Des bugs/erreurs de calibration ont été attrapés dans **presque tous** les
streams, **quasi toujours dans le glue/composition NON-testé**, et ils **fabriquent quasi toujours un POSITIF** :
S2 (look-ahead `dom` non-lagué → +1.76% fantôme), S6 (le null IC circular-shift est **anti-conservateur sur
inputs en niveau, 16% FPR à N=60** → tous les IC en niveau VOID), S8 (stabilité brute 0.88 = le hasard 0.89 →
chance-correction). **Le PATTERN (les bugs de composition fabriquent des positifs) n'émerge qu'en croisant.**
→ S10 doit : (a) **ériger en règles gouvernance** les nulls calibrés, la chance-correction, le vrai trial-count
par-livre, l'orthogonalité **sur semaines co-actives** (S5) ; (b) **imposer un test d'injection-de-fuite sur
chaque DRIVER** avant tout verdict, et **le re-rejouer sur chaque SURVIVANT au merge** (les survivants sont
exactement là où une fuite fabriquant-un-positif est la plus dangereuse) ; (c) **vérifier que chaque stream a
utilisé la vue funding PROFONDE** (`coinglass_funding_rate`), pas la `binance_funding` forward-only (un stream
qui s'est trompé de vue a rapporté un **négatif fantôme**).

### AM-5 — Une classe entière de frontières « non-backfillables mais accruables », sans infra unifiée
**Croise : S4 + S5 + S7.** Des crons de **collecte forward** sont suggérés à 3+ endroits (S4 skew = FAIT, S5
courbe COMEX = suggéré, S7 consensus macro = suggéré) mais **éparpillés**. Une **infra de collecte forward
unifiée** démarrée **maintenant** (un cron, plusieurs flux) ouvre ces frontières en **12-18 mois**. **Le reframe
stratégique** : les backtests sont contaminés par la bull-beta, et le **régime ACTUEL (bear/range 2026) est le
labo naturel pour falsifier la bull-beta** → le livrable doit **basculer** de « trouver un edge en backtest »
vers **« accumuler de l'OOS forward shadow dans le régime actuel + bâtir l'infra de collecte »**.

### AM-6 — L'économie de densité-d'edge diffère par livre → incline la chasse
**Croise : S5 + S10.** Le coût FX bas (1.6-7bps) laisse survivre des edges faibles que le coût crypto (15-20bps)
tue — le « mur d'overfitting remplace le mur de coût » de S10. Donc **le livre des NOMBREUX paris faibles
appartient aux venues à bas coût (FX/futures)**, tandis que **le crypto doit se concentrer sur les RARES edges
forts** (funding_squeeze) **+ le micro à capacité limitée**. Reframe de *où* chasser quoi.

### AM-7 — Le micro à capacité limitée n'a jamais été testé (le programme contre son propre edge)
**Croise : la prémisse fondatrice « capacité » + toutes les morts de signaux-macro-que-les-institutions-arbitrent.**
Chaque signal testé (funding, CB-demand, ton, courbe) est un signal que **les institutions tradent déjà** → déjà
arbitré. L'edge **déclaré** du compte est la **capacité** (inefficiences trop petites pour les gros). Personne
n'a testé le **micro à capacité limitée** (timing du settlement funding, petit basis cross-venue, micro-réversion
de dépeg stablecoin). *(Caveat honnête : 15bps crypto tue le HFT → niche étroite « basse-fréquence-mais-
capacité-limitée », longues cotes. Priorité plus basse, mais vrai trou stratégique.)*

**+ Décision Amberdata (désormais informable, croise S4 + S10) :** DVOL-VRP mort + tout le prior surface options
mort + le skew serait probablement un **re-emballage RSKEW** (fear/dip-beta, déjà mort) → **NE PAS acheter
Amberdata** ; laisser le forward-log gratuit s'accumuler ; revisiter dans 6-12 mois. **$ économisés.**

**+ Correctif d'honnêteté URGENT (S10) :** `newdata.py:12` + `library.py:226-241` annoncent des sleeves
**rétrogradés comme confirmés** avec des chiffres ~15× gonflés. Tout est shadow size-0 (zéro capital en risque
*pour l'instant*), mais **à corriger au merge avant toute promotion live**.

---

## 2. CARTE D'ACTION PRIORISÉE (avec dépendances)

| P | Action | Stream/session | Débloque | Dépend de |
|---|--------|----------------|----------|-----------|
| **P0** | Re-freeze lake **4h klines + funding natif profond** | Stream 0 (re-freeze) | AM-2 → S2/S3 | — (à merger en premier) |
| **P0** | **Correctifs honnêteté** live-labels + règles gouvernance + décision Amberdata | S10 | AM-4, AM-7 | — |
| **P1** | **Cross-sectionnel crypto market-neutral** | S2 v-next | AM-3 | 4h (idéal) ; 1d en interim |
| **P1** | **MERGE : test du livre combiné** (le joyau) | session MERGE (S1+S9+S10) | AM-1 | tous les FINDINGS + co-active-ρ |
| **P2** | Cron collecte forward **COMEX** + contribuer fx_aud_cad | S5 v-next | AM-5 | — |
| **P2** | Ingestion **co-active-ρ** + prépa composition | S1, S9 | AM-1, AM-4 | — |
| **P3** | Re-run généralisation **à 4h** + cross-sec univers | S3 v-next | AM-2/AM-3 | **P0 4h** |
| **—** | Exhaustés : contribuer au merge, sinon STOP | S4, S6, S7, S8 | AM-1/AM-5 | — |

---

# 3. MANDATS v-NEXT — un par session, prêts à coller

> Tous : lire `STREAM_COMMON.md` + ce fichier. Tout shadow size-0, rien live, porte de merge humaine. TDD,
> fail-soft, **vérifie-contre-la-réalité**, étiquette `[VÉRIFIÉ]/[PROBABLE]/[PRÉSUMÉ]`. Enregistre TOUT essai
> auprès du `TrialRegistry` de S10.

---

## ▸ STREAM 0 — RE-FREEZE 4h + FUNDING NATIF PROFOND `[P0 — branche hermes/s0-refreeze-4h]`
**Pourquoi (AM-2) :** S2/S3/S9 citent tous « 4h profond multi-cycles » comme LA donnée manquante. Elle est
**gratuite** et le lake l'a omise (il a matché 6h à la fenêtre 360j CoinGlass).

**Tâche :**
1. **`binance_klines` 4h** pour les 7 paires, **profondeur maximale** (depuis le listing — l'endpoint klines
   sert 4h en deep). Vérifie la date de début réelle par paire.
2. **Funding natif Binance profond** via `/fapi/v1/fundingRate` (depuis l'inception ; pagine par temps). Gère
   l'**intervalle variable** (8h par défaut, **4h pour beaucoup de paires depuis 2023-10-12**, parfois 1h en
   cap/floor) via `/fapi/v1/fundingInfo` → nouvelle vue **`binance_funding_native`** (PROFONDE), **distincte** de
   la `binance_funding` forward-only existante (qui ne démarre qu'en ~2026-04 et a fait backtester sur du vide —
   cf. la correction trouvée par S8/S9).
3. **Attache le funding aux barres 4h** (forward-fill du funding settlé sur les barres 4h, **laggé** de sa
   latence réelle ; documente la convention — c'est la construction exacte où l'edge S2 a été validé).
4. *(Optionnel)* CoinGlass 4h **seulement si** le tier Hobbyist le permet (probable cap 360j comme le 6h — si
   capé, note-le ; le funding natif Binance est de toute façon plus profond et gratuit).

**Garde-fous :** TDD (test rouge d'abord) ; fail-soft ; **idempotent** ; **spot-check contre la réalité** (un pic
de funding connu, p.ex. un squeeze daté, doit apparaître à la bonne barre) ; vérifie la couverture par paire.
**KILL/honnêteté :** si la profondeur 4h d'une paire est trop courte pour un test cross-régime, dis-le par paire.

**Livrable :** vues `binance_klines` 4h + `binance_funding_native` dans le lake + note de couverture/profondeur
par paire. **C'est une modification du lake PARTAGÉ → à merger EN PREMIER, puis rebaser les autres branches.**
**STOP** — ingestion seulement, zéro recherche.

---

## ▸ STREAM 2 — CROSS-SECTIONNEL FUNDING MARKET-NEUTRAL `[P1 — branche hermes/s2-xsec]`
**Pourquoi (AM-3) :** le directionnel est mort de bull-beta (universel). S5 a prouvé que **seul le neutre/RV
retire la bull-beta**. Le funding en **cross-sectionnel long-short** la retire *par construction* — jamais testé.

**Tâche (la nouvelle construction) :**
- À chaque rebalance (sur **settlement funding** = lent, pas intraday), **classe l'univers** par `funding_z` ;
  **long le quantile haut, short le quantile bas**, **dollar-neutre** (gross long = gross short) → le beta BTC
  des deux jambes s'annule. C'est `funding_squeeze` exprimé **market-neutral cross-sectionnel**.
- Univers : le sous-ensemble à historique profond (BTC/ETH/SOL/AVAX) ; élargis aux 7 si la profondeur 4h (Stream
  0) le permet.

**Gate / KILL :**
- **MC random-SIGN null** (pas random-entry : c'est un long-short) ; **coût 2-jambes honnête** (15bps × 2,
  rebalance lent pour contrôler le turnover) ; **cross-régime** (idéalement sur le **4h** de Stream 0 ; sur 1d en
  interim **avec le caveat**) ; **|ρ| < 0.3 vs `funding_squeeze`** (c'est un cousin — s'il est trop corrélé, ce
  n'est **pas un nouveau pari**) ; **LOO-Δ-Sharpe** sur le livre.
- **MORT si :** l'edge long-short = juste la jambe long (re-importe la bull-beta) ; OU il ne bat pas le coût
  2-jambes ; OU il est le même pari que le directionnel `funding_squeeze` (ρ haut).
- **Bonus (après 4h) :** re-teste le **`funding_squeeze` directionnel cross-régime À 4h** — le test exact que
  S2/S3 n'ont **pas pu** faire (le lake n'avait que 6h-mono-régime et 1d-mauvaise-cadence).

**Livrable :** verdict du sleeve cross-sectionnel + (si 4h dispo) le test cross-régime propre de `funding_squeeze`.
Code `src/hermes/research/s2_xsec.py`. **STOP.**

---

## ▸ SESSION MERGE — LE LIVRE COMBINÉ (LE JOYAU) `[P1 — branche hermes/merge-combined-book]`
**Pourquoi (AM-1) :** le test fondateur jamais fait. La machinerie existe (S1 `combine_book`, S9 cross-book) ;
ce qui manque, c'est la **composition des nouveaux survivants** que la structure parallèle a empêchée.

**Tâche :**
1. **Assemble les survivants à-bon-signe** *qui passent les 3 filtres* : (a) **mécanisme a-priori clair** (pas de
   fishing), (b) **bon signe OOS**, (c) **construction market-neutral ou régime-symétrique**. Candidats : `fx_aud_cad`
   (décayé mais réel), S6 **CB-demand-accel** (bon signe), S6 **dollar-decoupling**, `funding_squeeze` (modeste),
   le **cross-sectionnel** (si S2 en livre un), + tout autre survivant à-bon-signe. *(Le ton S7 entre comme
   **conditionneur**, pas comme sleeve — il identifie le régime, ce n'est pas de l'alpha.)*
2. **Un livre pré-enregistré**, poids **non-optimisés** : equal-weight, ou HRP **clusterisé sur ρ co-active**
   (leçon S5) et **dimensionné sur la covariance daily causale** (leçon S9, PSD).
3. **Gate AU NIVEAU DE LA COMBINAISON** : la combinaison = **UN essai pré-enregistré** (enregistré au
   `TrialRegistry` S10) ; elle doit **battre un null combiné** (random-sign par sleeve, autocorr préservée) ;
   CPCV/PBO/DSR sur la série combinée ; **LOO-Δ-Sharpe > 0 par sleeve** (chacun gagne sa place) ; orthogonalité
   **sur semaines co-actives**, pas la grille zéro-remplie.

**Le danger, à énoncer (S10 gate brutalement) :** c'est *exactement* la machine à overfit (« combiner le
bruit »). Discipline : **seuls** les sleeves à mécanisme-a-priori **ET** bon-signe-OOS qualifient ; poids **non
optimisés** ; le trial count **inclut la combinaison comme un essai** ; **si le livre combiné ne bat pas le null
combiné au vrai trial count → NÉGATIF PROPRE.** Et un négatif propre ici **est** le end-state honnête du
programme (« la frontière données-gratuites est épuisée ; l'edge exige données nouvelles ou accrual forward »).
**Ce test est une due-diligence qu'on doit à la thèse, pas une victoire promise.**

**Livrable :** verdict du livre combiné (succès = il bat le null combiné déflaté ; sinon négatif propre précisant
ce qui manque). **STOP.**

---

## ▸ STREAM 10 — GOUVERNANCE-HUB + DÉCISIONS `[P0 — branche hermes/s10-honesty]`
**Pourquoi (AM-4, AM-7) :** tu es le hub d'intégrité ; le croisement a révélé des règles à imposer et des
décisions à trancher.

**Tâche :**
1. **Ériger en règles ENFORCÉES** (au-delà du per-stream) : (a) **nulls calibrés** — bannir l'IC en *niveau*
   (anti-conservateur 16% FPR à N=60, S6) ; exiger différencié/stationnaire ; (b) **chance-correction** sur toute
   métrique de stabilité/accord (S8) ; (c) **vrai trial-count par-livre** (déjà fait) ; (d) **orthogonalité sur
   semaines co-actives**, pas la grille zéro-remplie (S5).
2. **Imposer l'injection-de-fuite sur chaque DRIVER** avant verdict, **et la re-rejouer sur chaque SURVIVANT au
   merge** — parce que le pattern croisé est : **les bugs de composition fabriquent des POSITIFS** (S2/S6/S8).
3. **Cross-check de provenance** : confirmer que **chaque stream a utilisé `coinglass_funding_rate` (profonde)**,
   pas `binance_funding` (forward-only) — flaguer tout négatif fantôme dû à la mauvaise vue.
4. **Décision Amberdata = NON** (DVOL-VRP mort + surface morte + skew = probable RSKEW-redux) ; laisser le
   forward-log gratuit s'accumuler ; revisiter 6-12 mois. **Mets à jour l'acquisition map.**
5. **Correctif live-labels (diff exact, tâche MERGE pour Malachie — édition live)** : `newdata.py:12` (retirer
   « stablecoin strong BTC OOS 1.96/ETH 1.47 », ~15× le honnête ~0.12/0.13) ; `library.py:226-232` (DEMOTED →
   real-but-fragile, fail deflation, shadow) ; `library.py:235,241` (ajouter caveat : BTC/ETH ~0.8 corrélés →
   ~1.3-1.5 paris effectifs, pas 2 ; les ρ sont prose-only/sous-testés).
6. **Possède le gate du livre combiné** (joyau, AM-1) : la combinaison comptée comme **un** essai pré-enregistré.

**Livrable :** règles durcies + décisions tranchées + le diff live-labels prêt pour le merge. **STOP.**

---

## ▸ STREAM 1 — INTÉGRATION HARNESS `[P2 — branche hermes/s1-book-harness]`
**Pourquoi (AM-1, AM-4) :** ton harness est la fondation du joyau.
- **Ingère la leçon co-active-ρ (S5)** : quand le livre grandira, l'inverse-vol **sur-crédite les sleeves
  colinéaires qui co-firent rarement** (le ρ bas est un artefact de semaines (0,0)). Le HRP doit clusteriser sur
  **ρ co-active** (participation-ratio), pas le ρ brut zéro-rempli.
- **Prépare `combine_book`** à accepter les survivants mergés pour le test du joyau (interface propre).
Court. **STOP.**

---

## ▸ STREAM 9 — INTÉGRATION CROSS-BOOK `[P2 — branche hermes/s9-cross-book]`
**Pourquoi (AM-1, AM-4) :** tu portes la machinerie de combinaison.
- **Ingère le diagnostic co-active-ρ / participation-ratio (S5)** dans `orthogonality_matrix` (ta triple-vue le
  voit déjà partiellement via weekly ; rends-le explicite : ρ sur semaines où **les deux** sleeves firent).
- **Peuple 9.2** (cross-book diversifier) avec les **vrais survivants** au merge.
- **Fournis la machinerie de composition** au joyau.
Court. **STOP.**

---

## ▸ STREAM 5 — CRON COMEX + CONTRIBUTION `[P2 — branche hermes/s5-fx-metals-rv]`
**Pourquoi (AM-5) :** tu as nommé **deux fois** la courbe à terme COMEX comme « le seul axe pouvant casser le mur
2-D », bloqué par survivorship sur yfinance.
- **Bâtis le cron de collecte forward COMEX** : snapshot quotidien de la chaîne 8-maturités (gold/silver/PGM) →
  série de **calendar-spread PIT propre** en 12-18 mois. *(Pas de backfill — accrual forward, comme le skew de S4.)*
- **Contribue `fx_aud_cad`** (réel mais décayé) au **livre combiné (joyau)** comme **pari faible** avec son caveat
  de décay — laisse la **combinaison** décider, pas le solo.
Court-moyen. **STOP.**

---

## ▸ STREAM 3 — RE-RUN À 4h `[P3 — branche hermes/s3-universe — DÉPEND DE STREAM 0]`
**Pourquoi (AM-2/AM-3) :** ton négatif était sur **1d** (mauvaise cadence) et 6h (mono-régime). À **4h** (la
cadence de validation de l'edge), le verdict peut changer.
- **DÈS que le 4h de Stream 0 est mergé** : re-run la **généralisation S2 à 4h** sur les 7 paires (per-asset
  MC-null + joint-null daté + déflation) ; re-run le **cross-sectionnel** sur l'univers 4h profond.
- **Jusqu'au 4h : HOLD.** Ne re-run PAS sur 1d (épuisé, AVAX-seul rejeté).
Court. **STOP.**

---

## ▸ STREAMS EXHAUSTÉS — contribuer au merge, sinon STOP

- **S4 (options) — FAIT.** Contribue le **forward-log skew** à l'infra de collecte unifiée (AM-5). **Amberdata =
  NON** (S10 tranche). **Aucune nouvelle chasse.**
- **S6 (or) — FAIT.** Contribue **CB-demand-accel** + **dollar-decoupling** au **livre combiné (joyau)** comme
  paris faibles market-neutral (bon signe, laisse la combinaison décider). **Aucune chasse** sauf si une série de
  flux officiel **plus haute fréquence** apparaît (le seul levier contre le mur N≈60).
- **S7 (ton) — FAIT.** Le classifieur identifie les régimes d'assouplissement (label-shuffle p=0.013) → offre-le
  comme **conditionneur de régime** (pas un sleeve) à S8 / au livre combiné. **Aucune nouvelle chasse.**
- **S8 (régime) — FAIT.** Fournis les **labels risk-off** à l'overlay S9 + aux checks de régime-symétrie du livre
  combiné. **Aucune nouvelle chasse.**

---

## 4. LE RAPPEL STRATÉGIQUE (AM-5, transversal)

Les backtests sont contaminés par la bull-beta ; le **régime bear/range ACTUEL (2026) est le labo naturel pour la
falsifier**. Au-delà des mandats ci-dessus, la vraie bascule est : **faire tourner en shadow size-0, dès
maintenant, tout sleeve à-bon-signe** (cross-sectionnel, fx_aud_cad, CB-accel) **pour accumuler de l'OOS forward
dans ce régime** — un mois d'OOS bear vaut plus que n'importe quel backtest pour distinguer l'edge réel de la
bull-beta. Le programme passe de « trouver un edge en backtest » à « **accumuler l'OOS forward + bâtir l'infra de
collecte, prêt à promouvoir quand un sleeve survit au régime, pas au backtest** ».
