# STREAM 9 — ORTHOGONALITÉ CRYPTO↔FX & ALLOCATION CROSS-LIVRE
**Séniorité : Senior portefeuille · Branche : `hermes/s9-cross-book` · Après S1+S5 (infra constructible maintenant)**

> **LIS D'ABORD** : `docs/streams/STREAM_COMMON.md`, le programme §Stream 9, le dead-ends ledger,
> `research_fx/LAB_JOURNAL.md`. Réalise « **un seul Hermès, deux livres, un cerveau** » : mesurer l'orthogonalité
> crypto↔FX et trouver les sleeves qui **diversifient les DEUX livres**. Le labo a montré que crypto↔FX
> **lead-lag = zéro** (X1/X2 morts) → **ce n'est PAS de la prédiction lead-lag** ; c'est de la **diversification
> par structure de corrélation**.

## MANDAT
Mesurer en continu l'orthogonalité crypto↔FX et identifier les sleeves à **ρ faible dans le livre combiné**.
**Note de parallélisme :** les sleeves des autres streams n'existeront qu'**au merge** → tu **construis la
machinerie maintenant** (matrice vivante, allocateur cross-livre, overlay) et la **peuples à l'intégration** ; en
parallèle, fais-la tourner sur ce qui est committé (S2 `funding_squeeze` + le livre existant).

## BORNES DE PÉRIMÈTRE
Tu ne re-testes **PAS** X1/X2 (lead-lag crypto→FX, morts). Tu travailles la *matrice de corrélation* du livre
global et l'**allocateur de risque cross-livre**. Tu ne crées pas de signal mono-livre (ça appartient aux streams
mono-livre).

## INPUTS LAKE
Séries de rendements des sleeves des deux livres (produites par les autres streams — au merge ; en parallèle :
S2 + le livre committé). `binance_klines`, `gold`/`dxy` pour le contexte. Réutilise le harness du Stream 1 (au
merge ; stand-in `book.py` en parallèle).

## LES 3 VECTEURS

### 9.1 — Matrice d'orthogonalité vivante crypto↔FX + allocation par cluster
- **H :** même sans lead-lag, des sleeves crypto et FX à **ρ≈0** réduisent la variance du livre *global*.
- **Méca :** Medallion = beta négatif/nul au marché par *diversification* de centaines de facteurs décorrélés.
- **Méthode :** implémente `orthogonality_matrix()` (corrélation **production-lag**, daté/niveau-régime) + un
  **allocateur par cluster** (HRP cross-livre). Branche-le au dashboard (read-only).
- **KILL :** **matrice honnête** ; un sleeve qui *paraît* diversifiant mais **crowde le même beta niveau-régime**
  (comme S1×S2 = 0.35 > 0.30 daté) est **démasqué → ne compte pas**. ρ brut ≠ ρ daté.
- **Liv. :** `orthogonality_matrix()` + allocateur cross-livre + matrice ρ vivante au dashboard.

### 9.2 — Sleeves explicitement « cross-book diversifiers »
- **H :** certains signaux (ex. un état de stress global) améliorent *les deux* livres simultanément.
- **Méca :** la valeur d'un sleeve dans le cerveau partagé **>** sa valeur dans un agent-frère aveugle à l'autre livre.
- **Méthode :** infra prête maintenant ; à l'intégration, teste chaque sleeve candidat des autres streams sur la
  **LOO-Δ-Sharpe du livre GLOBAL** (pas d'un seul livre).
- **KILL :** doit améliorer la **LOO-Δ-Sharpe du livre global** ; sinon il appartient à un stream mono-livre.
- **Liv. :** identification (le cas échéant) de tels sleeves — au merge ; **en parallèle : le test prêt + documenté**.

### 9.3 — Overlay anti-black-swan partagé (la lacune de l'état des lieux) `[constructible maintenant]`
- **H :** l'état des lieux note « **aucune couverture anti-black-swan** » — un overlay de coupure global (gating
  sur stress extrême, dépeg, gap) protège les deux livres.
- **Méca :** protection structurelle au **niveau livre**, pas par stops individuels.
- **Méthode :** design d'un overlay = règles de coupure sur stress extrême (vol spike, dépeg stablecoin via
  `defillama`, gap de prix). **Mesure son coût en Sharpe** sur le livre committé (un overlay trop sensible saigne
  l'edge). Shadow d'abord.
- **KILL :** **ce n'est pas un alpha** — c'est une réduction de tail risk ; il doit **améliorer le Calmar/le
  drawdown SANS détruire le Sharpe net**. Un overlay trop sensible = rejet.
- **Liv. :** **design d'overlay (shadow) + son coût mesuré** → c'est le livrable concret de ce stream en parallèle.

## INVARIANTS (rappel)
Pas de lead-lag (mort). ρ **daté/niveau-régime**, pas brut. LOO-Δ-Sharpe du **livre global** pour 9.2. L'overlay
est de la réduction de tail risk, pas de l'alpha. Production-lag dans toute corrélation. Shadow. TDD.

## SORTIE
`research_streams/S9/FINDINGS.md` (**9.1 matrice + 9.3 overlay = livrables parallèles ; 9.2 = test prêt pour le
merge**) · `research_streams/S9/CHECKPOINT.md` · code `src/hermes/research/s9_cross_book.py` (+ tests). **STOP.**
