# STREAM 6 — CONDITIONNEUR DE RÉGIME « OR »
**Séniorité : Mid-senior macro · Branche : `hermes/s6-gold-regime` · Après S1, en parallèle de S5**

> **LIS D'ABORD** : `docs/streams/STREAM_COMMON.md`, le programme §Stream 6, le dead-ends ledger,
> `research_fx/HONEST_NEGATIVE_REPORT.md`. La donnée or WGC est **déjà dans le lake** (addendum). Tu construis
> un **état de régime**, **jamais un pari long-or** (= bull-beta que le Stream 10 tuera).

## MANDAT
Exploiter une découverte fraîche : la relation **or↔taux réels a rompu** — R²(or, TIPS 10y) **84 % (2005-21) →
3 % (2022-23) → 7 % (depuis 2024)**. L'or est désormais piloté par achats de banques centrales (>1000 t/an depuis
2022), dé-dollarisation, géopolitique. Cela **recontextualise** la mort de B2 (kill correct *sur l'échantillon
historique*, mais **le régime a basculé**). Tu construis l'**état de régime** comme **conditionneur** pour les
Streams 5/8.

## BORNES DE PÉRIMÈTRE
Tu ne ressuscites **PAS** B2/B3 comme signaux *directionnels* (morts). Tu ne fais **pas** de pari long-or. Tu
produis des **états/conditionneurs**, validés seulement en terme d'interaction.

## INPUTS LAKE (or WGC — dans le lake)
`fred_dfii10` (+ `_vintage` ALFRED = point-in-time **obligatoire**), `wgc_cb_demand_gdt` (★ **fait autorité**,
trimestriel depuis 2010), `wgc_cb_changes_monthly` (par pays — **caveats : déclaré-seulement, jumeaux Turkey/Turkey***),
`wgc_cb_reported_monthly_global_proxy`, flux GLD + `etf_iau_flows`, `gold`, `dxy`. **Lague :** demande CB
trimestrielle (publication décalée), ETF ≥3 j, FRED +1 j.

## LES 3 VECTEURS

### 6.1 — Variable d'état « régime or » (libre, lente, exogène)
- **H :** un état latent {régime taux-réels vs régime flux/CB-buying} conditionne *quels* signaux or marchent.
- **Méca :** en régime « taux-réels » les signaux DFII10 ont du sens ; en régime « flux/CB » ce sont les signaux
  de flux/positionnement.
- **Données :** **corrélation roulante TIPS-or** (`fred_dfii10` vintage), **demande CB** (`wgc_cb_demand_gdt`),
  **corrélation roulante DXY-or** (`dxy`/`gold`).
- **Méthode :** construis `gold_regime_state` = état discret/continu combinant (corr roulante TIPS-or basse vs haute)
  + (régime de demande CB élevée). **Point-in-time strict** (vintages, lags de publication).
- **KILL :** **c'est un état, pas un trade** ; validé **uniquement** comme *terme d'interaction* (**shuffle-label
  null** contre un sleeve accessible) ; s'il n'améliore aucun sleeve → inutile.
- **Liv. :** feature `gold_regime_state` → `research_streams/S6/features/` (consommée par S8 au merge).

### 6.2 — Choc taux-réels DFII10, *conditionné* au régime taux-réels uniquement
- **H :** le signal B2 mort *non conditionné* peut vivre *restreint au régime où il s'applique*.
- **Méca :** réintroduit le fondamental, mais **filtré par l'état 6.1** → retire la période 2011-15/le bull-beta
  qui le tuait.
- **Méthode :** B2 (choc DFII10) appliqué **seulement** quand `gold_regime_state` = « taux-réels » ; exprimé
  **neutre** (pas long-or directionnel).
- **KILL :** le labo a tué X3 (« conditioning ») car « conditionné ≈ base, le *complément* score mieux,
  shuffle-p=0.42 ». **La barre est haute :** prouver que le conditionnement bat *à la fois* la base **ET** son
  complément, **shuffle-null p<0.05** ; sinon mort (non-événement).
- **Liv. :** **verdict honnête conditioning-or** (probable échec, mais à *prouver* proprement avec l'état de
  régime *neuf*, pas l'ancien split).

### 6.3 — Flux ETF or × divergence prix, *en relative-value*
- **H :** le flux GLD+IAU (onces-en-trust) en *divergence* du prix, exprimé *neutre* (vs un panier), porte de l'info.
- **Méca :** analogue or du win on-chain stablecoin.
- **Données :** flux GLD (spdrgoldshares) + `etf_iau_flows` (déjà dans le lake) ; lague ≥3 j.
- **Méthode :** flux ETF or en **divergence** du prix + **expression neutre** conditionnée régime (6.1) — PAS le
  flux directionnel.
- **KILL :** B5 est mort « flux décalé sur le prix, p=0.203 » — **n'y retourne pas en directionnel** ; la seule
  mutation est flux-en-divergence + neutre + conditionné. Si le lag structurel persiste → mort.
- **Liv. :** sleeve candidat ou négatif daté.

## INVARIANTS (rappel)
Conditionneur, **jamais** un pari long-or. Point-in-time **obligatoire** (vintages ALFRED). Respecte les caveats
WGC (déclaré-seulement, jumeaux Turkey). Shuffle-label null pour tout conditionneur. Shadow. TDD.

## SORTIE
`research_streams/S6/FINDINGS.md` (l'état de régime + verdicts 6.2/6.3) · `research_streams/S6/CHECKPOINT.md` ·
feature dans `research_streams/S6/features/` + code `src/hermes/research/s6_gold_regime.py` (+ tests). **STOP.**
