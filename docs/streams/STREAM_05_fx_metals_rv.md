# STREAM 5 — LIVRE COMBINÉ FX/MÉTAUX (RELATIVE-VALUE, MARKET-NEUTRAL)
**Séniorité : Senior portefeuille + FX RV · Branche : `hermes/s5-fx-metals-rv` · Cœur du 2ᵉ livre, après S1**

> **LIS D'ABORD** : `docs/streams/STREAM_COMMON.md`, le programme §Stream 5, le dead-ends ledger,
> `research_fx/HONEST_NEGATIVE_REPORT.md`, `research_fx/LAB_JOURNAL.md`. **Le labo a prouvé que TOUT signal
> directionnel FX/métaux en données gratuites est mort.** Ta cible n'est PAS un signal de plus — c'est un
> **livre** de sleeves faibles, orthogonaux, **market-neutral**, dont le *combiné* est bon.

## MANDAT
Assembler **beaucoup de sleeves faibles, orthogonaux, market-neutral** sur XAU/XAG/FX, et prouver que le **LIVRE
combiné** passe le gate — en **retirant la bull-beta par construction** (jambes neutres), ce qui tuait B2/B3/X1.
Le « redoutable » se mesure en **Sharpe du livre combiné et |ρ| faible**, pas en Sharpe solo (thèse Medallion :
raison 50.75 % du temps, edge = combinaison de paris décorrélés à beta négatif).

## BORNES DE PÉRIMÈTRE
Tu utilises le harness du Stream 1 (au merge) et le conditionneur or du Stream 6 (au merge ; stand-in en
parallèle), mais tu **ne re-testes AUCUN dead-end directionnel du ledger**. Ton terrain = constructions
**relatives/neutres**. Le crypto n'est pas ton livre.

## INPUTS LAKE
`data/fx/` (XAUUSD/XAGUSD existants), `gold` (GC=F), `silver` (SI=F), `gdx`, `dxy`, crosses FX ; flux GLD +
`etf_iau_flows` ; `wgc_cb_demand_gdt` (conditionneur, au merge via S6) ; `research_fx/fxgate.py` (réutilise).
**Coût FX honnête : 1.6 bps majors / 2.1 crosses / 7.1 exotiques. Swap-free annule le carry classique.**

## LES 3 VECTEURS

### 5.1 — Gold-Silver & Gold-vs-Miners relative-value (neutre au beta or)
- **H :** le *ratio* XAU/XAG et le spread or-vs-mineurs (proxy GDX) ont des dynamiques propres, **décorrélées de
  la dérive haussière de l'or**.
- **Méca :** Renaissance trade les *paires* qui co-bougent (or-argent cités explicitement) ; le ratio retire le
  facteur commun « or monte ».
- **Données :** `gold`/`silver` (vrai volume via GC=F/SI=F), `gdx` (ETF gratuit).
- **Méthode :** le ratio comme **facteur lent/conditionné**, PAS le pair-trade rapide (le labo note half-life
  gold-silver ~10 j → pas de réversion rapide, mort). MC random-sign null + LOO-Δ-Sharpe.
- **KILL :** si l'edge = encore la **jambe longue or** (bull-beta) → mort. ρ vs facteur or doit être ≈0 par
  construction.
- **Liv. :** **1-2 sleeves RV neutres**, jugés par leur contribution au livre.

### 5.2 — FX cross-section : VRP & skewness risk-premium (market-neutral, dollar-neutre)
- **H :** trier les devises par *volatility risk premium* (Della Corte) ou *skewness risk premium*
  (Li-Sarno-Zinna 2025) donne un long-short market-neutral — axes que le labo n'a PAS testés.
- **Méca :** prime d'assurance contre l'asymétrie/la vol, distincte du carry.
- **Données — ⚠️ VIGILANCE :** nécessite des **IV options *par paire FX*** ; le lake **n'en a pas** (Amberdata
  non acheté), et les indices vol FX CBOE (EVZ) ont **cessé ~jan-2025**. **Prouve la disponibilité point-in-time
  AVANT de chasser.** Le DVOL crypto n'est PAS un substitut FX.
- **Méthode :** d'abord, **audit de disponibilité de données IV-FX libres/peu chères en point-in-time**. Si rien
  → verdict « données verrouillées » (livrable valide) + note ce qu'un achat débloquerait.
- **KILL :** SRPHML est *negatively-skewed* (crash GFC/Covid) = potentiellement crowded ; le gate doit montrer une
  **contribution nette positive au livre après coût honnête**, pas une prime de risque déguisée. **Pas de data
  point-in-time libre → verdict « données verrouillées ».**
- **Liv. :** soit un sleeve RV-vol FX validé, soit le **verdict « données verrouillées » précis**.

### 5.3 — Le LIVRE FX/métaux lui-même (la vraie cible)
- **H :** même si chaque sleeve échoue un gate *solo*, un panier de **4-6 sleeves ρ≈0** peut être net-accretif
  (des morts « en héros » peuvent vivre en sleeves faibles orthogonaux — *let the LOO-Δ-Sharpe decide*).
- **Méca :** réduction de variance ∝ √N de paris décorrélés (Medallion).
- **Méthode :** assemble tous les candidats RV/neutres (5.1, 5.2 si dispo, + variantes neutres d'idées du ledger
  **uniquement sous forme neutre**) dans un livre ; mesure le combiné via le harness (stand-in : `book.py` +
  `fxgate.py`).
- **KILL :** **PBO du livre < 0.5**, DSR positif, **LOO-Δ-Sharpe > 0 *par sleeve***, à travers régimes (incl. le
  bear or hypothétique / la décennie 2011-15). **Attention `[ACQUIS]` §3.7 : coût FX bon marché → plus de candidats
  survivent l'arithmétique → le mur d'overfitting REMPLACE le mur de coût.** Coordonne avec Stream 10.
- **Liv. :** **verdict du LIVRE combiné FX/métaux** (succès = ≥2-3 sleeves orthogonaux dont le combiné passe ;
  sinon négatif propre précisant *quelle* donnée/structure manquerait).

## INVARIANTS (rappel)
Pas de directionnel FX/métaux en données gratuites (mort, prouvé). Jambes **neutres** (la bull-beta retirée par
construction). Coût FX réel + swap-free. Déflation **plus stricte** côté FX (Stream 10). Shadow size 0. TDD.

## SORTIE
`research_streams/S5/FINDINGS.md` (**5.3 = le verdict du livre ; 5.1/5.2 = les briques**) ·
`research_streams/S5/CHECKPOINT.md` · code dans `src/hermes/research/s5_fx_metals.py` (+ tests). **STOP.**
