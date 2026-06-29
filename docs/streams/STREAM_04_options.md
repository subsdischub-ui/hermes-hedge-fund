# STREAM 4 — SURFACE OPTIONS CRYPTO 2.0 (SKEW & GEX)
**Séniorité : Senior vol/dérivés · Branche : `hermes/s4-options` · Après S1, en parallèle**

> **LIS D'ABORD** : `docs/streams/STREAM_COMMON.md`, le programme §Stream 4, le dead-ends ledger,
> `research_fx/LAB_JOURNAL.md`.
>
> ⚠️ **RÉALITÉ DONNÉES À INTÉGRER (change ton ordre de priorité vs le programme).** Amberdata n'a **pas**
> été acheté. Donc : le **25Δ skew / Greeks n'est PAS backfillé** — `deribit_surface_snapshot` est un
> **forward-log** démarré ~2026-06 (1 ligne/ccy/jour), **non backtestable historiquement**. Seul le **DVOL**
> est backfillable (`deribit_dvol`, BTC/ETH horaire depuis 2021). **Conséquence : ton axe backtestable
> immédiat est le 4.3 (VRP-DVOL) ; 4.1 (skew) et 4.2 (GEX) sont en *collecte forward limitée* — tu démarres
> le chrono et fais ce qui est possible, sans fabriquer d'historique.**

## MANDAT
Reprendre l'axe options là où le labo s'est arrêté faute de données *backtestables*. Le labo a tué la grille
1024-config (VRP/TERM/VOV/RSKEW : grid-max 0.193 < null 0.195) et a noté que le **25Δ risk-reversal littéral**
était le seul axe inexploré « mais NON backtestable rétroactivement ».

## BORNES DE PÉRIMÈTRE
Tu ne touches **pas** au positionnement perp (Stream 2). Ton terrain = la *surface* options Deribit (skew, term,
GEX) et le DVOL. Pas de FX (Stream 5).

## INPUTS LAKE
`deribit_dvol` (**BACKFILLABLE**, 2021+, BTC/ETH) ; `deribit_surface_snapshot` (**forward-only**, skew 25Δ +
Greeks) ; `coinglass_options_oi` / `_vol` / `_info` (BTC/ETH, OI/volume agrégés — **pas la surface par strike**) ;
`binance_klines` (réalisé). **BTC/ETH uniquement** (pas de marché options sur les autres paires).

## LES 3 VECTEURS (réordonnés par disponibilité réelle des données)

### 4.3 (PRIORITÉ — backtestable) — VRP DVOL comme *gate de sizing* (pas comme direction)
- **H :** le DVOL VRP (implied − realized) ne donne pas de direction (sleeve DVOL rétrogradé), mais module la
  *taille* : réduire l'exposition quand le VRP signale la complaisance, l'augmenter dans la peur.
- **Méca :** le VRP crypto reste positif mi-2026 malgré des implieds bas (vol réalisée effondrée) → le carry de
  vol existe encore.
- **Données `[VÉRIFIÉ — backfillable]` :** `deribit_dvol` (implied) vs réalisé calculé depuis `binance_klines`,
  2021→2026.
- **Méthode :** VRP = DVOL − realized_vol (fenêtre alignée, **DVOL +1 bar de lag**). Utilise-le comme **multiplicateur
  de taille** sur un sleeve confirmé (S2), pas comme signal directionnel.
- **KILL :** un gate de sizing **ne crée pas d'edge** ; il doit **relever le Sharpe *déflaté* d'un sleeve confirmé
  sur held-out**, sinon inutile.
- **Liv. :** feature de sizing pour la couche du Stream 1.3, **validée hors-échantillon** → `research_streams/S4/features/`.

### 4.1 (forward-limité) — Skew 25Δ (risk-reversal / butterfly) littéral
- **H :** le vrai skew OTM put/call (pas le realized-skew du prix) porte de l'info de peur/positionnement
  orthogonale au funding/momentum.
- **Méca :** demande d'assurance asymétrique des dealers/buy-side.
- **Données `[RÉALITÉ]` :** **PAS d'historique** (Amberdata non acheté). `deribit_surface_snapshot` ne contient
  que le forward-log court (~depuis 2026-06). **Tu ne peux PAS faire de backtest pluri-annuel du skew.**
- **Méthode :** (a) **vérifie/renforce le forward-log** `deribit_iv.py` (cron quotidien, 1 ligne/ccy/jour, dédupé) ;
  (b) sur la **fenêtre courte disponible**, fais une analyse *exploratoire* (corrélation contemporaine skew↔forward
  return, dé-overlap fear/drawdown) **étiquetée explicitement « non concluante, échantillon insuffisant »** ;
  (c) documente *exactement* ce qu'un achat Amberdata (skew depuis 2021-05-21) débloquerait, pour la décision $.
- **KILL :** **pas de verdict de gate possible sur historique court** — tout « signal » sur quelques mois est
  rejeté par construction. Le vrai gate (quand l'historique existera) devra prouver que le 25Δ skew **n'est pas**
  un re-emballage du fear-reversion déjà mort (RSKEW), *après* dé-overlap.
- **Liv. :** **verdict honnête « chrono démarré, non backtestable aujourd'hui »** + cron forward-log opérationnel
  + note d'arbitrage data (Amberdata) pour Malachie.

### 4.2 (forward/limité) — Dealer-GEX crypto (gamma flip → régime)
- **H :** le signe du gamma des dealers conditionne le *type* de marché — gamma positif → mean-reversion ;
  gamma négatif → momentum.
- **Méca :** flux de hedging *mécaniques* (Baltussen-Da-Lammers-Martens, JFE).
- **Données `[RÉALITÉ]` :** `coinglass_options_oi` donne l'OI options **agrégé** (pas par strike) → un **proxy GEX
  grossier** (max-pain, OI par expiry) est possible ; le GEX *propre* (OI par strike × gamma) nécessite la chaîne
  Deribit avec Greeks (forward via `deribit_surface_snapshot`, ou achat Amberdata pour l'historique).
- **Méthode :** construis un **proxy GEX agrégé** sur ce qui existe ; étiquette sa limite. C'est un **CONDITIONNEUR**,
  jamais un signal standalone.
- **KILL :** il doit *améliorer* un sleeve existant (gate-interaction, **shuffle-label null**), jamais être jugé seul.
  Le labo a tué le « 10am pinning » faute de panel — ici l'axe est le *régime gamma*, pas le pinning de strike.
  Sur données limitées : verdict exploratoire, pas de gate prématuré.
- **Liv. :** feature `crypto_gex_regime` (proxy) livrée au Stream 8 comme conditionneur, **avec sa limite data
  documentée** → `research_streams/S4/features/`.

## INVARIANTS (rappel)
**Ne fabrique JAMAIS d'historique de skew/GEX** : forward-only = forward-only. DVOL +1 bar de lag. Conditionneur ≠
signal. Gate niveau-livre. Shadow size 0. TDD.

## SORTIE
`research_streams/S4/FINDINGS.md` (**4.3 = le livrable concret ; 4.1/4.2 = chrono démarré + note data**) ·
`research_streams/S4/CHECKPOINT.md` · code dans `src/hermes/research/s4_options.py` + cron `deribit_iv.py`. **STOP.**
