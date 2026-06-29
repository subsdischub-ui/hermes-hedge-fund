# STREAM 2 — DÉBLOCAGE DATA POSITIONNEMENT & LIQUIDATIONS (CRYPTO)
**Séniorité : Senior trader + data-eng · Branche : `hermes/s2-positioning` · Priorité HAUTE (plus forte VE)**

> **LIS D'ABORD** : `docs/streams/STREAM_COMMON.md`, le programme §Stream 2, le dead-ends ledger,
> `research_fx/LAB_JOURNAL.md`. **Tu portes le seul lead crypto réellement ouvert du labo.** Bonne nouvelle :
> la donnée est **déjà dans le lake** (CoinGlass souscrit) — tu gates en entier *maintenant*.

## MANDAT
Aller chercher le **seul lead crypto réellement ouvert** : **top-trader LSR contrarian** (H3 : ρ(top_lsr, fwd 6h)
= **−0.27 / −0.33 / −0.11** BTC/ETH/SOL, *signe stable* sur les deux moitiés de fenêtre — **non réfuté, juste
data-gated** à 30 jours). Le débloquer avec l'historique CoinGlass, et au passage refaire la **cascade de
liquidations** avec de **vraies** données (pas le proxy OHLCV mort).

## BORNES DE PÉRIMÈTRE
Tu ne touches **ni** aux options (Stream 4) **ni** à l'expansion d'univers (Stream 3) **ni** au FX. Ton terrain =
positionnement (LSR top-trader/compte/position), open interest, liquidations, sur **BTC/ETH/SOL d'abord**.

## INPUTS LAKE (CoinGlass — débloqué)
`coinglass_top_lsr_account`, `coinglass_top_lsr_position`, `coinglass_global_lsr_account` (contrôle),
`coinglass_open_interest` (+ agrégé), `coinglass_funding_rate`, `coinglass_liquidation`, `coinglass_taker_volume` ;
`binance_klines` pour les rendements forward. Intervalles : **`6h` (la cadence du signal H3, 360j) ET `1d`
(all-time, pour le test cross-régime)**. **Vérifie la date de début par paire.**

## LES 3 VECTEURS

### 2.1 — Top-trader LSR contrarian, débloqué par historique `[LEAD N°1 — plus forte VE]`
- **H :** quand les top traders perp sont nets-longs à l'extrême, le forward return est négatif (épuisement/crowding).
- **Méca :** axe de *positionnement réglementé* (qui détient le risque), **orthogonal à tous les dead-ends
  dérivés du prix**.
- **Données `[VÉRIFIÉ — dans le lake]` :** `coinglass_top_lsr_account` / `_position` remontent à ~2020 (BTC),
  `6h` sur 360j + `1d` all-time. Les **12-18 mois** que le labo réclamait sont là (et plus).
- **Méthode :** (a) signal contrarian = −sign(z(top_lsr)) ou continu ; cible = forward return 6h (et teste 1d sur
  l'historique long). (b) Construis le sleeve, gate-le par la machinerie. (c) **Test cross-régime explicite** :
  bull 2021, bear 2022, range 2026 — le signe doit tenir.
- **KILL :** gate complet 2020→2026 incl. **bear/range** ; doit battre le **joint-null daté de-overlappé** (le
  test qui a tué S1 à 0.69 et n'a laissé passer que S2 à 0.017) **ET** la déflation **DSR/PBO**. **Si le signe
  flippe sur l'historique long ou s'effondre OOS-40 → mort.** |ρ|<0.3 vs S2/S3.
- **Liv. :** sleeve `positioning_top_lsr_contrarian` en **shadow size 0** + **verdict daté** dans FINDINGS.

### 2.2 — Cascade de liquidations avec VRAIES données (refaire H5 proprement)
- **H :** un flush de liquidations forcées crée un rebond mean-reverting exploitable.
- **Méca :** désendettement forcé → dislocation temporaire prix vs valeur.
- **Données `[VÉRIFIÉ]` :** le labo avait échoué (`allForceOrders` Binance = HTTP-400 → proxy OHLCV mort,
  joint-null p=0.907 = bull-beta). `coinglass_liquidation` fournit les **liquidations historiques réelles** ≥2019.
- **Méthode :** définis le flush sur les vraies liquidations (taille/cluster), cible = rebond forward ; gate complet
  avec **séparation bull/bear**.
- **KILL :** même joint-null daté + séparation régime ; **si l'edge vit seulement en jambe bull (comme le proxy),
  c'est encore de la bull-beta → mort.**
- **Liv. :** verdict honnête « real-data H5 » — **probablement encore mort, mais à *prouver* sur vraies données**.
  Un négatif propre est un livrable valide.

### 2.3 — Term-structure du funding × OI (crowding multi-venue)
- **H :** la *structure* du funding (dispersion cross-venue, accélération de l'OI à funding constant) marque un
  build fragile qui se dénoue.
- **Méca :** 77 % du volume crypto est perp ; le price discovery s'est déplacé vers les perps ; funding/OI/liquidations
  sont les signaux structurels.
- **Données :** `coinglass_funding_rate` multi-venue + `coinglass_open_interest` (candlesticks) ; `coinglass_taker_volume`.
- **Méthode :** **dispersion funding cross-venue** + **OI-acceleration à funding constant** — PAS le simple term-spread.
- **KILL :** le labo a tué H2 funding-term (0.7 bps gross « 30 bps coût, signe inversé) — **ne le ré-essaie PAS à
  l'identique** ; la mutation autorisée est *dispersion multi-venue + OI-accélération*. Si tout retombe sur « high
  funding continues up » (= **S2 déjà capté**) ou « buy dips » (= bull-beta) → mort.
- **Liv. :** verdict + sleeve candidat le cas échéant.

## INVARIANTS (rappel)
Lague la donnée à sa latence réelle. Gate niveau-livre (LOO-Δ-Sharpe), jamais Sharpe solo. **Vérifie les chiffres
contre la réalité** (le signe ρ, les dates par paire). Shadow size 0. TDD.

## SORTIE
`research_streams/S2/FINDINGS.md` (verdict des 3 vecteurs, **2.1 en tête**) · `research_streams/S2/CHECKPOINT.md` ·
code dans `src/hermes/research/s2_positioning.py` (+ tests). **STOP.**
