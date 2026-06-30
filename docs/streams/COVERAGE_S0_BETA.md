# COVERAGE — Stream 0-β (acquisition pour la Boucle β / Illiquid-Niche Carry)

> Branche `round3/s0-beta`. Données + code de source + mécanique de calcul **seulement** —
> rien live, aucun essai, aucun backtest de stratégie (la récolte sur l'illiquide se mesure au
> mandat suivant : la Boucle β). Étiquettes : `[VÉRIFIÉ]` lu/exécuté · `[PROBABLE]` inférence ·
> `[PRÉSUMÉ]` non confirmé.

Ce Stream-0-β ajoute la **queue illiquide** du carry au data lake gelé — les perps Binance
USDT-M **au-delà des 23 majors** et les marchés **exotiques Hyperliquid** — plus la **métrique de
liquidité par barre** (`datalake/liquidity.py`) qui rend l'axe H1 (funding-APR vs liquidité)
*calculable*. La machinerie carry de la Boucle α (`carry_join` / `carry_harvest`) tourne sur ces
marchés **inchangée** (elle est agnostique au symbole).

L'hypothèse de β `[rappel mandat]` : le carry est arbitré là où il y a de la liquidité spot ; il
devrait **persister sur la queue illiquide non-arbitrable** (H1). Le test tueur (H2) : survit-il à
la **slippage et à la queue brutale** de l'illiquide ? Ce Stream-0-β acquiert les **données** ; il
ne teste rien.

---

## 1. CE QUI EST INGÉRÉ (vues additives au lake gelé)

Toutes les nouvelles vues sont **additives** : elles vivent dans des répertoires `raw/` NEUFS, donc
les vues des 23 majors (`binance_klines_4h`, `binance_spot_klines_4h`, `hyperliquid_funding_1h`, …)
et **leurs tests de couverture « exactement 23 »** sont **byte-intacts**. `ts` = epoch **millisecondes**
int64 partout (convention du lake). Univers exclu des 23 = `ALL_PAIRS` (7 live + 16 extended).

### A. La queue illiquide Binance — `binance_tail_*`  `[VÉRIFIÉ]`
- **Source** : Binance USDT-M PERP + SPOT, publics, no-key, joignables depuis la France. Code :
  `src/hermes/datalake/sources/binance_tail.py` (réutilise les paginators de `binance.py` /
  `binance_spot.py` — **jamais forkés**).
- **Sélection (la politique β)** : filtre dur **≥1 an d'historique** (viabilité de données, via
  `onboardDate`), puis **stratification par déciles de dollar-volume** (24h quote-volume) qui
  **sur-représente le bas illiquide** (`tail_select.stratified_decile_sample`, pur + déterministe).
  Objectif = **variance de liquidité maximale**, PAS « les plus liquides ». Le « sweet spot » H1/H2
  est un *résultat à découvrir*, pas un critère d'entrée.
- **Univers ingéré : 18 marchés, déciles 0→9**, dollar-volume de **$0.77 M (CVC, D0)** à
  **$855 M (HYPE, D9)**. **13/18 ont une jambe spot** (cash-and-carry) ; 5 sont perp-only
  (cross-venue uniquement). Profondeur réelle `[VÉRIFIÉ — requête lake]` :

  | base | déc | perp bars 4h | perp début | spot bars | funding rows | funding début | spot ? |
  |------|----:|-------------:|-----------|----------:|-------------:|--------------|:------:|
  | BNT       | 0 |  6 331 | 2023-08-10 | 14 017 | 3 530 | 2020-12-28 | ✓ |
  | CHILLGUY  | 0 |  3 481 | 2024-11-27 |      0 | 3 479 | 2024-11-27 | — |
  | CVC       | 0 |  2 462 | 2025-05-16 | 14 071 | 7 944 | 2020-11-11 | ✓ |
  | HAEDAL    | 1 |  2 554 | 2025-05-01 |  2 431 | 2 554 | 2025-04-30 | ✓ |
  | VELODROME | 1 |  3 384 | 2024-12-13 |  3 385 | 3 383 | 2024-12-13 | ✓ |
  | ANKR      | 2 | 11 889 | 2021-01-26 | 15 203 | 6 290 | 2021-01-25 | ✓ |
  | MTL       | 2 | 11 505 | 2021-03-31 | 15 019 | 5 757 | 2021-03-30 | ✓ |
  | CYBER     | 3 |  6 265 | 2023-08-21 |  6 301 | 6 147 | 2023-08-16 | ✓ |
  | LPT       | 3 | 10 156 | 2021-11-11 | 11 157 | 8 965 | 2021-11-11 | ✓ |
  | BIGTIME   | 4 |  5 953 | 2023-10-12 |  2 671 | 5 951 | 2023-10-12 | ✓ |
  | BR        | 4 |  2 797 | 2025-03-21 |      0 | 2 795 | 2025-03-21 | — |
  | BAN       | 5 |  3 536 | 2024-11-18 |      0 | 3 554 | 2024-11-18 | — |
  | NMR       | 5 |  6 625 | 2023-06-22 | 12 847 | 6 316 | 2023-06-21 | ✓ |
  | HMSTR     | 6 |  3 853 | 2024-09-26 |  3 853 | 3 851 | 2024-09-26 | ✓ |
  | MOODENG   | 6 |  3 680 | 2024-10-25 |      0 | 3 678 | 2024-10-25 | — |
  | APE       | 7 |  9 397 | 2022-03-17 |  9 397 | 4 698 | 2022-03-17 | ✓ |
  | RIF       | 8 |  5 899 | 2023-10-21 | 12 001 | 5 943 | 2023-05-09 | ✓ |
  | HYPE      | 9 |  2 378 | 2025-05-30 |      0 | 2 376 | 2025-05-30 | — |

- **Vues** (chaque répertoire → une vue auto-découverte) : `binance_tail_klines_4h` (perp),
  `binance_tail_spot_klines_4h`, `binance_tail_funding_native`, `binance_tail_funding_on_4h`
  (perp OHLCV **+ funding causal** sur la grille 4h → volume ET funding_rate dans une seule vue),
  `binance_tail_universe` (le manifeste de sélection : base, décile, quote_volume, onboard, has_spot).
  Symbole = **slash `BASE/USDT`** (contrat de join carry). `exchange` = `binanceusdm` (perp) /
  `binance` (spot).
- **PIÈGE relisting** `[VÉRIFIÉ]` : certains symboles **délistés-puis-relistés** (CVC, BR, …) ont des
  **klines perp courtes** (depuis le relisting, ex. CVC 2025-05) alors que le **funding natif** remonte
  loin (CVC 2020-11). Le filtre `≥1 an` porte sur `onboardDate` ; la fenêtre carry *exploitable* est
  bornée par la jambe la plus courte (les klines perp). β doit lire la profondeur **par leg**, pas par
  funding seul.

### B. Les marchés exotiques Hyperliquid — `hyperliquid_exotic_*`  `[VÉRIFIÉ API publique]`
- **Source** : `POST https://api.hyperliquid.xyz/info` (no key, no KYC, FR OK, read-only strict —
  **jamais d'ordre**). Code : `src/hermes/datalake/sources/hyperliquid_exotic.py` (réutilise
  `_make_hl_poster`, les paginators backward-klines / forward-funding et `attach_funding_to_bars` —
  jamais forkés). **DISTINCT** de `src/hermes/execution/hyperliquid.py` (adaptateur d'exécution live =
  porte de merge humaine, non touché).
- **`hyperliquid_exotic_ctx`** — **snapshot de liquidité de TOUT l'univers exotique**
  `[VÉRIFIÉ — metaAndAssetCtxs]` : **207 marchés exotiques** (hors les 23), schéma
  `[name, dayNtlVlm, openInterest, markPx, oi_usd, funding, premium, midPx, isDelisted, maxLeverage]`.
  C'est l'artefact **survivorship + distribution de liquidité** (cheap, un appel). Symbole = **bare**.
- **`hyperliquid_exotic_funding_1h`** (funding horaire) / **`hyperliquid_exotic_klines`** (4h) /
  **`hyperliquid_exotic_funding_on_4h`** (funding horaire épinglé causalement sur les barres 4h →
  volume + funding_rate, recette H1 directe) — pour la **tranche sélectionnée** (mêmes politique de
  sélection + filtre ≥1 an, profondeur d'inception **sondée à l'ingestion**).
- **Univers exotique** `[VÉRIFIÉ]` : 207 marchés ; **55 délistés** (`isDelisted=True`) **présents dans
  le snapshot** ; **152 vivants** (non-délistés, volume>0). Distribution dollar-volume des vivants :
  86 @ $100k-1M · 32 @ $1-10M · 24 @ <$100k · 9 @ $10-100M · 1 @ ≥$100M (HYPE). La zone H1/H2 « sweet
  spot » candidate = la bande $100k-$10M (~118 marchés).
- **Tranche exotique sélectionnée (18 marchés, déciles de `dayNtlVlm`)** `[VÉRIFIÉ — requête lake]` :

  | name | déc | dayNtlVlm $ | funding rows | funding début | klines 4h |
  |------|----:|------------:|-------------:|--------------|----------:|
  | TRB      | 0 |        49 323 | 24 419 | 2023-09-17 | 5 000 |
  | XAI      | 0 |        50 394 | 21 524 | 2024-01-15 | 5 000 |
  | MINA     | 0 |        53 026 | 23 519 | 2023-10-24 | 5 000 |
  | AIXBT    | 1 |       101 059 | 13 090 | 2025-01-01 | 3 342 |
  | ME       | 1 |       104 007 | 13 608 | 2024-12-10 | 3 403 |
  | MERL     | 1 |       107 527 | 19 151 | 2024-04-23 | 4 789 |
  | COMP     | 1 |       108 735 | 25 885 | 2023-07-18 | 5 001 |
  | GALA     | 2 |       131 726 | 22 506 | 2023-12-05 | 5 000 |
  | POLYX    | 2 |       131 871 | 23 519 | 2023-10-24 | 5 000 |
  | GRIFFAIN | 3 |       178 157 | 12 945 | 2025-01-07 | 3 265 |
  | ALT      | 3 |       180 190 | 21 287 | 2024-01-25 | 5 000 |
  | POL      | 4 |       239 578 | 15 632 | 2024-09-17 | 3 931 |
  | CAKE     | 4 |       241 254 | 21 911 | 2023-12-30 | 5 000 |
  | kSHIB    | 5 |       384 461 | 25 286 | 2023-08-11 | 5 001 |
  | RENDER   | 5 |       394 598 | 16 779 | 2024-07-31 | 4 226 |
  | KAITO    | 6 |       786 017 | 11 881 | 2025-02-20 | 2 971 |
  | MORPHO   | 7 |     1 450 405 | 12 702 | 2025-01-17 | 3 481 |
  | HYPE     | 9 | 512 873 408   | 13 733 | 2024-12-05 | 3 434 |

  Le décile HL = rang dans la distribution `dayNtlVlm` des **152 exotiques vivants** (snapshot ctx).
  **Plancher illiquide bien plus bas que Binance** : TRB/XAI/MINA à **~$50 k/jour** (vs $0.77 M côté
  Binance). **Span combiné des deux venues : ~$49 k → $855 M (~17 000×)** — la variance de liquidité
  cible de H1. Funding horaire profond (jusqu'à 25 885 lignes ≈ 2,9 ans) ; klines plafonnées à 5 000
  (cap `candleSnapshot`). Inception ≥1 an respectée (le plus jeune = KAITO 2025-02 ≈ 1,4 an).
  Le funding `funding_on_4h` épingle le taux horaire le plus récent ≤ open de barre (annualiser ×8760).

---

## 2. DELIVERABLE C — la métrique de liquidité (l'axe H1)  `[VÉRIFIÉ — testé]`

`src/hermes/datalake/liquidity.py` rend **calculable** la régression funding-APR vs liquidité, et
**rien de plus** (pas de régression, pas de stratégie — c'est β).

- **La métrique** : **dollar-notional = `volume * close`** par barre. Le `volume` base-actif existe
  dans toute vue klines mais n'est pas comparable entre marchés (1 BTC ≠ 1000 DOGE) ; ×close donne un
  notionnel USD **comparable**, par barre, **causal** (row-local). C'est la **seule** métrique de
  liquidité proprement acquérable par-barre sur **les deux** venues — CCXT et les candles HL **jettent**
  le quote-volume et le trade-count (limite honnête).
- **`attach_dollar_volume(frame)`** → ajoute `dollar_volume` (row-local, pas de look-ahead possible).
- **`attach_trailing_liquidity(frame, lookback_bars)`** → médiane glissante **backward** par symbole
  (un régresseur lissé) — **strictement causale**, c'est la cible du **test L0 #1**.
- **`funding_vs_liquidity[_from_lake]`** → la recette H1 : `[ts, symbol, funding_apr, dollar_volume]`,
  funding annualisé sur la base APR commune (`carry_join.annualize_funding` ; HL 1h ×8760, Binance 8h
  ×1095, 4h ×2190), garde-fou de format symbole (`assert_joinable`).
- **Bonus HL** : `dayNtlVlm` / `openInterest` / **`impactPxs`** (proxy spread) sont dans `*_ctx` — plus
  riches mais **snapshot-only** (pas une histoire par-barre).

---

## 3. SURVIVORSHIP — le PIÈGE #1 de β (section critique)

Le carry mesuré **uniquement sur les survivants** est **massivement sur-estimé** : les délistings et
rug-pulls sont **précisément les cas où le carry s'est retourné en catastrophe**. β doit borner ses
conclusions au **survivor-set** et le savoir.

- **Binance — survivorship SÉVÈRE** `[VÉRIFIÉ]` : `/fapi/v1/exchangeInfo` ne liste que les contrats
  **TRADING actuels**. Les perps **délistés sont ABSENTS** et **non-énumérables** — leurs klines
  restent récupérables *par symbole connu* (sondé : FTT/LUNA/SRM/HNT répondent), mais on n'a **pas de
  liste** des symboles historiquement listés-puis-délistés. ⇒ La tranche Binance tail est un
  **SURVIVOR-SET pur**. Pour une couverture sans survivorship il faudrait une **liste externe**
  d'historique de listing (vendeur tiers) — non acquise.
- **Hyperliquid — survivorship OBSERVABLE (plus clément)** `[VÉRIFIÉ]` : HL **garde les marchés
  délistés dans `metaAndAssetCtxs`** avec `isDelisted=True`. **55 des 207 exotiques sont délistés et
  PRÉSENTS dans `hyperliquid_exotic_ctx`** → la **taille du trou de survivorship est mesurable** côté
  HL (≈27 % de mortalité de l'univers exotique). La tranche *tradeable* exclut les délistés (non
  forward-tradeables), mais le snapshot rend le biais **quantifiable** — un levier que Binance n'offre
  pas.
- **Conséquence pour β** : (a) borner toute conclusion H1/H2 au survivor-set ; (b) utiliser le 27 % de
  mortalité HL comme **borne basse** de l'effet de survivorship (Binance est probablement pire) ;
  (c) traiter un carry « propre » sur les survivants comme un **plafond optimiste**, pas un central.

---

## 4. CAVEATS (à ne pas sur-interpréter)

- **Profondeur HL exotique = STRONGLY FORWARD** : les exotiques sont plus jeunes que les majors ;
  l'inception funding est sondée à l'ingestion. Beaucoup de marchés du bas-décile sont récents → le
  backtest est court. Ne pas tirer de conclusion historique profonde. `[VÉRIFIÉ]`
- **Liquidité de SÉLECTION ≠ liquidité par-barre** : la sélection stratifie sur un **snapshot**
  (24h quote-volume Binance / `dayNtlVlm` HL). La métrique H1 par-barre (`volume*close`) est calculée
  **séparément** et est la grandeur à régresser. Un marché peut changer de décile dans le temps.
- **`volume` base-actif uniquement** : CCXT/HL ne donnent ni quote-volume natif ni nombre de trades ni
  direction du flux taker → `dollar_volume` est un proxy de notionnel, pas de la profondeur de carnet.
  Le **spread** n'est dispo que via `impactPxs` HL (snapshot). `[VÉRIFIÉ — grounding]`
- **Relisting → klines perp courtes** (cf. §1.A) : lire la profondeur **par leg**.
- **Liquidité de sélection non-PIT** : le snapshot de liquidité est *as-of l'ingestion*. Pour H1, la
  liquidité par-barre est causale ; pour la *sélection* elle ne l'est pas (et n'a pas à l'être — c'est
  un choix d'univers, documenté).

---

## 5. RECETTES DE JOIN (les formats EXACTS à utiliser)

> Garde-fous réutilisés : `hermes.research.s10_provenance`
> (`detect_symbol_format` / `validate_symbol_join` / `normalize_symbol` / `assert_joinable`).
> Un join inter-format donne **0 ligne en silence** — valider/normaliser AVANT de joindre.

### (a) H1 — funding-APR vs liquidité (l'axe décisif de β)  `[VÉRIFIÉ — testé sur le lake]`
```python
import duckdb
from hermes.datalake import liquidity as liq
con = duckdb.connect("data/hermes_datalake.duckdb", read_only=True)

# Binance tail (8h funding) — une seule vue porte volume ET funding_rate sur la grille 4h :
h1 = liq.funding_vs_liquidity_from_lake(
    con, view="binance_tail_funding_on_4h", symbol="ANKR/USDT", funding_interval_hours=8.0)
#   -> [ts, symbol, funding_apr, dollar_volume, ...]  → régresser funding_apr ~ dollar_volume (H1)

# HL exotic (funding horaire) — interval_hours=1.0 :
h1_hl = liq.funding_vs_liquidity_from_lake(
    con, view="hyperliquid_exotic_funding_on_4h", symbol="BIGTIME", funding_interval_hours=1.0)
```
La distribution complète de liquidité (pour borner la régression et le survivorship) est dans
`hyperliquid_exotic_ctx` (207 marchés, délistés inclus) et `binance_tail_universe` (la tranche + décile).

### (b) H2 / carry net sur un marché de la queue (la mécanique α, INCHANGÉE)  `[VÉRIFIÉ — testé]`
La machinerie carry de la Boucle α est **agnostique au symbole**. Cash-and-carry sur un marché tail
avec jambe spot (slash) :
```python
from hermes.datalake import carry_join as cj
aligned = cj.align_cash_and_carry(spot_tail, perp_funding_tail)   # spot ⋈ perp+funding (ts, symbol)
aligned = cj.carry_returns(aligned)                               # spot_ret / perp_ret causal
pnl = cj.delta_neutral_pnl(spot_ret=aligned.spot_ret, perp_ret=aligned.perp_ret,
                           funding_collected=aligned.funding_rate)
```
Cross-venue (perp-only tail vs HL exotic) via `cj.align_cross_venue` (normalise les deux en bare).
**Extension nécessaire (NON codée ici — c'est β)** : les conveniences `*_from_lake` de `carry_join`
lisent les **noms de vue fixes** des majors (`binance_spot_klines_4h`, `binance_funding_on_4h`, …) ;
pour pointer sur les vues `*_tail` / `*_exotic` il faut leur passer un **paramètre de nom de vue**
(une ligne). Les fonctions *pures* (`align_cash_and_carry`, `cash_and_carry_stream`, `build_book`)
tournent déjà sur n'importe quel marché — prouvé en test (`test_carry_mechanic_runs_on_a_tail_market`).

---

## 6. GOUVERNANCE (S10 propriétaire — `governance/acquisition_map.json`)

Deux sources ajoutées (statut `free`), via le **constant code-as-truth** `DEFAULT_ACQUISITION_MAP`
puis régénération du snapshot (`test_governance_snapshots_are_synced_with_embedded_defaults` PASS) :
- **« Binance USDT-M tail (perps beyond the 23) »** — `$0` public (exchangeInfo/klines/fundingRate/
  ticker) → `binance_tail_*`. Note survivorship #1 incluse.
- **« Hyperliquid exotic (/info full universe) »** — `$0` public `POST /info` → `hyperliquid_exotic_*`.
  Note survivorship-observable incluse.

**`governance/trials_by_book.json` n'est PAS touché** — une acquisition de données n'est pas un essai
(compte canonique crypto **1900** inchangé). `[VÉRIFIÉ — §8 du mandat + git status]`

---

## 7. TESTS (TDD, rouge d'abord ; aucun réseau live dans les unitaires)

| Fichier | Couvre |
|---------|--------|
| `tests/test_datalake_tail_select.py` | le sélecteur stratifié pur : sur-échantillonne l'illiquide (~40 % bas-déciles), couvre tout le spectre, déterministe, sans doublon |
| `tests/test_datalake_liquidity.py` | **L0 leak-injection = test #1** (`assert_no_lookahead` passe la liquidité trailing causale, attrape `shift(-1)`), dollar_volume=vol×close, recette H1, garde-fou format symbole, **carry mécanique sur un marché TAIL** (vues in-memory) |
| `tests/test_datalake_binance_tail.py` | discover (USDT/PERPETUAL/TRADING/≥1 an/hors-23), format slash+api, map liquidité ticker, sélection drop-zéro + stratifiée, frame d'univers + has_spot |
| `tests/test_datalake_hyperliquid_exotic.py` | parse ctx (dayNtlVlm/oi_usd/isDelisted), sélection hors-23 + hors-délistés + hors-zéro-vol, filtre profondeur via inception, snapshot survivorship-visible, bare |
| `tests/test_datalake_s0_beta_coverage.py` | dépths réelles du lake (skip si lake absent) : **additivité** (les 23 majors intacts), tail présent + déciles illiquides couverts, funding_on_4h porte volume+funding, **recette H1 sur un tail**, ctx snapshot survivorship, exotiques bare |

Tout vert ; `ruff` clean ; **0 régression** sur la suite (hors 2 cassures **pré-existantes**
d'environnement : `fxgate` gitignoré [4 modules S6] + `openpyxl` absent [1 test gold xlsx], sans
rapport avec S0-β).

---

## 8. FRONTIÈRE (non franchie)

Aucun backtest de stratégie · aucun câblage live / exécution / ordre · aucun bump du registre de
déflation (crypto reste **1900** ; seul `acquisition_map.json` est mis à jour). La recherche de la
survie du carry sur l'illiquide (H1) et de sa survie à la queue brutale (H2) = **Boucle β**, mandat
suivant. Le câblage live = **porte de merge humaine** (Malachie). La machinerie datalake +
`carry_join`/`carry_harvest` (α) est **réutilisée, rien reconstruit**.
