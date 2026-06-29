# COVERAGE — Stream 0-α (acquisition pour la Boucle α / Carry-Harvest)

> Branche `round3/s0-alpha`. Données + code de source + mécanique de calcul **seulement** —
> rien live, aucun essai, aucun backtest de stratégie (la récolte se mesure au mandat suivant).
> Étiquettes : `[VÉRIFIÉ]` lu/exécuté · `[PROBABLE]` inférence · `[PRÉSUMÉ]` non confirmé.

Ce Stream-0-α ajoute la **jambe spot** du cash-and-carry et **Hyperliquid** (funding horaire +
klines + univers) au data lake gelé, et fournit la **mécanique de calcul** du P&L delta-neutre
(`datalake/carry_join.py`) — sans calculer aucune stratégie.

---

## 1. CE QUI EST INGÉRÉ (vues additives au lake gelé)

Toutes les nouvelles vues sont **additives** : les vues perp / funding / multi-venue existantes
ne sont **pas touchées**. `ts` = epoch **millisecondes** int64 partout (convention du lake).

### A. `binance_spot_klines_4h` — la jambe SPOT du cash-and-carry  `[VÉRIFIÉ]`
- **Source** : Binance SPOT, klines 4h (`ccxt.binance()` défaut = spot ; host public, **joignable
  depuis la France, pas de géo-blocage**). Code : `src/hermes/datalake/sources/binance_spot.py`.
- **Univers** : les **23 perps** exactement (mirroir de `binance_klines_4h`), même ordre.
- **Schéma** : `[ts, open, high, low, close, volume, symbol, exchange, interval]`,
  `symbol` = **slash `BTC/USDT`** (identique au perp → join sans perte), `exchange='binance'`
  (la venue SPOT, distincte de `binanceusdm` = perp), `interval='4h'`.
- **Profondeur (316 448 lignes, 23 symboles, max depth `since=0`)** `[VÉRIFIÉ — requête lake]` :

  | coin | bars 4h | début | coin | bars 4h | début | coin | bars 4h | début |
  |------|--------:|-------|------|--------:|-------|------|--------:|-------|
  | BTC  | 19 417 | 2017-08-17 | DOGE | 15 307 | 2019-07-05 | NEAR | 12 509 | 2020-10-14 |
  | ETH  | 19 417 | 2017-08-17 | BCH  | 14 433 | 2019-11-28 | FIL  | 12 500 | 2020-10-15 |
  | BNB  | 18 932 | 2017-11-06 | SOL  | 12 893 | 2020-08-11 | INJ  | 12 467 | 2020-10-21 |
  | LTC  | 18 710 | 2017-12-13 | DOT  | 12 847 | 2020-08-18 | OP   |  8 938 | 2022-06-01 |
  | ADA  | 17 966 | 2018-04-17 | UNI  | 12 672 | 2020-09-17 | APT  |  8 100 | 2022-10-19 |
  | XRP  | 17 863 | 2018-05-04 | AVAX | 12 641 | 2020-09-22 | ARB  |  7 167 | 2023-03-23 |
  | TRX  | 17 635 | 2018-06-11 | LINK | 16 325 | 2019-01-16 | SUI  |  6 921 | 2023-05-03 |
  | ATOM | 15 709 | 2019-04-29 |      |        |            | WIF  |  5 079 | 2024-03-05 |

  Fin uniforme : **2026-06-29**. **Plancher** = WIF 5 079 bars (~2,3 ans) ; tous ≥ 4 000 (≥2 ans).
- **Perps sans spot équivalent** : **AUCUN** — les 23 ont un spot Binance USDT actif `[VÉRIFIÉ]`.

### B. Hyperliquid — cross-venue + funding horaire  `[VÉRIFIÉ API publique]`
- **Source** : API publique `POST https://api.hyperliquid.xyz/info` (**no key, no KYC, FR OK pour
  les données**). Read-only strict — **jamais d'ordre**. Code :
  `src/hermes/datalake/sources/hyperliquid.py`.
  **DISTINCT** de `src/hermes/execution/hyperliquid.py` (l'adaptateur d'exécution live = hors-scope,
  porte de merge humaine — non touché).
- **`hyperliquid_funding_1h`** — funding **horaire** réglé. HL paie le funding chaque heure =
  **1/8 du taux 8h** calculé ; le `funding_rate` stocké est donc le **taux horaire**. Schéma
  `[ts, funding_rate, premium, symbol]`, `symbol` = **bare `BTC`** (format natif HL).
- **`hyperliquid_klines`** — OHLCV perp 4h (aligné sur la grille carry). Schéma OHLCV + tags,
  `exchange='hyperliquid'`, `symbol` bare.
- **Univers HL** : **230 marchés** au total `[VÉRIFIÉ — metaAndAssetCtxs]` ; **les 23 bases Binance
  sont TOUTES présentes sur HL** (overlap 23/23). HIP-3 (builder-deployed) apparaît dans le même
  `metaAndAssetCtxs` si listé `[PROBABLE]` — l'ingestion logge l'univers complet pour visibilité.
- **Profondeur (réelle, ingérée — les 23 coins)** `[VÉRIFIÉ — requête lake]` :
  - **`hyperliquid_funding_1h`** : 23 coins, **596 126 lignes horaires**, fin 2026-06-29.
    Inception funding : 15 coins au **2023-05-12** (set profond HL : BTC, ETH, SOL, AVAX, SUI, ARB,
    ATOM, BNB, DOGE, INJ, LTC, OP…), les autres entre 2023-06 et 2023-12 (XRP 2023-06-18,
    APT/BCH 2023-06-30, TRX 2023-08-06, UNI 2023-08-11, DOT 2023-09-12, ADA 2023-10-22,
    NEAR 2023-11-01, FIL 2023-11-14, **WIF 2023-12-22** = le plus court). ~22k-27k lignes/coin
    (~2,6-3 ans horaire).
  - **`hyperliquid_klines`** (4h) : 23 coins, **115 012 lignes**, inception **uniforme 2024-03-18**
    (la portée de `candleSnapshot`), ~5k bars 4h/coin → ~2,3 ans.
  Tous les 23 funding sondent ≥ 22k lignes ; le funding (3 ans) est plus profond que les klines
  (2,3 ans) sur HL.

---

## 2. CAVEATS (à ne pas sur-interpréter)

- **Hyperliquid est COURT vs Binance (7 ans).** Funding HL ~3 ans (2023-05), klines ~2,3 ans
  (2024-03). → **axe PRINCIPALEMENT FORWARD** + backtest régime-récent seulement. Ne pas tirer de
  conclusion historique profonde d'un backtest HL court. `[VÉRIFIÉ]`
- **Inception discrepancy résolue par sondage** : le mandat estimait HL ~nov-2024 (~1,5 an) ; la
  recherche secondaire suggérait ~fév-2023 `[PROBABLE]`. Le **réel sondé** : funding 2023-05-12,
  klines 2024-03-18. C'est ce réel qui fait foi (manifest `earliest_ts` par coin).
- **`candleSnapshot` est END-ANCHORED** (cap 5000, ancré sur `endTime`, ignore un `startTime`
  lointain) `[VÉRIFIÉ — probe]`. Le paginator HL klines **walk BACKWARD** en conséquence ; un
  forward-walk naïf tronquerait silencieusement l'historique ancien (bug attrapé en grounding,
  régression verrouillée par test).
- **Rate limit HL** : 1200 weight/min/IP. Le throttle est **par requête** dans le poster (pas
  par-coin) + backoff 429 — sinon la pagination funding (~54 pages/coin) burst → 429. `[VÉRIFIÉ]`
- **Profondeur spot variable** : majors ~9 ans, récents (ARB/SUI/WIF) ~2-3 ans. Le harness doit
  juger sur les semaines co-actives, pas sur la grille zéro-remplie.

---

## 3. RECETTES DE JOIN (les formats EXACTS à utiliser)

> Garde-fous réutilisés : `hermes.research.s10_provenance`
> (`detect_symbol_format` / `validate_symbol_join` / `normalize_symbol` / `assert_joinable`).
> Un join inter-format donne **0 ligne en silence** — valider/normaliser AVANT de joindre.

### (a) Cash-and-carry : `spot + perp + funding` → P&L delta-neutre  `[VÉRIFIÉ — testé sur le lake]`
Le perp + funding réglé causal vivent **déjà** dans `binance_funding_on_4h` (perp OHLCV + funding
attaché sans look-ahead). La jambe spot est le seul manquant. Tout est en **slash `BTC/USDT`** :

```python
import duckdb
from hermes.datalake import carry_join as cj
con = duckdb.connect("data/hermes_datalake.duckdb", read_only=True)

aligned = cj.cash_and_carry_from_lake(con, symbol="BTC/USDT")
#   -> colonnes [ts, symbol, spot_close, perp_close, funding_rate], join sur (ts, symbol)
aligned = cj.carry_returns(aligned)                  # ajoute spot_ret / perp_ret (pct_change causal)
pnl_dn = cj.delta_neutral_pnl(                       # l'IDENTITÉ du mandat (pas une stratégie)
    spot_ret=aligned["spot_ret"], perp_ret=aligned["perp_ret"],
    funding_collected=aligned["funding_rate"], costs=0.0,
)   # pnl_dn = spot_ret + (-perp_ret) + funding_collected - costs
```
**Convention de signe (documentée, overridable au merge)** : position = **long spot + short perp**.
Quand `funding_rate > 0` le **short reçoit** le funding → `funding_collected = +funding_rate` par bar
réglé. Inverser le signe si on modélise la jambe opposée.

### (b) Cross-venue : `funding_binance` vs `funding_hyperliquid` → différentiel récoltable  `[VÉRIFIÉ]`
Binance funding natif = slash `BTC/USDT` ; HL = bare `BTC`. On normalise **les deux en bare**, on
annualise sur une **base APR commune**, et on aligne **causalement** (backward as-of) :

```python
out = cj.cross_venue_from_lake(con, "BTC", binance_interval_hours=8)
#   -> [ts, ..., apr_binance, apr_hl, apr_diff]
#   apr = rate * 8760 / interval_hours  (HL horaire ->x8760 ; Binance 8h ->x1095 ; 4h ->x2190)
#   apr_diff = apr_binance - apr_hl  -> short la venue à funding haut, long la venue à funding bas
#                                       (les DEUX jambes en perp -> pas de spot requis)
```
`interval_hours` Binance par coin via `binance_funding_info` (`funding_interval_hours_for`) :
8h par défaut, 4h pour certains majors (BTC/SOL/ETH ont migré).

---

## 4. GOUVERNANCE (S10 propriétaire — `governance/acquisition_map.json`)

Deux sources ajoutées (statut `free`), via le **constant code-as-truth** `DEFAULT_ACQUISITION_MAP`
puis régénération du snapshot (`test_governance_snapshots_are_synced_with_embedded_defaults` PASS) :
- **« Binance SPOT (klines) »** — `$0` public `/api/v3/klines` → `binance_spot_klines_4h`.
- **« Hyperliquid (public /info) »** — `$0` public `POST /info`, no key/KYC → `hyperliquid_funding_1h`
  / `hyperliquid_klines`.

**`governance/trials_by_book.json` n'est PAS touché** — une acquisition de données n'est pas un essai
(compte canonique crypto **1892** inchangé). `[VÉRIFIÉ — §7 du mandat]`

---

## 5. TESTS (TDD, rouge d'abord ; aucun réseau live dans les unitaires)

| Fichier | Couvre |
|---------|--------|
| `tests/test_datalake_binance_spot.py` | univers=23, schéma/dtypes, paginator forward (client injecté), symbol-join vs perp |
| `tests/test_datalake_hyperliquid.py` | normalize funding/klines, parse univers, paginator **backward** klines (END-anchored), forward funding, **hl_request throttle + backoff 429**, symbol bare |
| `tests/test_datalake_carry_join.py` | **leak-injection = test #1** (`assert_no_lookahead` passe le causal, attrape `shift(-1)`), align cash-and-carry, identité P&L, annualize, cross-venue causal, conveniences lake (DuckDB in-memory) |
| `tests/test_datalake_s0_alpha_coverage.py` | dépths réelles du lake (skip si lake absent) : spot 23 pairs / floor, **join lossless prouvé**, vues HL bare/présentes, différentiel cross-venue calculable |

Tout vert ; `ruff` clean ; **0 régression** sur la suite (hors 2 cassures pré-existantes
d'environnement : `fxgate` gitignoré + `openpyxl` absent, sans rapport avec S0-α).

---

## 6. FRONTIÈRE (non franchie)

Aucun backtest de stratégie · aucun câblage live / exécution / ordre · aucun bump du registre de
déflation. La recherche de la récolte (capacité, survie en régime, risque de queue) = **Boucle α**,
mandat suivant. Le câblage live du spot/Hyperliquid = **porte de merge humaine** (Malachie).
