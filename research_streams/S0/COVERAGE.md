# Stream 0 — RE-FREEZE ÉLARGI : note de couverture

**Mandat :** ajouter au lake les données qui débloquent le cœur crypto de Round 2, pour **0 $**
(ingestion + vues + tests + cette note). Aucune stratégie, aucun backtest.

**Statut :** ✅ les 4 tâches livrées. Lake materialisé localement, vérifié contre la réalité.
Code committé sur `main` ; le `.duckdb` et les `*.parquet` restent gitignorés (`/data/`) —
re-matérialiser via `python -m hermes.datalake.run --sources binance_extended,coinglass_multi`.

- **run_as_of :** 2026-06-29 (UTC)
- **vues du lake :** 29 → **35** (+6 nouvelles, listées plus bas)
- **lignes ajoutées :** klines 4h = **276 192** · funding natif = **140 896** · multi-venue ≈ **80 850**
- **convention ts :** epoch **millisecondes** (BIGINT) → `datetime64[ns]` [VÉRIFIÉ]
- **tests :** 35 nouveaux (17 + 6 offline-purs + 12 couverture skip-si-pas-de-lake) ; suite **459 passed** (hors `tests/research`)

---

## Nouvelles vues (chaque dossier `data/raw/<a>/<b>/` ⇒ une vue `<a>_<b>`)

| Vue | Source dossier | Contenu |
|---|---|---|
| `binance_klines_4h` | `binance/klines_4h/` | OHLCV 4h, profondeur max (`since=0`), 23 paires |
| `binance_funding_native` | `binance/funding_native/` | funding natif PROFOND depuis l'inception (TACHE 3) |
| `binance_funding_info` | `binance/funding_info/` | snapshot cadence funding courante (8h vs 4h) + caps |
| `binance_funding_on_4h` | `binance/funding_on_4h/` | funding settlé **collé aux barres 4h**, causal/laggé (jointure) |
| `coinglass_multi_funding_rate` | `coinglass_multi/funding_rate/` | funding cross-venue (Binance/OKX/Bybit/Bitget) |
| `coinglass_multi_open_interest` | `coinglass_multi/open_interest/` | OI cross-venue, unité USD |

---

## TACHE 1 + 2 — klines 4h + univers étendu

**23 paires retenues** (les 7 live + 16 étendues), toutes ≥ 2 ans de 4h. `since=0` ⇒ profondeur
maximale (la cap 360j du 6h n'est PAS appliquée). Date de début et profondeur **PAR PAIRE** [VÉRIFIÉ] :

| paire | klines 4h | début 4h | ans | funding natif | début funding | cadence h |
|---|---:|---|---:|---:|---|---:|
| BTC | 14 916 | 2019-09-08 | 6.81 | 7 453 | 2019-09-10 | 8 |
| ETH | 14 439 | 2019-11-27 | 6.59 | 7 219 | 2019-11-27 | 8 |
| BCH | 14 306 | 2019-12-19 | 6.53 | 7 152 | 2019-12-19 | 8 |
| XRP | 14 198 | 2020-01-06 | 6.48 | 7 099 | 2020-01-06 | 8 |
| LTC | 14 180 | 2020-01-09 | 6.47 | 7 090 | 2020-01-09 | 8 |
| TRX | 14 144 | 2020-01-15 | 6.45 | 7 072 | 2020-01-15 | 8 |
| LINK | 14 132 | 2020-01-17 | 6.45 | 7 066 | 2020-01-17 | 8 |
| ADA | 14 048 | 2020-01-31 | 6.41 | 7 059 | 2020-01-19 | 8 |
| ATOM | 14 005 | 2020-02-07 | 6.39 | 7 005 | 2020-02-06 | 8 |
| BNB | 13 988 | 2020-02-10 | 6.38 | 6 994 | 2020-02-10 | 8 |
| DOGE | 13 082 | 2020-07-10 | 5.97 | 6 541 | 2020-07-10 | 8 |
| DOT | 12 825 | 2020-08-22 | 5.85 | 6 418 | 2020-08-20 | 8 |
| SOL | 12 687 | 2020-09-14 | 5.79 | 6 420 | 2020-09-13 | 8 |
| UNI | 12 663 | 2020-09-18 | 5.78 | 6 334 | 2020-09-17 | 8 |
| AVAX | 12 633 | 2020-09-23 | 5.76 | 6 318 | 2020-09-22 | 8 |
| NEAR | 12 500 | 2020-10-15 | 5.70 | 6 252 | 2020-10-14 | 8 |
| FIL | 12 495 | 2020-10-16 | 5.70 | 6 247 | 2020-10-16 | 8 |
| OP | 8 935 | 2022-06-01 | 4.08 | 4 468 | 2022-06-01 | 8 |
| INJ | 8 476 | 2022-08-17 | 3.87 | 4 239 | 2022-08-16 | 8 |
| APT | 8 098 | 2022-10-19 | 3.69 | 4 050 | 2022-10-18 | 8 |
| ARB | 7 165 | 2023-03-23 | 3.27 | 3 583 | 2023-03-23 | 8 |
| SUI | 6 918 | 2023-05-03 | 3.16 | 3 459 | 2023-05-03 | 8 |
| WIF | 5 359 | 2024-01-18 | 2.44 | 5 358 | 2024-01-18 | **4** |

- **Les 7 live :** BTC/ETH/SOL profonds (~2019-20) ; AVAX 2020-09 ; ARB 2023-03 ; SUI 2023-05 ;
  **WIF la plus courte (2024-01, 2.44 ans)** — conforme à l'attente du mandat [VÉRIFIÉ].
- **Univers étendu retenu (16) :** LINK, DOGE, BNB, XRP, ADA, DOT, LTC, BCH, ATOM, NEAR, FIL, INJ,
  APT, OP, TRX, UNI. Tous ≥ 3.16 ans de 4h. **0 candidat écarté pour profondeur** (MATIC n'a PAS été
  inclus — renommé POL fin 2024, symbole perp instable ; remplacé par TRX/UNI, deux majors profonds).
- **Pourquoi :** S3 a maintenant **≥ 3 instruments profonds** pour le killer-filter cross-asset
  (AVAX seul avait été rejeté faute de généralisation) ; S2-xsec a la largeur pour un long-short
  market-neutral non-squelettique.

---

## TACHE 3 — funding natif PROFOND (+ le piège résolu)

**`binance_funding_native`** ≠ **`binance_funding`** — DISTINCTION CRITIQUE, documentée dans la
docstring de la vue et verrouillée par un test :

| vue | profondeur | usage |
|---|---|---|
| `binance_funding` (existante) | **~200 lignes/sym, 2026-04 →** (forward-only) | stub live ; **backtester dessus = backtester sur du vide** |
| `binance_funding_native` (NOUVELLE) | **3 459 → 7 453 lignes/sym, depuis l'inception** | l'historique backtestable profond |

Distinctness mesurée : stub **max 200**/sym vs natif **min 3 459**/sym (17×+) [VÉRIFIÉ].

### ⚠️ Le piège `/fapi/v1/fundingRate` (vert-en-test ≠ vrai-en-prod)
Le premier pull est revenu **shallow (200 lignes/sym depuis 2026-04)** — identique au stub. Cause
**vérifiée contre l'API réelle** : `startTime=0` (ou absent) déclenche un clamp « 200 plus récentes »
au lieu de partir de l'inception. C'est la **racine du `binance_funding` forward-only existant** (même
bug dans `binance.py`). **Correctif :** walk forward avec `startTime=1` (Binance le clampe ensuite à
la vraie date de listing). Régression verrouillée par un test reproduisant le clamp
(`FUNDING_DEEP_START_MS >= 1`). Mes tests passaient (le fake fetcher servait du profond) — seul le
spot-check **contre la réalité** a attrapé le bug.

### Funding collé aux barres 4h — causal, sans look-ahead
`binance_funding_on_4h` : chaque barre 4h (open-time `T`) porte le dernier funding settlé à
`fundingTime ≤ T` (lag = 0 ms ; la latence réelle est ~0, le taux settlé est publié à `fundingTime`).
Forward-fill jusqu'au print suivant. Machinerie `attach_funding_to_bars(..., lag_ms=)` testée ;
recherche voulant un embargo plus strict peut re-joindre `binance_funding_native` × `binance_klines_4h`
avec un lag plus grand.

**SPOT-CHECK vs réalité [VÉRIFIÉ] :** print BTC du **2023-06-15 08:00 UTC = +6.19 bps**. La barre
4h de 08:00 le porte (et forward-fill jusqu'à 12:00) ; la barre de **04:00 (avant settlement) porte le
print PRÉCÉDENT** (00:00 = −3.27 bps), **pas** le print de 08:00 → **aucun look-ahead**.

### Cadence variable (TACHE 3, gérée via `/fapi/v1/fundingInfo`)
Snapshot `binance_funding_info` = 678 symboles (1× 1h, 439× 4h, 238× 8h). Parmi mes 23 paires :
**seul WIF est passé en 4h** ; les 22 autres restent 8h. Le comptage de trous funding utilise la
cadence historique 8h (un perp passé 8h→4h a des prints *plus* denses, jamais un faux trou).

---

## TACHE 4 — scout multi-venue : **PROFOND ✅ → ingéré**

**Fait testé (read-only) [VÉRIFIÉ] :** le tier CoinGlass souscrit sert le funding ET l'open-interest
**multi-exchange en profondeur historique** — pas Binance-only, pas capé.

- `supported-exchange-pairs` expose **30 exchanges**. Sondage des endpoints `funding-rate/history` et
  `open-interest/history` à 1d : **OKX, Bybit, Bitget remontent jusqu'à 2020-2021** (pagination
  backward, page 1 = 1000 récentes → page 2 continue ; pas de clamp « 1000 récentes »).
- Décision (mandat : « si OUI et profond, ingère ») ⇒ **ingéré** un substrat de dispersion cross-venue :
  **{funding, OI} × {Binance, OKX, Bybit, Bitget} × {BTC, ETH, SOL, XRP, DOGE} × 1d**, toutes venues
  dans **une vue par série** (`GROUP BY exchange` en une requête).

Profondeur cross-venue (funding, début par venue, BTC repère) [VÉRIFIÉ] :
Binance 2020-03-30 · OKX 2020-10-30 · Bybit 2020-10-30 · Bitget 2021-08-12. 20 cellules
(exchange×coin) présentes pour chaque série. **Débloque la dispersion cross-venue de S2.3**
(auparavant data-gated « lake Binance-only » — voir [[hermes-s2-positioning]], [[hermes-s9-cross-book]]).

Note : `coinglass_multi_*` (4 venues, 5 coins, 1d) recouvre partiellement `coinglass_funding_rate`/
`coinglass_open_interest` (Binance seul, 7 paires, 1d+6h) — ce dernier reste pour back-compat ; le
nouveau est le substrat cross-venue canonique. Résolveur d'instrument perp testé (piège Bybit : le
perp `BTCUSDT` côtoie les futures datés `BTCUSDT-03JUL26`).

---

## Garde-fous respectés
- **TDD strict** : test rouge d'abord pour chaque fonction pure neuve (incl. la régression du clamp funding).
- **Idempotent** : re-run skip via `cached_pull` ; nouveaux dossiers ⇒ additif, vues existantes intactes.
- **Fail-soft** : un échec par-unité = un blocker, jamais un crash (seul blocker = forexfactory, mort connu).
- **Spot-check contre la réalité** : funding daté → bonne barre 4h (pas seulement vert-en-test).
- **Couverture** : `tests/test_datalake_s0_coverage.py` asserte la profondeur min/paire (skip si lake absent).
- **Rien en live** : aucun fichier de config/skills/agent touché ; ingestion pure.

## Pour les autres streams
- Prix 4h : `binance_klines_4h` (23 paires). Funding backtestable : **`binance_funding_native`**
  (JAMAIS `binance_funding`). Funding-par-barre prêt : `binance_funding_on_4h`.
- Dispersion cross-venue : `coinglass_multi_funding_rate` / `coinglass_multi_open_interest`.
- Re-freeze : `python -m hermes.datalake.run --sources binance_extended,coinglass_multi`.
