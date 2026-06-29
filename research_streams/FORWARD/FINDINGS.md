# STREAM FORWARD — Infra de collecte forward + shadow-logging size-0

> **Branche** `round2/forward-shadow` (de `main` post-S0) · **Pré-enregistré le 2026-06-29**
> **Doctrine** : RIEN LIVE. Read-only sur les feeds, write-only sur `data/forward/`. Le câblage
> live (market_analyst / pulse / config / allowed_skills) est la **porte de merge humaine** — non touché.
> Étiquettes : `[VÉRIFIÉ]` (lu/exécuté directement), `[PROBABLE]` (inférence haute confiance),
> `[PRÉSUMÉ]` (non confirmé sur cette branche).

Le reframe stratégique (AM-5) : les backtests sont contaminés par la bull-beta ; le **régime
bear/range 2026 est le labo naturel pour la falsifier**. Le livrable bascule de « trouver un edge en
backtest » vers **« accumuler de l'OOS forward propre + bâtir l'infra de collecte »**. Deux livrables :
(A) un cron de collecte forward unifié, (B) un harness de shadow-logging size-0.

---

## 0. Provenance & garde-fous de lecture

- `research_streams/S2/FINDINGS.md` et `src/hermes/research/s2_xsec.py` **ne sont PAS sur cette branche**
  (branchée de `main` post-S0). La cellule XS est reprise **verbatim du mandat round-2**, qui encode le
  résultat empirique S2 v-next. `[VÉRIFIÉ]` pour la cellule (mandat) ; `[PRÉSUMÉ]` pour le chiffre
  historique `+0.58` (mémoire S2 v-next, non re-dérivable ici) → **re-établi par le sanity-check ci-dessous**.
- La règle live `funding_squeeze_continuation` est lue **directement** dans `src/hermes/research/signals.py`
  (`_FUNDC_WIN=180`, `_FUNDC_Z=2.0`, `_FUNDC_HOLD_BARS=24`, LONG-only). `[VÉRIFIÉ]`
- Lake : `binance_funding_native` (23 perps, **8h natif profond**, BTC 2019-09 → 2026-06) est la source
  funding profonde ; `coinglass_funding_rate` ne couvre que 7 symboles ; le `binance_funding` stub (2 mois)
  est à ÉVITER. `[VÉRIFIÉ]` (requête lake)

---

## A. CRON DE COLLECTE FORWARD UNIFIÉ — `python -m hermes.forward.run`

Un job **idempotent** (re-run le même jour UTC écrase, ne duplique pas — délégué à
`s4_options.dedup_forward_log`), **fail-soft par flux** (un flux qui tombe → blocker enregistré, les
autres continuent), **PIT** (chaque snapshot horodaté à la collecte, jamais réécrit). Écrit dans
`data/forward/<flux>/log.parquet`. Code : `src/hermes/forward/` (`logio.py`, `run.py`, `collect_*.py`).

| Flux | Module | Transform PIT (testée) | Fetcher (seam read-only) | État initial `[VÉRIFIÉ]` |
|---|---|---|---|---|
| **skew** 25Δ Deribit | `collect_skew.py` | réutilise S4 `compute_surface_snapshot` (jamais de NaN fabriqué) | Deribit public `_get`/`_make_delta_fn` (réutilise S4) — **CÂBLÉ & LIVE** | 2 lignes (BTC atm_iv 43.5, ETH 55.2 ; rr25 NaN = greeks indispo, non fabriqué) |
| **comex** courbe → calendar-spreads | `collect_comex.py` | `curve_snapshot` (tri/rang/dte PIT) + `calendar_spreads` (far-near, contango/backwardation) | dated-contracts gratuits — **seam non câblé** (`NotImplementedError`, fail-soft) | 0 (blocker documenté) |
| **macro** dot-plot vs FF | `collect_macro.py` | `dot_vs_ff_gap` (SEP médian − FF-implied par horizon) | Fed SEP + CME FedWatch/ZQ — **seam non câblé** | 0 (blocker documenté) |

**COMEX — pourquoi** : S5 a nommé **deux fois** la courbe à terme métaux comme « le seul axe pouvant
casser le mur 2-D », bloqué en backtest par le **survivorship yfinance** (le continu front splice les
contrats et perd la structure par terme). L'évasion : accruer la **chaîne 8-maturités datées** vers
l'avant → en 12-18 mois, un panel de calendar-spreads PIT propre, sans survivorship. Pas de backfill.
Le transform (courbe→spreads) est pur et testé ; le fetcher réseau est un seam **interchangeable**
(Yahoo dated contracts → fichiers de settlement CME) sans toucher la logique PIT/spread. `[VÉRIFIÉ]` (tests)

**Idempotence prouvée** `[VÉRIFIÉ]` : 3 runs le même jour → skew reste à 2 lignes, funding à 1 trade.

---

## B. SHADOW-LOGGING SIZE-0

### B.1 — `funding_squeeze_continuation` (l'edge live, à trancher) — **CÂBLÉ & LIVE**

Reproduit **byte-identiquement** la règle live sur le feed Binance public (read-only, ccxt) : `|funding-z|≥2`
sur 180 barres → LONG, hold 24×4h, no-overlap. Module `shadow_funding_squeeze.py`.

- **Contrat de fidélité prouvé** `[VÉRIFIÉ]` : (a) la décision d'entrée **appelle** la fonction live
  `funding_squeeze_continuation` (jamais une copie) ; (b) le rendement des trades clos est **byte-identique**
  à `harness.evaluate_signal` (même triple-barrière, même `no_overlap`, même coût RT 20 bps) ; (c) seule
  différence assumée = **sémantique forward** : une position dont l'horizon 24-barres n'est pas écoulé (et
  sans barrière touchée) est loggée **OPEN (mark-to-market)**, jamais close prématurément — le log idempotent
  la finalise plus tard.
- **Univers** : la slate live 7-actifs (BTC/ETH/SOL/AVAX/SUI/ARB/WIF).
- **Ce n'est PAS un nouvel essai** (`is_new_trial=False`) : la cellule est déjà comptée dans le sweep
  funding de 648 (seed crypto 1869). On accrue juste de l'OOS bear propre.
- **État initial** `[VÉRIFIÉ]` : 1 trade — **WIF/USDT, entrée 2026-05-27, funding_z=13.4, hold 24×4h,
  time-exit, ret_net = −2.70 %** (un LONG **perdant** dans le régime récent). C'est exactement le signal que
  le lake gelé ne peut pas donner. *(Caveat : ce 1er trade chevauche la queue du lake ; l'OOS strictement
  propre = tout ce qui est horodaté après la collecte d'aujourd'hui.)*

### B.2 — XS funding-reversion (le lead pré-enregistré) — **PRÉ-ENREGISTRÉ, NON CÂBLÉ (fragile)**

#### Pré-registration datée (gelée le **2026-06-29**) — `src/hermes/forward/prereg.py`

| Champ | Valeur (cellule EXACTE) |
|---|---|
| Signe | **INVERSÉ** — *fade* high funding-z : **SHORT** tercile haut, **LONG** tercile bas |
| Cadence | **14 j** (bi-hebdo, lent → contrôle le turnover 2-jambes) |
| Neutralité | **dollar-neutre** (gross long = gross short) → retire la bull-beta **par construction** |
| Univers | **23 perps** (`binance_funding_native` profond) |
| Buckets | **terciles** |
| Fenêtre funding-z | **180 settlements** (8h natif ≈ 60 j) |
| Coût | 2-jambes plein turnover honnête (~40 bps/rebalance @ fee 7 + slip 3) |
| Null | MC random-**SIGN** (long-short), autocorr préservée |

> **Compte de déflation pour le VERDICT forward = 1892** = 1869 (seed crypto) + 22 (S2 v-next : 4 pré-reg +
> 18 exploratoires, à réconcilier au merge) + 1 (cette pré-reg). Le bump est fait par une **fonction
> committée + testée + idempotente** `prereg.register_trials(data_dir)` (consomme `prereg.ALL`, bump
> seulement `is_new_trial=True`, garde anti-double-comptage en lisant la provenance — car `bump()` n'en a
> pas) → `BookTrialRegistry` crypto **1869 → 1870** (source `forward_xs_funding_reversion_prereg`) ; le +22
> viendra du merge S2 v-next. `funding_squeeze` (`is_new_trial=False`) n'ajoute **aucun** essai.
> `[VÉRIFIÉ]` (test `test_register_trials_*` : 1 bump, re-run = no-op ; appelée dans `run.main()`)

#### Sanity-check de robustesse IN-SAMPLE — ⚠️ **CONTAMINÉ** (lake gelé, bull-beta incluse)

But unique : décider **wire vs ne-pas-wire**. Ce n'est **pas** un verdict (le verdict = OOS forward déflaté).
Code : `xs_reversion_sanity.py` (métriques unit-testées) + `shadow_xs_reversion.py`.

- **Reproduction** `[VÉRIFIÉ]` : cellule de base (inversée, 14j, terciles) → **ann-Sharpe net +0.457**
  (gross +0.65), cohérent avec le `+0.58` du mandat. Le signe **non-inversé** (squeeze) = **−0.842** →
  confirme que le bon signe est l'inversé (la XS funding **mean-reverse**).
- **Stabilité sous-périodes** (tiers chronologiques) : **[+1.57, −0.04, −1.27]** → edge concentré tôt,
  **négatif** dans le tiers récent.
- **Profil par année** (ann-Sharpe net) : 2020 **+1.81**, 2021 **+1.67**, 2022 −0.60, 2023 −0.83,
  2024 +0.50, 2025 −0.32, 2026 **−2.95** ; **2024+ (régime pertinent pour le forward) = −0.07**.
- **Voisinage de paramètres** (cadences 7/10/14/21 × quantiles 3/4/5) : **plage [−0.12, +0.85]** ;
  21j-terciles = **−0.12** (négatif). Sensible aux paramètres.
- **LOO instruments** : drop ETH → 0.11 (Δ −0.345), drop LTC → 0.145 ; plage Δ **[−0.345, +0.212]** →
  porté par quelques noms (ETH/LTC/LINK).

**VERDICT : FRAGILE.** Le `+0.58` est une cellule **2020-2021** : décroît à ~0/négatif depuis 2022, et le
régime pertinent-forward (2024+) est −0.07. Crucialement, la construction est **dollar-neutre** → cette
décroissance **n'est pas** un artefact de bull-beta qu'on retire ; c'est la **mort réelle** de l'edge de
réversion cross-sectionnelle post-2021. **→ Documenté, NON câblé** (`WIRE_XS_REVERSION=False` dans `run.py`).

> **Tension assumée** : la section-4 du mandat (« faire tourner en shadow tout sleeve à-bon-signe ») pousse
> à shadow-er quand même (size-0, coût ≈ 0, et l'OOS forward serait le seul juge propre). La porte est laissée
> **à un seul flag** : passer `WIRE_XS_REVERSION=True` (ou `--wire-xs-reversion`) active le collecteur
> (déjà implémenté + testé). Décision laissée au merge. `[VÉRIFIÉ]` (test `test_run.py`)

---

## C. CÂBLAGE LIVE DIFFÉRÉ (à faire au MERGE — **PAS fait ici**)

1. **Crons** : planifier `python -m hermes.forward.run` en quotidien (APScheduler `scheduler.py` /
   `datalake.run` order) dans le runtime déployé (où `/data` + réseau existent). Aujourd'hui, exécuté à la
   main pour seeder l'état initial.
2. **Fetchers seams non câblés** : COMEX (Yahoo dated contracts ou settlement CME) et macro (Fed SEP + CME
   FedWatch/ZQ). `collect_skew` + `funding_squeeze_shadow` sont **déjà câblés** (Deribit/Binance public).
3. **Promotion d'un sleeve confirmé** : si l'OOS forward (déflaté à 1892) confirme un sleeve, le câbler dans
   le runtime live est la **porte humaine** — `blend_live_shadow` (`combination.py`, `returns_by_skill={}`
   aujourd'hui), `allowed_skills` (`config/default.yaml`), `library.py`. **NON touché.** `[VÉRIFIÉ]` (off-limits)
4. **Correctif honnêteté S10** (rappel, hors-scope ici) : `newdata.py:12` + `library.py:226-241` étiquettent
   encore des sleeves rétrogradés comme « confirmés » → à corriger au merge.

---

## D. État & tests

- `data/forward/` : `skew/log.parquet` (2), `shadow/funding_squeeze_continuation/log.parquet` (1).
  *(`/data/` est gitignored → les logs s'accumulent localement/dans le runtime déployé ; le livrable Git =
  code + ce FINDINGS.)*
- **TDD** : 43 tests `tests/forward/` verts ; suite repo = 644 passed (les 4 erreurs S6 + `test_datalake_gold`
  sont **pré-existantes** : `import fxgate` / `openpyxl` absent, fichiers non touchés par ce stream). ruff
  clean sur `src/hermes/forward tests/forward`.
- Essais enregistrés au `BookTrialRegistry` via `prereg.register_trials` (testée, idempotente) : **+1
  crypto** (XS reversion pré-reg ; `funding_squeeze` n'en ajoute aucun). `[VÉRIFIÉ]`
- **Audit adversarial 5-agents** (look-ahead / PIT / read-only / déflation) : 3 UPHELD (fidélité
  funding_squeeze byte-identique, causalité + dollar-neutralité XS, confinement read-only/write-`data/forward`),
  2 REFUTED → **corrigés** : `logio` durci (garde key_cols non-vide + ts epoch-ms numérique + tri latest-ts
  déterministe, sans toucher le `dedup_forward_log` de S4) ; bump registry converti d'un script manuel en
  `register_trials` committée + testée. Fragilité latente de `s4_options.dedup_forward_log` (keep="last" sans
  tri) **signalée pour S4/merge** — non modifiée (isolation de stream).

**STOP — infra + shadow only, rien live.**
