# STREAM_COMMON — Constitution partagée des 10 streams Hermès
### À lire EN PREMIER par chaque session Claude Code, avant le prompt de stream.

> Ce fichier est la base commune. Chaque `STREAM_0N.md` y renvoie. Si tu lances une session sans
> ce fichier dans le repo, le prompt de stream contient un rappel compressé des non-négociables —
> mais lis ce fichier si disponible (`docs/streams/STREAM_COMMON.md`).

---

## 0. LANCEMENT — 10 sessions parallèles, 10 branches isolées

| # | Stream | Séniorité | Branche git |
|---|--------|-----------|-------------|
| 1 | Harness livre combiné + sizing | Senior quant-dev | `hermes/s1-book-harness` |
| 2 | Positionnement & liquidations (crypto) | Senior trader+data | `hermes/s2-positioning` |
| 3 | Expansion univers + calibration | Mid | `hermes/s3-universe` |
| 4 | Surface options crypto 2.0 | Senior vol/dérivés | `hermes/s4-options` |
| 5 | Livre combiné FX/métaux (RV) | Senior portefeuille | `hermes/s5-fx-metals-rv` |
| 6 | Conditionneur régime « or » | Mid-senior macro | `hermes/s6-gold-regime` |
| 7 | Ton banque centrale par LLM | Mid + NLP | `hermes/s7-cb-tone` |
| 8 | Moteur de régime (HMM/BOCPD) | Senior ML/stats | `hermes/s8-regime-engine` |
| 9 | Orthogonalité & alloc cross-livre | Senior portefeuille | `hermes/s9-cross-book` |
| 10 | Méta-honnêteté / déflation | Head-of-research adversarial | `hermes/s10-honesty` |

**Procédure par session :** `git checkout -b <branche>` → lire ce fichier + `STREAM_0N.md` +
`HERMES_RD_PROGRAM_30x10.md` (le programme) + le dead-ends ledger + `research_fx/LAB_JOURNAL.md` →
exécuter le mandat → écrire les livrables dans `research_streams/S{n}/` → **STOP**.

**Règle de parallélisme (la plus importante) :** tu écris **uniquement** dans `research_streams/S{n}/`
et dans des **modules namespacés à ton stream**. Tu **n'édites jamais** un fichier d'un autre stream
ni le `LAB_JOURNAL.md` partagé (conflits de merge). La **porte de merge est humaine** (Malachie) ;
c'est lui qui agrège les branches. **Rien en live — shadow size 0 partout.**

---

## 1. L'ÉTAT RÉEL DU LAKE (acquis depuis la rédaction du programme)

Le Stream 0 est **bouclé et vérifié** (freeze + addendum or). Tu **ne touches aucune API externe** —
tu requêtes le **lake gelé** :

- **DuckDB** : `data/hermes_datalake.duckdb` (≈29 vues, 170 Parquet, ~552k lignes).
- **Découvre les vues d'abord** (ne devine pas les noms) :
  ```sql
  -- noms exacts des vues :
  SELECT view_name FROM duckdb_views() WHERE NOT internal ORDER BY 1;
  -- schéma d'une vue :
  DESCRIBE <view_name>;
  ```
- Carte du lake par catégorie (noms *probables* — confirme par découverte) :
  - **Positionnement/dérivés (CoinGlass, 7 paires × {`1d` all-time, `6h` 360j})** : `coinglass_top_lsr_account`,
    `coinglass_top_lsr_position`, `coinglass_global_lsr_account`, `coinglass_taker_volume`,
    `coinglass_open_interest` (+ agrégé), `coinglass_funding_rate`, `coinglass_liquidation` ;
    **options (BTC/ETH seulement)** : `coinglass_options_oi`, `coinglass_options_vol`, `coinglass_options_info`.
  - **Prix (Binance)** : `binance_klines` (7 paires, `1d`/`6h`).
  - **Vol (Deribit)** : `deribit_dvol` (BTC/ETH horaire **depuis 2021-03-24 — BACKFILLABLE**) ;
    `deribit_surface_snapshot` (**risk-reversal 25Δ + Greeks — FORWARD-LOG SEULEMENT**, 1 ligne/ccy/jour,
    démarré ~2026-06 → **PAS d'historique backtestable**).
  - **Macro (FRED)** : `fred_dgs10`, `fred_dfii10`, `fred_t10y2y`, `fred_bamlh0a0hym2` (+ `_vintage`),
    `fred_dtwexbgs`. **Vintages ALFRED = point-in-time.**
  - **On-chain (DeFiLlama)** : offre stablecoins (total/USDT/USDC) depuis 2017.
  - **Or/FX prix** : `gold` (GC=F), `silver` (SI=F), `gdx`, `dxy`, crosses FX ; flux GLD ; `etf_iau_flows` (2005→).
  - **Or WGC** : `wgc_cb_demand_gdt` (★ **fait autorité**, déclaré+non-déclaré, annuel+trimestriel depuis 2010) ;
    `wgc_gdt_balance` (contient la ligne ETF) ; `wgc_cb_changes_monthly` (**déclaré seulement**, par pays,
    **jumeaux `Turkey`/`Turkey*` — NE PAS sommer naïvement**) ; `wgc_cb_changes_annual` ;
    `wgc_cb_reported_monthly_global_proxy` ; `wgc_official_holdings_snapshot`.
  - **Discours (BIS)** : `cbspeak_*` (20 365 discours depuis 1996 ; **filtre par `speaker`** — la colonne
    `institution` est peu peuplée).

**DÉCISION DATA — RÉSOLUE :** Malachie a souscrit **CoinGlass**. Conséquences concrètes :
- **Stream 2 est PLEINEMENT DÉBLOQUÉ** : top-LSR, liquidations, OI, funding profonds sont dans le lake
  (top-LSR BTC remonte à **2020-05-16**). Le lead n°1 se **gate en entier maintenant**.
- **Amberdata n'a PAS été acheté** → le **skew 25Δ / Greeks n'est PAS backfillé** ; seul le **DVOL** l'est
  (2021+). Donc **Stream 4 : l'axe backtestable est le VRP-DVOL** ; skew & GEX sont en **collecte forward
  limitée** (chrono démarré). Stream 4 doit être honnête là-dessus.

**Asymétrie d'historique des 7 paires :** BTC/ETH/SOL profondes (top-LSR ~2020→) ; **AVAX/SUI/ARB/WIF
récentes** (listées ~2023-24, WIF la plus courte, top-LSR possiblement mince/absent). **Vérifie la date
de début réelle par paire** avant tout backtest, et reporte-la.

---

## 2. LA MACHINERIE DE VALIDATION À RÉUTILISER (ne reconstruis RIEN)

Localise et **lis** avant d'écrire le moindre gate. Tes sleeves sont jugés par cette machinerie :
- `src/hermes/validation/` : `book.py` (gate niveau-livre), `cpcv.py` (Combinatorial Purged CV),
  `pbo.py` (Probability of Backtest Overfitting), `triple_barrier.py`.
- `learning/stats.py` : **Deflated Sharpe Ratio (DSR)**.
- `risk/meta_labeling.py` ; `combination.py` (le slate + le blend) ; `research_fx/fxgate.py`.

---

## 3. LE GATE (METAPROMPT §3) — tout sleeve passe TOUS ces tests

1. **Causal & point-in-time** : lague chaque source externe à sa vraie latence de publication
   (COT +1 sem, FRED +1 j ouvré, on-chain/ETF ≥3 j, DVOL +1 bar).
2. **No-overlap** : CV purgée/embargo.
3. **Coût honnête** : crypto 15-20 bps ; FX 1.6/2.1/7.1 bps ; maker vs taker explicite.
4. **Généralise ≥2 instruments** (cross-asset = *le killer filter*). Un « win » mono-instrument = rejeté.
5. **MC random-entry / random-sign null p<0.05** — c'est un **plancher**, pas la barre.
6. **Déflaté** : DSR>0 au **vrai trial count**, **PBO<0.5**.
7. **Orthogonal** : **|ρ|<0.3** vs chaque sleeve existant, **daté/niveau-régime** (pas seulement brut).
8. **Jugé par LOO-Δ-Sharpe** (contribution au **LIVRE**), **jamais** par le Sharpe solo.

---

## 4. DOCTRINE D'HONNÊTETÉ (non-négociable)

- **« Honnêteté avant résultat positif. »** Un **négatif propre et daté est un livrable valide.**
- **« L'unité, c'est le LIVRE, pas le signal. »**
- **« Armer plus souvent ≠ baisser les seuils »** (le whipsaw est le dead-end n°1).
- **« vert-en-test ≠ vrai-en-prod »** : vérifie les chiffres de chaque sleeve **contre la réalité**, pas
  seulement par des tests verts (leçon du Stream 0 : le bug 6h et le bug dtype étaient invisibles aux tests).
- **Ne ressuscite jamais le slate mort** (`ts_mom_vol_scaled`, `breakout_vol_expansion`, `pairs_eth_btc`,
  `intraday_tod_btc`, et les morts FX/phys du ledger) comme **signaux directionnels**.
- **Jamais trader à travers un print news** (la perte $12k) ; discipline news-blackout.
- **Tout en SHADOW size 0.** Rien en live. Porte de merge humaine.
- **TDD** : test qui échoue d'abord pour toute logique non-triviale ; **fail-soft** ; idempotent.
- **Ne fabrique pas** : étiquette `[VÉRIFIÉ]`/`[PROBABLE]`/`[PRÉSUMÉ]` ; trace la provenance.

---

## 5. COORDINATION PARALLÈLE (anti-duplication, sans conflit)

**Bornes de périmètre (qui possède quoi — verbatim du programme §4) :** positionnement perp = **S2** (pas 4) ;
options/skew/GEX = **S4** (pas 2) ; univers/seuils = **S3** (pas de nouveau signal) ; relative-value FX/métaux
= **S5** ; conditionneurs = **6/7/8** (pas de signal standalone) ; corrélation cross-livre = **9** (pas de
lead-lag, mort) ; déflation/GP = **10**.

**Tu écris uniquement :**
- `research_streams/S{n}/FINDINGS.md` — verdict **par vecteur** (survit / mort / données-verrouillées) +
  preuves chiffrées + provenance.
- `research_streams/S{n}/CHECKPOINT.md` — ≤8 lignes, réécrit à chaque boucle (revue read-only de Malachie).
- du **code namespacé** : `src/hermes/research/s{n}_*.py` (+ tests `tests/research/test_s{n}_*.py`), ou le
  module exact que le programme nomme pour ton stream.

**Dépendances cross-stream** (S1 harness, conditionneurs 6/7/8, matrice S9) : résolues au **merge humain**.
En parallèle : (a) gate tes sleeves avec la machinerie **déjà committée** (`book.py` + `cpcv`/`pbo`/`DSR`) ;
(b) valide ton **conditionneur en isolation** via shuffle-label null contre un sleeve **accessible**
(`funding_squeeze_continuation` = S2, présent dans le code committé) ; (c) re-gate contre les artefacts
mergés à l'intégration. **Ne hard-dépends jamais du code non-committé d'un autre stream.**

---

## 6. CRITÈRES DE SUCCÈS (la définition honnête de « gagné »)

Succès si, en 2-3 jours/équipe, on obtient *au moins un* de :
1. **Crypto** : poids mort retiré (S1 dé-pondéré/sorti, S3 maker-only ou sorti) **et** ≥1 *nouveau* sleeve
   orthogonal confirmé (candidat n°1 = `positioning_top_lsr_contrarian`) **et** univers étendu où S2
   généralise honnêtement → *le livre crypto armé plus souvent, sans baisser la barre*.
2. **FX/métaux** : un **livre combiné** de ≥2-3 sleeves market-neutral/RV dont le *combiné* passe le gate
   complet → *un second livre qui diversifie l'entreprise*.
3. **Infra** : harness livre combiné + allocateur HRP/cross-livre + moteur de régime + couche sizing
   **opérationnels et trustés** → *la machine Renaissance, pas un signal*.

Un **négatif honnête est tout aussi valide** : si une frontière est épuisée, écris *précisément quelle
donnée/structure manquerait*. **On ne déclare PAS** victoire sur un survivant solo-Sharpe, ni défaite parce
qu'une frontière est morte.
