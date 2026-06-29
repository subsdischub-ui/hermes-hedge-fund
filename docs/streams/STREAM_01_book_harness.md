# STREAM 1 — HARNESS DU LIVRE COMBINÉ + COUCHE DE SIZING
**Séniorité : Senior quant-dev · Branche : `hermes/s1-book-harness` · Tourne EN PREMIER (fondation)**

> **LIS D'ABORD** : `docs/streams/STREAM_COMMON.md`, le programme (`HERMES_RD_PROGRAM_30x10.md` §Stream 1),
> le dead-ends ledger, `research_fx/LAB_JOURNAL.md`. Tu construis **l'instrument de mesure** que tous les
> autres streams consomment — il doit être trusté avant tout.

## MANDAT
Construire/finir le gate qui rend *tout le reste* honnête : pointer la machinerie existante
(`src/hermes/validation/book.py`, `cpcv.py`, `pbo.py`, `learning/stats.py` DSR) sur une **série de
rendements du LIVRE pondéré hétérogène**, et accepter un sleeve par sa **LOO-Δ-Sharpe** (contribution),
**jamais** par son Sharpe solo. C'est *l'expérience jamais faite* et la thèse Renaissance littérale.
Risque-données quasi nul : **pur réemploi d'infra**.

## BORNES DE PÉRIMÈTRE
Tu ne *cherches* **aucun** signal — tu construis l'**instrument de mesure** et l'**allocateur**. Les sleeves
viennent des autres streams. Tu nettoies le slate mort de `combination.py` mais **ne ré-bénis aucune jambe
morte**. Tu ne touches pas aux données brutes (Stream 0 les possède).

## INPUTS LAKE
Séries de rendements des sleeves **déjà confirmés/committés** (surtout `funding_squeeze_continuation` = S2).
Reconstruis les events live du dashboard pour la validation de re-classement. `binance_klines` pour les
rendements forward. Pas de nouvelle donnée.

## LES 3 VECTEURS

### 1.1 — Harness combiné + nettoyage du slate
- **H :** la machinerie CPCV/PBO/DSR peut juger un *livre* (pas des variantes d'une stratégie).
- **Méca :** Renaissance = beaucoup de sleeves faibles ρ≈0 → Sharpe ~1+ par combinaison ; le bon critère est
  « ce sleeve améliore-t-il le livre ? ».
- **Méthode :** implémente `combine_book(sleeve_returns: dict) -> book_returns` et
  `leave_one_out_delta_sharpe(book, sleeve)` au-dessus de `book.py`. Purge `combination.py` de
  `ts_mom_vol_scaled`, `breakout_vol_expansion`, `pairs_eth_btc`, `intraday_tod_btc` (morts). Reconstruis le
  livre **à partir des seuls sleeves confirmés/orthogonaux**.
- **KILL :** si la LOO-Δ-Sharpe ne **re-classe pas correctement** S1 (négatif, ~−0.146) vs S2 (positif, ~+0.015)
  sur les events live re-générés, **le harness est faux** — corrige-le avant tout le reste.
- **Liv. :** `combine_book()` + `leave_one_out_delta_sharpe()` branchés ; `combination.py` purgé ; **test
  anti-drift vert** + un test de **re-classement S1<0<S2 contre les events réels** (vert-en-test ≠ vrai-en-prod).

### 1.2 — Pondération HRP / inverse-vol causale + activation shadow de la couche combinaison
- **H :** à faible data, inverse-vol / HRP / 1-N battent les allocateurs covariés OOS (DeMiguel-Garlappi-Uppal).
- **Méca :** clustering par corrélation → diversification réelle, pas re-découverte de beta.
- **Méthode :** étends `causal_inverse_vol_weights()` en **HRP** (poids forward, **zéro look-ahead**). Passe la
  couche `combination.py` d'`advisory shadow` à **blend live shadow (log-only)** pour mesurer son apport réel.
- **KILL :** si HRP n'améliore pas le Sharpe OOS du livre vs inverse-vol simple sur **≥3 folds**, **garde
  inverse-vol** (parcimonie).
- **Liv. :** poids forward via HRP/inverse-vol ; couche combinaison en **blend live shadow** ; comparatif daté.

### 1.3 — Couche méta-labeling / sizing (découpler DIRECTION et TAILLE)
- **H :** la direction est résolue (sur les sleeves confirmés) ; la **taille/timing** n'a jamais été optimisée.
- **Méca :** le méta-label ne *crée* pas d'edge d'un signal à zéro-gross, mais **relève le DSR d'un edge
  réel-mais-marginal** et peut *apprendre le news-blackout comme feature de veto* ; **fractional-Kelly**
  contrôle le drawdown (full-Kelly fragile).
- **Méthode :** couche de sizing au-dessus de `risk/meta_labeling.py` + `validation/triple_barrier.py`, appliquée
  à **S2** (et tout futur sleeve confirmé). Fractional-Kelly, veto news-blackout en feature.
- **KILL :** valider sur un **instrument tenu à l'écart** — si le sizing n'améliore le Sharpe **déflaté** qu'**in-sample**,
  c'est de l'overfit de taille → rejet.
- **Liv. :** couche de sizing sur S2, **gate sur held-out instrument**.

## INVARIANTS (rappel)
Réutilise `book.py`/`cpcv.py`/`pbo.py`/`stats.py` — ne reconstruis rien. TDD (test rouge d'abord). Zéro
look-ahead dans toute pondération. Tout en shadow. **Le harness est la fondation : s'il ment, tout le
programme ment** — d'où le test de re-classement contre la réalité.

## SORTIE
`research_streams/S1/FINDINGS.md` (verdict des 3 vecteurs + le comparatif HRP/inverse-vol + preuve de
re-classement) · `research_streams/S1/CHECKPOINT.md` · code dans `src/hermes/validation/` (extensions) et
`combination.py` (purge). **STOP** — aucune recherche de signal.
