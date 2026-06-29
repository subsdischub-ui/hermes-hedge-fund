# STREAM 8 — MOTEUR DE RÉGIME (HMM / CHANGE-POINT / ML)
**Séniorité : Senior ML/stats · Branche : `hermes/s8-regime-engine` · Démarrer tôt (alimente 1/3/6)**

> **LIS D'ABORD** : `docs/streams/STREAM_COMMON.md`, le programme §Stream 8, le dead-ends ledger,
> `research_fx/LAB_JOURNAL.md`. Renaissance utilise des **HMM pour détecter les changements de régime**. Notre
> routeur actuel est volontairement trivial (Kaufman ER + hystérésis 0.3/0.4). Tu construis un *vrai* moteur,
> comme **couche de conditionnement** pour les DEUX livres.

## MANDAT
Construire un moteur de régime robuste (**HMM + BOCPD + buckets vol/macro**) comme **couche de conditionnement**
pour les deux livres, et intégrer la **segmentation CV *régime-aware*** (conditionner train/test sur vol/macro
améliore la robustesse en non-stationnarité).

## BORNES DE PÉRIMÈTRE
Tu ne crées **aucun signal directionnel** — tu produis des *états* que les Streams 1/3/6 consomment comme
conditionneurs/gates. Tu *consommes* (au merge) les features régime des Streams 6 (or) et 7 (macro) ; en
parallèle, construis le moteur sur les données du lake.

## INPUTS LAKE
`binance_klines` + vol réalisée (crypto), `gold`/`dxy` (FX/métaux), `fred_bamlh0a0hym2` (HY spread, +1 j ouvré),
`fred_t10y2y` (2s10s, +1 j). (Features régime or/macro de S6/S7 : consommées au merge ; stand-in en parallèle.)

## LES 3 VECTEURS

### 8.1 — Moteur de régime multi-méthode (HMM + BOCPD + vol/macro)
- **H :** un état de régime robuste (Hidden Markov Model + Bayesian Online Changepoint Detection + buckets
  vol/macro) sépare mieux trend/range/stress que l'ER seul.
- **Méca :** non-stationnarité = la nature des marchés ; les bons signaux sont régime-dépendants.
- **Méthode :** entraîne un HMM (états trend/range/stress) + BOCPD pour les ruptures + buckets vol/macro ; **online,
  zéro look-ahead** (estimation causale, pas de ré-étiquetage rétrospectif). Sortie = `regime_engine` (état + proba).
- **KILL :** un régime n'est utile que s'il **améliore un sleeve confirmé en interaction** (**shuffle-label null**) ;
  **un HMM qui overfit l'historique est PIRE que l'ER trivial** — prouve la **stabilité OOS des états** (les états
  ne doivent pas re-segmenter à chaque ré-estimation).
- **Liv. :** `regime_engine` servant des conditionneurs → `research_streams/S8/features/`.

### 8.2 — Macro regime-gates comme conditionneurs (C1 généralisé)
- **H :** HY-spread sur les proxies risk (AUD/JPY), 2s10s pour le switch carry-vs-trend, conditionnent les sleeves
  sans être des signaux.
- **Méca :** état macro exogène, pas prix-sur-prix.
- **Méthode :** gates = fonctions de `fred_bamlh0a0hym2` / `fred_t10y2y` (laggés), testés **en interaction** sur un
  sleeve accessible.
- **KILL :** testé **uniquement** en terme d'interaction/gate (**shuffle-label null**), **jamais standalone**
  (sinon = re-découverte de beta macro).
- **Liv. :** **gates macro validés** pour Streams 3/5 → `research_streams/S8/features/`.

### 8.3 — CV régime-aware pour durcir la validation
- **H :** segmenter les folds CPCV par régime (vol/macro) réduit l'optimisme de backtest en non-stationnarité.
- **Méca :** un edge qui ne tient que dans un régime doit être *vu* comme tel par la validation.
- **Méthode :** étends `cpcv.py` pour stratifier/segmenter les folds par état de régime (8.1). **N'altère pas
  l'API existante** — ajoute un mode régime-aware optionnel.
- **KILL :** si la CV régime-aware **ne change pas le verdict** d'un sleeve connu (p.ex. S2), **garde la CPCV
  standard** (parcimonie) ; ne complexifie pas sans gain prouvé.
- **Liv. :** **extension `cpcv.py` régime-aware** (mode optionnel), livrée aux Streams 1/10.

## INVARIANTS (rappel)
Aucun signal directionnel — uniquement des **états/gates**. Estimation **causale, zéro look-ahead** (un moteur de
régime qui « voit le futur » invalide tous les backtests aval). Conditionneur validé par shuffle-null. **Stabilité
OOS** plus importante que le fit in-sample. Shadow. TDD.

## SORTIE
`research_streams/S8/FINDINGS.md` (moteur + stabilité OOS + verdict CV régime-aware) ·
`research_streams/S8/CHECKPOINT.md` · `regime_engine` + extension `cpcv.py` + code
`src/hermes/research/s8_regime_engine.py` (+ tests). **STOP.**
