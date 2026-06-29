# STREAM 3 — EXPANSION D'UNIVERS + CALIBRATION PAR INSTRUMENT
**Séniorité : Mid · Branche : `hermes/s3-universe` · Après S1, en parallèle de S2**

> **LIS D'ABORD** : `docs/streams/STREAM_COMMON.md`, le programme §Stream 3, le dead-ends ledger,
> `research_fx/LAB_JOURNAL.md`. Ton job = *le bon sens* de « armer plus souvent » : **plus d'instruments qui
> arment sous la MÊME règle**, jamais des seuils plus bas.

## MANDAT
Attaquer la limite réelle « **7 paires** ». Faire en sorte que le **livre soit armé plus souvent par PLUS
d'instruments** sous la **MÊME** règle de qualité. Tu testes la **généralisation cross-asset** des sleeves
**déjà confirmés** (S2 surtout) et la **calibration seuil-par-instrument**. Le filtre cross-asset est *le
killer filter*.

## BORNES DE PÉRIMÈTRE
Tu **ne crées aucun nouveau signal ni aucune nouvelle donnée** — tu testes la généralisation et la calibration
de sleeves existants. Le positionnement (S2), les options (S4), le FX (S5) ne sont pas ton terrain.

## INPUTS LAKE
`coinglass_funding_rate` (7 paires) + `binance_klines` (7 paires, `1d`/`6h`) pour S2/S3. Pour le sleeve range :
prix + vol. (Le conditionneur régime du Stream 8 sera consommé **au merge** ; en parallèle, utilise des **buckets
vol** ou le routeur trivial existant comme stand-in.)

## LES 3 VECTEURS

### 3.1 — Généralisation de S2 (funding-squeeze) sur univers élargi
- **H :** `funding_z ≥ 2.0` continuation tient au-delà de BTC/ETH/SOL (les autres perps USDT-M liquides du lake :
  AVAX, SUI, ARB, WIF ; et si la donnée le permet, élargis ta réflexion à LINK/DOGE/BNB en notant la dispo).
- **Méca :** funding extrême = demande à levier persistante, mécanisme **non spécifique à 3 actifs**.
- **Données :** `coinglass_funding_rate` + `binance_klines` (déjà câblés). **Vérifie la profondeur par paire**
  (AVAX/SUI/ARB/WIF récentes → échantillons plus courts ; WIF possiblement trop court pour un gate valide).
- **Méthode :** applique **exactement** la règle S2 (même `funding_z`, même horizon) à chaque nouvelle paire ;
  MC-null par-actif + joint-null daté sur l'ensemble.
- **KILL :** **même règle, ≥ plusieurs instruments** ; SOL était déjà la jambe faible (MC p=0.086) — si les
  nouveaux actifs échouent le MC-null par-actif **ou** le joint-null daté, **ils ne rentrent pas**. Un « win »
  mono-instrument est **rejeté par défaut**.
- **Liv. :** **liste des instruments où S2 généralise honnêtement** → univers tradable étendu (le livre arme
  mécaniquement plus souvent).

### 3.2 — Seuils vol-scaled *par instrument* (pas plus bas — mieux calibrés)
- **H :** un seuil unique (`funding_z` 2.0, conviction 1.0) est sous-optimal car la vol/le funding-baseline
  diffèrent par actif ; un seuil **normalisé par la dispersion propre** de chaque actif arme « au bon endroit »
  **sans baisser la barre de qualité**.
- **Méca :** scale-free (comme l'IR vol-scaled du momentum).
- **Méthode :** seuil par instrument = fonction de la dispersion historique propre (z-score sur baseline
  par-actif), pas une constante abaissée. Winsorize + re-check.
- **KILL :** prouver sur **held-out** que le seuil par-instrument **n'augmente PAS le taux de faux-trades** ; si
  « mieux calibré » = juste « plus de trades » → **whipsaw déguisé → mort**.
- **Liv. :** **table de seuils par instrument**, validée OOS, **avec preuve que le win-rate/Sharpe par trade ne
  se dégrade pas**.

### 3.3 — `mean_reversion_regime` : la ré-armer honnêtement en régime range
- **H :** le sleeve range existe dans le slate (`RANGE_SKILLS`) mais n'a jamais tradé ; en vrai régime range
  (≈ maintenant), une réversion bien gatée peut contribuer.
- **Méca :** complémentarité de régime avec S1/S2 (qui crowdent les longs en bull).
- **Méthode :** réversion **lente**, **conditionnée régime range** (stand-in : buckets vol/ER ; au merge :
  Stream 8). Jugée par **LOO-Δ-Sharpe**.
- **KILL :** le labo a tué l'intraday mean-reversion et les pairs (half-lives 20-99 j) — **n'y retourne PAS sur
  ces cadences**. La seule mutation autorisée est une réversion lente conditionnée régime. Si **ρ trop élevé**
  avec un sleeve existant ou si l'edge = bull/bear-beta → mort.
- **Liv. :** sleeve candidat conditionné régime, ou négatif daté.

## INVARIANTS (rappel)
Pas de nouveau signal. Cross-asset = le killer filter. **Jamais de seuil abaissé** — uniquement mieux calibré,
avec preuve de non-dégradation. Gate niveau-livre. Shadow size 0. TDD.

## SORTIE
`research_streams/S3/FINDINGS.md` (univers étendu validé + table de seuils + verdict range) ·
`research_streams/S3/CHECKPOINT.md` · code dans `src/hermes/research/s3_universe.py` (+ tests). **STOP.**
