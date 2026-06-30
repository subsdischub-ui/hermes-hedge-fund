# CHECKPOINT — Boucle α Carry-Harvest (iter 0)

- **VERDICT : négatif propre de déploiement** (red-teamé 3 agents → `negative_is_robust`, aucun bug changeant le verdict). Ni CC ni XV ne passent §4(b). Structure saine, edge arbitré.
- **CC** : delta-neutre confirmé (no bull-beta, maxDD ~−4.6 % / 7 ans). Headline ~12 %/an = artefact bull (**72 % du cumul = 2019–2021**). Forward : 2025 ~0.5–3 %, **2026-H1 ~1.5 % annualisé** (sous le seuil). Sélection de coins = **0 lift forward** (oracle look-ahead : 2026 −0.09 pp). DSR = mauvais instrument (≈0 était artefact d'unités → 1.0 corrigé). Queue base RÉELLE et SOUS-estimée (SOL −11.9 % close / −28.6 % intra ; cluster depeg 2023-03-24).
- **XV** : buy&hold −4.5 %/an ; gated \|d\|≥10 % marginal (+1.6 %/an) MAIS Sharpe<1, 2.25 ans, meurt au taker, **artefact mono-coin (OP)**. Pas déployable.
- **Correction load-bearing** : `funding_on_4h` backward-carry (double-compte ~2×, 43 % règlements sans match) → funding sourcé de `binance_funding_native` (ancre : BTC-2024 +11.96 % reproduit exactement).
- **Livré** : `research/carry_harvest.py` (18 tests TDD, ruff clean, 0 régression / 807 pass). Grille pré-enreg → crypto **1900**. Rien live ; S3 non touché.
- **Rouvre** : (1) couche contrôle iter-1 (sélection coins via funding-mean-reversion morte + timing funding_squeeze + régime S6/S8) ; (2) régime funding plus élevé ; (3) XV bi-directionnel + profondeur HL pré-2024.
- **Pré-existant hors-sujet** : `openpyxl` manquant (1 test gold), `import fxgate` path (4 tests S6).
