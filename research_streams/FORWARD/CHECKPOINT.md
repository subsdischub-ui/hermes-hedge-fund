# FORWARD — CHECKPOINT

- **A. Cron unifié** `python -m hermes.forward.run` (idempotent, fail-soft, PIT) → `data/forward/`. Flux : skew (CÂBLÉ, réutilise S4), comex (calendar-spreads PIT, seam non câblé), macro (dot-vs-FF, seam non câblé).
- **B.1 funding_squeeze shadow** : CÂBLÉ & LIVE, byte-identique à `evaluate_signal`/règle live. 1er trade : WIF 2026-05-27, z=13.4, ret_net −2.70 % (LONG perdant = OOS bear propre). Pas un nouvel essai.
- **B.2 XS reversion** : pré-enregistrée (2026-06-29 ; signe INVERSÉ, 14j, terciles, 23 perps, 180-settl). Sanity IN-SAMPLE = **FRAGILE** (par-an : 2020-21 +1.7, 2024+ **−0.07** ; voisinage [−0.12,+0.85] ; LOO ETH-porté) → **NON câblé** (`WIRE_XS_REVERSION=False`, 1 flag pour activer).
- Déflation verdict forward = **1892** ; registry crypto +1 → 1870. 43 tests verts ; ruff clean.
- Câblage live = porte de merge (NON touché). **STOP.**
