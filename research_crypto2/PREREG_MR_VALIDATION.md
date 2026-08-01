# Pré-enregistrement — `mean_reversion_regime` face à la barre de S2
(2026-08-01, écrit AVANT tout calcul. Registre crypto 1911 → 1912, bumpé avant lecture.)

## Le fait qui déclenche ce test
`mean_reversion_regime` est **89 % du flux de trades du livre live** (24 des 27 trades forward
du mois écoulé) — et il n'a **jamais** passé de test statistique dédié. Ce n'est pas une
allégation : `docs/SYSTEME_AUTONOME.md:387` l'écrit noir sur blanc (« pas de p-value/joint-null
dédiée trouvée dans le repo, contrairement à S1/S2 ; son statut VIVANT s'appuie sur sa présence
dans le slate, pas sur un test statistique »), et `research_crypto2/` ne contient aucun
`_mr_*.json` là où S1, S2 et S3 ont chacun le leur. Son statut est un **héritage**, pas une
preuve.

Mesure forward qui rend le test urgent (2026-07-02 → 2026-08-01, règle gelée, coût 15 bps RT,
instrument validé — le stop BTC sort à −176,4 bps contre 1,7637 % de `stop_pct` calculé par la
fonction réelle) : **MR = −40,22 bps/trade net sur 24 trades**, et le **BRUT est déjà négatif**
(−16,87 bps/trade sur l'ensemble du livre). Un mois n'est pas une preuve (t = −0,51, p = 0,61) —
c'est précisément pourquoi il faut le test complet plutôt qu'une réaction au dernier mois.

## L'hypothèse (H1)
La règle GELÉE `mean_reversion_regime` (RSI(2) seuils 10/90 · Bollinger %B(20,2) seuils 0,1/0,9 ·
filtre MA200 · gate régime Kaufman ER · stop 2×ATR, TP 3×ATR, max_hold 12 barres 4h — AUCUN
paramètre re-tunable) possède un edge net réel sur l'univers live (BTC/ETH/SOL/AVAX/SUI/ARB/WIF),
au-delà du hasard et au-delà de la dérive du marché.
**H0** : son edge apparent est du bruit et/ou de la beta de marché déguisée.

## Le test (UN essai, aucune grille)
Réplication EXACTE de la barre qu'a franchie S2 (`research_crypto2/audit_s2_funding_squeeze.py`),
appliquée sans adoucissement :
1. Événements par symbole avec la règle gelée, warmup ≥ 250 barres, coût **15 bps RT**.
2. **MC random-entry null, 5 seeds** → `mc_p_max_5seeds` et `mc_p_stable_lt05`.
3. **OOS-by-date** (`oos_sharpe_40` : dernier 40 % chronologique).
4. **Contrôle de la marée (beta)** : `edge_above_tide_bps` = rendement du signal − rendement
   moyen d'une entrée quelconque au même horizon. Un edge qui n'est que la dérive du marché
   n'est pas un edge.
5. **Split bull/bear** : le bear doit rester non-catastrophique (S2 est bear-positif).
6. **Joint-null panier date-level dé-chevauché**, 5 seeds.
7. Déflation au N réel du registre au moment du calcul (1912).

## Critères de verdict (lus à la lettre, écrits avant)
- **PASS** ssi : `mc_p_max_5seeds < 0,05` ET `oos_sharpe_40 > 0` ET `edge_above_tide_bps > 0`
  ET joint-null panier `p_max5 < 0,05`, sur le panier complet.
- **FAIL** sinon → `mean_reversion_regime` sort du slate exécutable (proposition à l'opérateur,
  jamais un retrait automatique), le livre live devient **S2-only** (~13 événements/an), et la
  conséquence en cascade est assumée et publiée : le débit de trades s'effondre, le gate
  `edge_paper_trades: 40` devient inatteignable à horizon raisonnable, et le null structurel
  (P5) est adopté formellement.
- **Zéro tolérance** : pas de re-tuning, pas de re-découpage de fenêtres, pas de retrait post-hoc
  d'un symbole perdant, pas de 2e fenêtre après lecture.

## Ce que ce test ne prétend pas
Un PASS ne rend pas l'entreprise rentable : il confirmerait seulement que le flux principal du
livre n'est pas du bruit. Un FAIL ne « casse » rien — il rend visible ce qui était déjà vrai et
non mesuré. Dans les deux cas, **la décision de financer un compte réel reste celle de
l'opérateur**, et le forward négatif du mois écoulé reste au dossier.
