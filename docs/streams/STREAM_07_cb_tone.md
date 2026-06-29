# STREAM 7 — TON BANQUE CENTRALE PAR LLM : LE DRIFT (PAS LE PRINT)
**Séniorité : Mid + NLP · Branche : `hermes/s7-cb-tone` · En parallèle (LLM-natif)**

> **LIS D'ABORD** : `docs/streams/STREAM_COMMON.md`, le programme §Stream 7, le dead-ends ledger,
> `research_fx/LAB_JOURNAL.md`. Le corpus de discours BIS est **déjà dans le lake** (20 365 discours depuis
> 1996). Exploite l'edge *natif* d'Hermès : **son cerveau est un LLM** → c'est lui qui classe le ton.

## MANDAT
Le labo a tué E3 (FOMC tone-change) parce que « **fully repriced at 2pm print** » — mais E3 testait le *statement
au print*. L'axe inexploré, c'est le **drift lent du ton hawk/dove des discours *entre* réunions**, classé par
**LLM** (pas par dictionnaire), tradé en *résidu*, **jamais à travers le print** (discipline news-blackout — la
perte $12k).

## BORNES DE PÉRIMÈTRE
Tu ne trades **JAMAIS** la minute du communiqué/de la conf. Ton signal est le *drift* multi-jours. Le calendrier
ForexFactory **est mort** (cf. BLOCKERS : pas d'historique, pas de champ `actual`) — ne t'appuie pas dessus pour
le blackout historique ; utilise les dates de réunion connues (FOMC/ECB/BoE) pour fenêtrer le blackout.

## INPUTS LAKE
`cbspeak_*` (BIS, 20 365 discours depuis 1996 — **filtre par `speaker`** : la colonne `institution` est peu
peuplée ; identifie Fed/ECB/BoE via les noms de gouverneurs, p.ex. Powell, Lagarde, Bailey, Carney, Broadbent…) ;
cibles = `gold`, crosses FX (EUR/JPY/GBP), `fred_dgs10`/`fred_t10y2y` pour les taux. Pour 7.3 : SEP (Fed) + Fed
Funds futures (vérifie la disponibilité — possiblement à collecter).

## LES 3 VECTEURS

### 7.1 — Indice de drift hawk/dove par locuteur (LLM-natif, corpus libre profond) `[axe principal]`
- **H :** le ton agrégé glissant des discours individuels (Fed/ECB/BoE) porte de l'info sur la trajectoire *future*
  du taux, distincte du choc de taux au print.
- **Méca :** littérature 2025-2026 active (BIS *CB-LMs*, IMF, RBA *Ornithologist*, ECB *Dovish Coos or Hawkish
  Screech*) ; échelle hawk-dove par locuteur (Bordo-Istrefi) ; le *forward-guidance shock* est distinct du *rate
  shock* (Bauer-Swanson).
- **Méthode :** (a) **classe chaque discours** sur une échelle hawk↔dove via LLM (prompt structuré, score continu +
  justification ; c'est *ta* valeur ajoutée). (b) Agrège en **drift glissant par banque** (moyenne mobile, résidu
  vs niveau). (c) Cible = drift multi-jours sur devises/or/taux, **entrée APRÈS la fenêtre de blackout**.
- **KILL :** MC-null + déflation ; **si l'edge n'existe qu'au print (= E3 déjà mort) → mort**. Doit tenir hors
  blackout, multi-jours, ≥2 cibles.
- **Liv. :** sleeve `cb_tone_drift` (XAU/EUR/JPY/GBP) en **shadow** + verdict daté + le **classifieur LLM réutilisable**.

### 7.2 — Surprise macro auto-construite (B4 — réalité : ForexFactory mort)
- **H :** un indice de *surprise* (actual − forecast, standardisé, décroissant 60 j) donnerait un différentiel
  tradable en drift, après blackout.
- **Données `[RÉALITÉ]` :** **ForexFactory est mort** (BLOCKERS : current-week-only, pas de champ `actual`, mirror
  404, site Cloudflare-403). **B4 reste bloqué pour le backfill libre.** ALFRED peut backfiller les *actuals-en-vintages*
  mais **n'a pas de consensus forecast** → il **ne peut pas, seul, reconstruire une surprise**.
- **Méthode :** **audit honnête** : peux-tu obtenir un panel actual+forecast point-in-time libre/peu cher ? Sinon,
  **verdict « données verrouillées »** (valide) + note ce qu'un vendeur (Econoday/TradingEconomics) débloquerait,
  ou un cron forward-accruant (snapshot hebdo + re-pull post-release).
- **KILL :** historique point-in-time trop court/payant → **verdict « données verrouillées »**. Pas de fabrication
  de forecasts.
- **Liv. :** **verdict « verrouillé » daté** + option forward-cron documentée (décision Malachie).

### 7.3 — Dispersion du dot-plot / désaccord Fed-marché comme conditionneur
- **H :** l'écart entre le dot-plot (SEP) et les Fed Funds futures (désaccord sur le chemin) est un état macro lent.
- **Méca :** incertitude de politique = état exogène, pas prix-sur-prix.
- **Données :** SEP (Fed, libre, trimestriel) + Fed Funds futures (libre) — **vérifie la disponibilité dans le lake
  ou collecte point-in-time** ; sinon note le verrou.
- **Méthode :** `fed_market_disagreement` = écart dot-plot médian vs courbe FF futures, par horizon.
- **KILL :** **conditionneur uniquement** (interaction/**shuffle-null**), jamais standalone ; alimente le Stream 8.
- **Liv. :** feature `fed_market_disagreement` → `research_streams/S7/features/` (ou verdict « données à collecter »).

## INVARIANTS (rappel)
**Jamais trader à travers un print.** Le LLM classe, mais le **gate reste le même** (MC-null, déflation, drift
multi-jours). Conditionneur ≠ signal (7.3). Lague tout. Shadow. TDD.

## SORTIE
`research_streams/S7/FINDINGS.md` (**7.1 = le sleeve ; 7.2 = verdict verrouillé ; 7.3 = conditionneur/à-collecter**) ·
`research_streams/S7/CHECKPOINT.md` · classifieur LLM + code `src/hermes/research/s7_cb_tone.py` (+ tests). **STOP.**
