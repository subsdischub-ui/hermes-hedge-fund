# Positions structurelles — la recherche « partir de zéro » (2026-07-18)

**Mandat opérateur** : « déploie des workflows d'agents qui partiront de 0 pour trouver
l'approche à côté de laquelle toi et moi passons depuis des mois. Les hedge funds gagnent en
algorithmique, une minorité de particuliers trouve son compte. THERE IS SOMETHING TO CATCH. »

**Méthode** : deux workflows indépendants et OPPOSÉS par construction, pour que la convergence
ait valeur de preuve. (1) **Divergence aveugle** : 7 personas SANS le cadre Hermès (aucun accès
au ledger, au registre, aux morts) — 35 idées brutes, chacune avec mécanisme/payeur/évidence
sourcée. (2) **Décomposition structurelle** : POURQUOI les gagnants algorithmiques gagnent
(6 catégories), audit de notre propre cadre (coût, Grinold, capacité), 5 positions candidates,
panel adversarial 3 lentilles × 5. Les deux synthèses ont été tuées par une limite de session ;
récupérées intégralement depuis les journaux, puis **chaque allégation porteuse re-vérifiée par
exécution par l'orchestrateur** (fichiers, matcher, données, web).

> **La réponse au mandat tient en une phrase : les recherches indépendantes CONVERGENT — il n'y
> a pas d'approche de marché manquée à notre échelle ; ce qui était manqué est à côté du
> marché : le péage qu'on se paie à soi-même (P1), et la discipline d'assumer le null (P5).**

---

## 1. Les 35 idées aveugles — le tri que la panne avait empêché

Le workflow de divergence a rendu `n_novel: 0` — **artefact de panne** (l'agent de tri est mort
sur la limite de session AVANT de lire les idées ; le zéro comptait un résultat nul). Les 35
idées, récupérées du journal, ont été triées ici. Composition :

| Verdict | n | Contenu |
|---|---|---|
| Déjà couvert par le ledger/les sleeves | 9 | cash-and-carry (=S3 suspendu / P4), cross-venue funding (#35-36, axe légal = P2), cascade de liquidations (#22/#27), réversion illiquide (famille #15 + Dobrynskaya : la prime ne vit QUE sur les petits coins — coût pire), He-Manela Sharpe 1,8 (échantillon 2020-22, supplanté par nos mesures 2025-26 + Boucle β), airdrops/points (mort en mémoire), keeper/MEV long-tail (winner-take-most documenté, 80-90 % du brut brûlé en enchères) |
| **Mur légal France** | 4 | Polymarket MM/arb (ANJ : transactions interdites nov. 2024, **blocage FAI ordonné le 16/07/2026**), Polymarket↔Kalshi, matched betting (écosystème = exchanges UK, interdits en France), +EV betting (pas de Pinnacle/exchanges sous ANJ) |
| Décisions d'allocation opérateur, hors moteur | 5 | HLP vault (15-30 % APR, short-vol, JELLY −$13,5M), Pendle PT (~11 % en unités sUSDe, risques empilés), Morpho (4-8 %), T-bills, points farming |
| Pivots business, hors moteur | 7 | freelance IA ($100-300/h documenté), vente de pelles, audit de backtests productisé, newsletter du cimetière (corpus unique : 42 morts pré-enregistrés), Darwinex/Collective2 (plafonds documentés ~17 k€ cumulés au MIEUX), Numerai (distribution individuelle non publiée) |
| **Cadre neuf : louer du capital (prop firms)** | 5 | HyroTrader (Bybit API, bots OK), Breakout/Kraken, Topstep (futures — notre edge n'y existe pas), FTMO (CFD — idem), méta-cadre optionnel |
| Confirmation de nos positions | 5 | les personas 4 et 5, PARTIS DE ZÉRO, re-dérivent funding delta-neutre + liquidité subventionnée comme « les deux seuls trading rationnels restants » — la convergence qui donne sa force au verdict |

**Le seul cadre réellement neuf — les props crypto — attaque la vraie variable (le capital),
et son arithmétique honnête le tue AU LIVRE ACTUEL** : passer un challenge (+10 % avant −6 %)
avec notre livre à Sharpe 0,0739 et sizing validé 0,5 %/trade exigerait des années ; le passer
en 3 %/trade (autorisé par les règles) contredirait la propre architecture de sécurité de
l'opérateur (6× le cap validé). L'entonnoir publié du secteur dit le reste : ~5-12 % de
passage, payout moyen par inscrit ≈ $130 (FTMO : $329M de revenus 2024 CONTRE les candidats).
**Re-visitable UNIQUEMENT si le prereg Bybit (P2) passe** — même venue que HyroTrader, flux
d'événements ×2. Jusque-là : espérance négative, documentée, close.

## 2. Les 5 positions structurelles — votes et vérifications

| Position | Panel (3 lentilles) | Vérifié par l'orchestrateur |
|---|---|---|
| **P1 — Auto-péage** : maker+BNB sur nos propres entrées ; le mur de 15-20 bps est aux 2/3 un choix (frais 9→3,6 bps ; slippage 3 bps = hypothèse jamais mesurée) | 0/3 réfuté | config `maker_orders: false` + justification exacte ✓ ; `research_maker/results.json` : fill 99,25 %, sélection adverse ≈ 0,47 bp, 18 winners ratés / 2 417 ✓ |
| **P2 — Flux forcés élargis** : les deux SEULS axes légaux de S2 (revive #41 verbatim) — Bybit (≥ 2026-11), Hyperliquid horaire | 0/3 réfuté | revive_condition #41 verbatim ✓ ; BIS WP1087 (le payeur : retail trend-chasing à levier) ✓ ; lake insuffisant pour dater les événements 8h ✓ → collecte lancée |
| **P3 — Cadence 1h sous mur réduit** | **3/3 RÉFUTÉ** | `data/intraday/_results.json` re-lu : **124 configs MR, 0 pass ; les 62 en 1h TOUTES négatives MÊME à coût maker** (meilleure : −1,7 bps/trade). La porte cadence était déjà fermée dans nos propres données — au mur réduit. |
| **P4 — Vendeur de bilan/assurance** : REFUS confirmé | 0/3 réfuté (le refus tient) | Trou de gouvernance re-vérifié : 5/5 formulations passaient le matcher → **entrée #42 mécanisée** (chiffres Boucle β vérifiés à la source ; mutation 0/4 → 4/4 ; réveil S3 protégé par test) |
| **P5 — Null structurel** : les rentes restantes sont fermées par CONTRAT (PFOF/OTC, colocation — top 6 firmes > 80 % des courses QJE 2022, tiers VIP $1-15M/30j) ; la variable dominante est le capital | 0/3 réfuté | Registre 1969 exact ✓ ; aucune entrée « gel » n'existe encore ; l'adoption du null a ses triggers : échec P1 → échec P2 → gel formel |

## 3. Ce qui a été FAIT (mécanismes, pas narration)

1. **Entrée #42 au ledger** — le verdict Boucle β (pente signée +0.047 t=+2.72 : la récolte
   monte AVEC la liquidité ; queue méd −2,3 % ; reverse-carry −1,9 % au borrow honnête) ne vit
   plus en prose. Décisifs calibrés pour ne JAMAIS bloquer le réveil légitime de S3 (testé).
2. **PREREG P1 committé AVANT toute mesure** (`research_maker/PREREG_SHADOW_FILL.md`) : fill
   STRICT/AU-TOUCHER, coût de chasse empirique, verdict gelé à 40 signaux (fill ≥ 0,70 ET gain
   ≥ +2 bps), pas de bump registre (mesure de coût, pas d'essai — justifié). FAIL → porte coût
   fermée, P3 morte par dépendance, réveil S3 non rempli, null P5 avance.
3. **Module shadow maker** (en implémentation) : à chaque entrée live du slate, l'ordre maker
   hypothétique est loggé et résolu sur la barre suivante ; les DEUX bornes publiées au
   dashboard + racontables au chat. Le verdict à 40 sera mécanique, comme le TRM à 100.
4. **Collecteur de funding natif Bybit 8h** (en implémentation) : backfill ~6 ans + cron —
   pour que le prereg P2 de novembre trouve ses données prêtes.

## 4. Ce que ceci ne prétend pas

Un PASS de P1 vaut **~60-90 bps/an de drag rendus** au livre — une réparation de fidélité, pas
un edge. Un PASS de P2 en novembre doublerait le FLUX d'événements du seul edge prouvé — pas sa
taille. Le Sharpe du livre (0,0739) reste le fait qui gouverne : à cette force de signal, le
capital multiplie surtout la variance. Les pivots business (vendre la compétence, la machine,
le corpus) restent les options au rendement documenté le plus élevé — et restent des décisions
d'opérateur, hors du mandat du moteur.

*Preuves : journaux `wf_3018f6b1-a9a` (divergence) et `wf_58776fc3-549` (positions) ;
vérifications d'exécution dans ce dossier ; sources France/Polymarket : France 24, iGaming
Business, NEXT.io (16-18/07/2026).*
