# Pré-enregistrement — Mesure shadow du fill-rate maker sur NOS entrées live (P1)
(2026-07-18, écrit AVANT toute mesure. Position P1 du chantier « positions structurelles » :
cesser d'être le payeur du spread — la seule fraction du péage market-maker occupable à notre
échelle est celle qu'on se verse à soi-même en cessant de la payer.)

## Ce que ceci N'est PAS (statut épistémique, à ne pas maquiller)
- Ce n'est PAS un re-test du mort #20 « Maker-execution free lunch ». #20 a tué le maker comme
  RESSUSCITEUR de signaux rapides morts bruts (gross +1,6-4,8 bps < même le coût maker ; 0/7
  actifs au gate). Ceci mesure le maker comme EXÉCUTION du livre déjà validé — précisément la
  `revive_condition` écrite dans #20 (« a verified rebate on uninformed flow » : notre propre
  flux d'entrée EST le flux non-informé à l'horizon du fill) et la condition de réveil de S3
  (`api.py` : « fill-rate maker réel mesuré au stage_1 »).
- AUCUN bump du registre d'essais. Justification explicite : cette mesure ne sélectionne pas
  une règle génératrice de rendement parmi des candidates (aucune sémantique de déflation) —
  elle mesure un COÛT d'exécution. La protection contre l'auto-flatterie n'est pas le compteur,
  ce sont : les seuils gelés ci-dessous + la borne basse stricte + la double comptabilité.
- Le flip `maker_orders: true` reste INTERDIT tant que le verdict n'est pas PASS — et le flip
  lui-même sera un changement visible (dashboard + chat, règle de visibilité).

## La mesure (définitions gelées)
À chaque entrée LIVE du slate (mean_reversion_regime, funding_squeeze_continuation) exécutée
par le moteur paper (qui paie taker aujourd'hui), logger un ordre maker HYPOTHÉTIQUE :
- prix limite = close de la barre de décision − offset (achat) / + offset (vente) ;
- offsets mesurés : 0 bp et 5 bp (les deux, toujours) ;
- **fill STRICT** (borne basse honnête) : la barre 4h SUIVANTE traverse STRICTEMENT la limite
  (low < limite pour un achat, high > limite pour une vente — jamais ≤/≥ : ignorer la position
  de queue est conservateur) ;
- **fill AU-TOUCHER** (borne haute) : low ≤ limite / high ≥ limite. Les deux bornes publiées —
  l'encadrement est gratuit et empêche de se raconter le chiffre flatteur ;
- si non-fillé STRICT : coût de chasse mesuré = mouvement adverse entre le close de décision
  et l'open de la barre suivante (le prix payé pour re-entrer taker en retard). Pas de modèle :
  la moyenne EMPIRIQUE des non-fills du log.

## Économie par trade (formule gelée)
`gain_net_bps = fill_strict × gain_par_fill − (1 − fill_strict) × coût_chasse_mesuré`
avec `gain_par_fill` = (perp_taker − perp_maker) = 3,0 bps (config actuelle, jambe d'entrée)
— le slippage paper (3 bps/côté) N'est PAS compté comme gain tant qu'il n'est pas re-mesuré
(chantier parallèle hors-prereg : mesure one-shot des spreads réels des 7 perps, correction de
`slippage_bps` et du commentaire « Hyperliquid » périmé de `config/default.yaml`).

## Verdict (seuils gelés, lus à la lettre)
Sur les **40 premiers signaux consécutifs** post-déploiement (aucune fenêtre re-découpée,
aucun signal exclu post-hoc) :
- **PASS** ssi `fill_strict(offset 0) ≥ 0,70` ET `gain_net_bps ≥ +2,0 bps/trade` ;
- **FAIL** sinon → la porte coût est FERMÉE : `maker_orders` reste false, P3 (cadence) reste
  morte par dépendance, la condition de réveil de S3 reste non remplie, et le null structurel
  (P5) avance d'un cran. Pas de 2e fenêtre, pas de seuil révisé après lecture.
- Un PASS ne flippe RIEN automatiquement : il autorise la proposition du flip à l'opérateur,
  avec les deux bornes affichées et le modèle de fill mesuré câblé derrière.

## Ce qu'un PASS vaudrait (arithmétique honnête, écrite avant)
~25 trades/mois × ~2-3 bps nets ≈ **60-90 bps/an de drag rendus au livre** — une réparation
de fidélité, PAS un edge. Le Sharpe du livre (0,0739) reste le problème ; ceci en enlève le
frottement, il ne crée pas le muscle.

Sources internes : `research_maker/results.json` (sélection adverse à-la-touche ≈ 0,5 bp,
18 gros winners ratés / 2 417 signaux) ; `data/intraday/_results.json` (le régime maker est
DÉJÀ simulé et négatif pour toute cadence < 4h — ce prereg ne rouvre pas #15) ;
`config/default.yaml` (fees, `maker_orders: false` et sa justification).
