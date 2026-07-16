# Pré-enregistrement DURCI — S2 en largeur, univers sans biais de survie
(2026-07-16, écrit AVANT tout calcul — l'unique suite autorisée par le verdict du 2026-07-02 ;
en cas d'échec : CLÔTURE DÉFINITIVE de la piste largeur, pas de 3e chance.)

## Pourquoi ce test existe
Le test original (PREREG.md) a produit +75.8 % d'événements marginaux (p_max5=0.003) mais a
ÉCHOUÉ à la lettre (fenêtre bear 2024-07→09 n=4 négative), avec deux réserves d'audit :
(1) l'univers ne contenait QUE des survivants — un délisting suit souvent une chute, donc les
événements manquants des morts sont plausiblement PERDANTS et le +211 bps/évt publié est
plausiblement gonflé ; (2) 2022 : zéro événement z≥2 — le bear-check est aveugle au cœur d'un
crash. Ce test durci traite (1) par construction. Il ne peut PAS traiter (2) : si l'univers
durci n'a toujours aucun événement 2022, cette cécité reste une réserve PERMANENTE de la piste.

## Ce qui ne change PAS (verbatim du PREREG original)
- Règle S2 GELÉE : fenêtre z = 180 barres de funding, z ≥ 2.0, long-only, hold = 24 barres 4h,
  coût taker 15 bps RT, warmup 200 barres. AUCUN paramètre re-tunable.
- Panier TOUTES-DATES dé-chevauché au niveau date (un jour = un pari max).
- Barres de verdict, zéro tolérance, identiques : joint-null MC date-level p < 0.05 stable
  5 seeds sur le panier élargi complet ; OOS-by-date (moitiés chronologiques) positif ;
  déflation contre le N réel du registre au moment du calcul ; bear-window check : TOUTES les
  sous-fenêtres bear du test original (dont 2024-07→09) positives sur le panier élargi.
- Interdits : re-tuning, retrait post-hoc de symboles perdants, re-découpage de fenêtres,
  changement d'OOS. Tout écart = essai supplémentaire compté au registre.

## Ce qui change (le durcissement, fixé ICI)
1. **Univers POINT-IN-TIME incluant les morts.** Critères d'inclusion du PREREG original
   ((a) volume ≥ $200M/24h, (b) ≥ 3 ans d'historique funding, (c) exclusions stablecoins/
   wrapped/métaux/actions-indices), évalués À CHAQUE DATE et non à aujourd'hui :
   - un perp ENTRE dans l'univers à la première date où il satisfait (a)+(b)+(c)
     (volume mesuré sur ses propres klines historiques, moyenne 30 j glissante) ;
   - un perp SORT à son délisting — jamais rétroactivement. Ses événements comptent
     jusqu'au bout.
   - Sources de la liste des morts, dans l'ordre, sans improvisation : (i) contrats
     non-TRADING de /fapi/v1/exchangeInfo (123 SETTLING au 2026-07-16) ; (ii) symboles du
     lake S0 (data/manifest, funding CoinGlass 6 ans) absents des TRADING actuels ;
     (iii) annonces de délisting Binance (web) pour les morts antérieurs à (i)-(ii).
     L'API fapi sert encore klines+funding des délistés (vérifié 2026-07-16 sur SRM/TOMO/
     HNT/COCOS/ANC/DODO).
2. **Sortie forcée au délisting.** Un événement encore ouvert quand son perp meurt est clos au
   dernier bar disponible avec coût de sortie DOUBLÉ (30 bps la jambe de sortie) — dur mais
   réaliste (liquidité de fin de vie).
3. **Le biais de survie est MESURÉ, pas seulement neutralisé** : rapporter séparément
   (événements marginaux, edge net/évt, bear-windows) pour l'univers survivants-seuls vs
   l'univers durci. L'écart EST le chiffre du biais.

## Critères de verdict (lettre, écrits avant de calculer)
- **PASS** ssi TOUTES les barres du PREREG original passent sur l'univers durci complet, ET
  les événements marginaux dé-chevauchés restent ≥ +50 % du panier actuel.
- **FAIL** sinon — et la piste largeur est CLOSE définitivement (au minimum jusqu'à données
  nouvelles d'une autre nature, ex. cross-venue Bybit ≥ 2026-11, qui est un AUTRE prereg).
- Cécité 2022 : si l'univers durci n'a aucun événement z≥2 en 2022, le PASS éventuel porte la
  réserve permanente « bear-de-crash non couvert » — affichée partout où S2-largeur serait
  citée (dashboard/notes), pas enterrée en annexe.
- Un PASS ne déploie RIEN : il rouvre seulement la discussion opérateur (élargir le panier S2
  live = décision opérateur, avec l'arithmétique de flux honnête).

## Registre
+1 essai crypto (source `s2_breadth_hardened_prereg`), bump 1910 → 1911 committé AVANT la
lecture de tout résultat. Un seul essai. Pas de grille.
