# Audit de rentabilité — 2026-08-01

**Mandat opérateur** : « cherche la rentabilité sans relâche, ne t'arrête pas avant de l'avoir
trouvée, et présente-moi l'agenda de dépôt de cryptos sur le compte réel qui conviendra au
mieux. »

Ce document ne contient que des chiffres **mesurés**, avec la commande ou le fichier:ligne qui
les produit. Tout ce qui est incertain est déclaré incertain.

---

## 1. Le chiffre qui manquait depuis 10 mois : la performance FORWARD

Personne n'avait jamais mesuré ce que le slate validé produit **en avant**, sur des données
postérieures à sa validation. Ce n'est pas un nouvel essai (aucun paramètre n'est sélectionné) :
c'est la vérité terrain.

**Méthode** : règle GELÉE rejouée du 2026-07-02 (date de la restructuration) au 2026-08-01, sur
les 7 perps, avec les barrières réelles de la config (MR : stop 2×ATR / TP 3×ATR / max_hold 12
barres ; S2 : hold 24 barres) et le coût honnête de 15 bps RT.

| | n | brut/trade | **net/trade** | total net |
|---|---|---|---|---|
| **Livre entier** | 27 | −16,87 bps | **−31,87 bps** | −860 bps |
| `mean_reversion_regime` | 24 | — | **−40,22 bps** | −965 bps |
| `funding_squeeze_continuation` | 3 | — | **+34,94 bps** | +105 bps |

**Signification** : t = −0,51, p = 0,61, IC 95 % = [−160 ; +97] bps/trade. **Le mois est négatif
mais statistiquement indistinguable de zéro.** Il ne réfute pas le livre ; il ne le confirme pas
non plus. C'est la seule évidence forward qui existe, et elle n'est pas positive.

**Instrument validé** (R2 — on vérifie l'outil avant de croire le résultat) : le stop BTC du
2026-07-23 sort à **−176,4 bps** contre un `stop_pct` de **1,7637 %** calculé par la vraie
fonction du repo. Correspondance exacte.

**Conséquence qui tue P1 par avance** : le **brut** est déjà négatif. Économiser 7-10 bps de
coût maker ne renverse rien. P1 reste utile en fidélité — jamais en sauvetage.

---

## 2. Quatrième défaut de fidélité : S2 tradait 4 symboles jamais validés

Même classe que les trois du 2026-07-10 (frais maker, no-pyramid, plancher de confiance) : le
live diverge de ce qui a été validé, en silence.

**Le fait** : `research_crypto2/audit_s2_funding_squeeze.py:45` → `ASSETS = ["BTC","ETH","SOL"]`.
Le joint-null p=0,0005 — l'**unique** preuve statistique du livre — porte sur **trois** actifs.
Or `funding_squeeze_continuation` était enregistré **sans `symbols=`** → il tournait sur les
**sept**. Le mécanisme de restriction existait pourtant déjà, et servait à trois autres skills.

**Mesuré** (règle gelée, horizon 24 barres, 15 bps RT, 2026-02 → 2026-08) :

| univers | n | moyenne/événement |
|---|---|---|
| **validé** (BTC/ETH/SOL) | 13 | −82,1 bps |
| **hors validation** (AVAX/SUI/ARB/WIF) | 9 | **−739,6 bps** |

~9× pires. ARB : −1 452 bps sur son unique événement. SUI : −727 bps sur 4.

**Ironie de gouvernance** : l'entrée #41 du ledger a CLOS la piste d'élargissement d'univers de
S2 après deux échecs pré-enregistrés — pendant que le live élargissait déjà.

**CORRIGÉ** (commit `8f64653`, TDD, mutation probante, 1080 tests verts).

---

## 3. Cinquième défaut de fidélité : le stop de S2 n'est pas dans sa validation

**Le fait** : `audit_s2_funding_squeeze.py:107` →
`r = close[i + horizon] / close[i] - 1.0 - COST_TAKER_RT`
La validation est un **rendement d'horizon pur** : entrée, sortie 24 barres plus tard. **Aucun
stop, aucun take-profit.** Le live ajoute `stop = 4×ATR` et `TP = 8×ATR`.

**Mesuré sur l'univers validé** (n=13, 2026-02 → 2026-08) :

| exécution | moyenne/événement |
|---|---|
| validée (horizon pur) | −82,1 bps |
| live (avec stop/TP) | −159,8 bps |
| **coût du stop non validé** | **−77,7 bps/événement** (6/13 tronqués) |

**NON corrigé, délibérément.** Retirer un stop est une décision de **sécurité**, pas de
fidélité : l'opérateur a perdu $12 000 en un jour sur une faille d'exécution. Le retrait
exposerait 24 barres de baisse sans plancher. C'est une décision d'opérateur, mesurée et
posée ici — pas un choix d'agent.

---

## 4. Le cheval de trait du livre n'a jamais été validé

`mean_reversion_regime` = **89 % du flux** (24 des 27 trades forward). Or
`docs/SYSTEME_AUTONOME.md:387` l'admet noir sur blanc : *« pas de p-value/joint-null dédiée
trouvée dans le repo, contrairement à S1/S2 »*. Et `research_crypto2/` ne contient aucun
`_mr_*.json` là où S1, S2 et S3 ont chacun le leur.

Son statut vivant est un **héritage**, pas une preuve. Test pré-enregistré et compté au registre
(1911 → 1912, commit `f5452e4`) : `research_crypto2/PREREG_MR_VALIDATION.md`, barre identique à
celle de S2, zéro adoucissement.

### VERDICT : **FAIL** — et il n'est pas marginal

Exécuté sur **8,96 ans** et **1 499 trades poolés** (7 actifs), barrières réelles du live,
coût 15 bps RT. Lu à la lettre du prereg :

| critère pré-enregistré | seuil | mesuré | verdict |
|---|---|---|---|
| `mc_p_max_5seeds` | < 0,05 | **0,4343** | ✗ (8,7× au-dessus) |
| `oos_sharpe_40` | > 0 | **−0,0299** | ✗ |
| joint-null panier `p_max5` | < 0,05 | **0,4345** | ✗ |
| `edge_above_tide_bps` | > 0 | +2,38 bps | ✓ (seul passé) |

**Une entrée au hasard — même mix long/short, mêmes barrières, même coût — bat le signal dans
43 % des simulations.** Le seul critère passé se lit honnêtement : MR perd *un peu moins* que la
marée moyenne, laquelle est elle-même négative après frais. « Moins mauvais que le hasard » n'est
pas un edge.

Net poolé : **−11,14 bps/trade** sur 1 499 trades. SOL seul est positif (mc_p 0,033) — le retenir
serait exactement le retrait post-hoc de symboles que le prereg interdit.

**Vérifié adversarialement** : ré-exécution indépendante avec la fonction canonique
`hermes.validation.triple_barrier` du repo → **chiffre pour chiffre identique** (n=1499,
net −11,1441 bps, sharpe −0,020302, oos −0,029912). Aucun symbole exclu en silence.

**Conséquence, pré-écrite avant le résultat** : MR sort du slate exécutable (proposition à
l'opérateur, jamais un retrait automatique), le livre devient **S2-seul sur BTC/ETH/SOL**, et la
porte des 40 trades passe de ~1,5 mois à **~2,4 ans**.

> **Ce que ça explique** : le Sharpe de 0,0739 du livre n'est pas « un edge faible ». C'est
> l'edge réel de S2 **dilué par 89 % de bruit**. Les −31,87 bps/trade du mois forward sont
> principalement du bruit MR, pas la faillite de S2.

---

## 5. Le fait structurel qui gouverne tout : la statistique n'existe pas

Sharpe du livre = **0,0739** sur 461 marks (`_book.json`). Erreur-type de Lo :
`SE = sqrt((1 + SR²/2)/T)` = 0,740 → **t = 0,10**.

**Années nécessaires pour que ce Sharpe atteigne p < 0,05 : ~705 ans.** Le meilleur sleeve solo
(S3, 0,179) : 122 ans. S2 solo (0,155) : 161 ans.

Ce n'est pas un livre faible — c'est un livre dont la statistique **ne peut pas exister** dans
un horizon humain, à ce débit de trades.

**Puissance de détection** (à partir de sd = 324,7 bps/trade mesuré) :

| edge à détecter | trades requis | durée au débit de 27/mois |
|---|---|---|
| 10 bps | 8 266 | 306 mois |
| 30 bps | 918 | **34 mois** |
| 100 bps | 83 | 3,1 mois |

**Corollaire brutal** : l'économie maker de P1 (7-10 bps) est **sous le seuil de détectabilité
de son propre banc d'essai**. Même si elle marchait, l'appareil de mesure ne pourrait jamais le
prouver.

---

## 6. Ce que S2 seul peut produire — l'arithmétique honnête

Notional **mesuré** en passant de vrais armements S2 dans le vrai moteur de risque :
**14,6 %** de l'équité par trade (et non 18,4 % : S2 utilise un stop 4×ATR, donc un notional
plus petit — ma première estimation était 26 % trop optimiste, corrigée par mesure).

- événements dé-chevauchés : **16,9/an** (115 sur 6,81 ans)
- net winsorisé validé : **+237 bps/événement**
- décote OOS observée : ×0,72 (Sharpe 0,427 in-sample → 0,306 OOS)

| | rendement annuel |
|---|---|
| in-sample | 5,85 % |
| **décoté OOS** | **4,19 %** |

Sharpe annualisé OOS ≈ **1,26** ; probabilité d'année négative ≈ **10 %** (gaussien — les vraies
queues sont plus grosses).

| capital | espérance annuelle |
|---|---|
| $100 | $4 |
| $2 000 | $84 |
| $10 000 | $419 |
| $50 000 | $2 097 |

**Le rendement en pourcentage est indépendant du capital.** Le capital ne change que les
dollars — il ne crée aucun edge.

**Réserve majeure** : sur la fenêtre récente (2026-02 → 08, n=13), l'univers validé donne
**−82,1 bps/événement**, pas +237. Six mois ne réfutent pas 6,81 ans, mais ne rassurent pas.

---

## 7. L'agenda de dépôt — ce que les règles du système exigent

**Je ne suis pas conseiller financier et je ne recommande aucun montant ni aucune date.** Voici
ce que les règles **pré-enregistrées de ton propre système** imposent, et l'arithmétique qui en
découle.

Portes de `stage_0_paper` → `stage_1_micro` (`config/default.yaml`, `advance_when`) :

| porte | cible | observé | statut |
|---|---|---|---|
| `edge_paper_trades` | 40 | **2** | ✗ |
| `walk_forward_sharpe` | 1,0 | — | ✗ |
| `max_drawdown_pct` | 15,0 | — | ✗ |

Le compteur ne retient que les trades du **slate courant** (`count_closed_by_skills`). Donc :

- **si MR reste au slate** : ~27 trades/mois → la porte des 40 tombe en ~1,5 mois ;
- **si MR échoue sa validation et sort** : le livre devient S2-seul sur 3 actifs, ~17
  événements/an → **~2,4 ans** pour atteindre 40.

**La conclusion mécanique, sans détour** : **le système n'autorise aucun dépôt aujourd'hui**, et
la seule porte qui pourrait s'ouvrir vite dépend d'un sleeve dont la validité est précisément ce
qu'on est en train de tester. Si MR échoue, l'agenda honnête n'est pas « dans X semaines » — il
est « pas avant que le débit d'événements ou la force du signal change de nature ».

---

## 8. Ce qui reste vrai, et qui a de la valeur

- **S2 est un vrai edge** : joint-null p=0,0005 sur 6,81 ans, bear-positif, orthogonal. Il est
  petit (~17 évts/an, ~4 %/an décoté) mais il n'est pas du bruit.
- **Deux défauts de fidélité viennent d'être trouvés et l'un est corrigé** — ils privaient S2 de
  son edge (univers non validé : −739,6 bps/évt ; stop non validé : −77,7 bps/évt).
- **La machine d'honnêteté fonctionne** : elle a attrapé, en une session, un bug d'orchestrateur
  qui bloquait tous les trades, deux défauts de fidélité, et l'absence de validation du sleeve
  dominant. C'est rare et c'est réel.
