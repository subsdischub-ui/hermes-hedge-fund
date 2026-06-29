# STREAM 10 v-next — CHECKPOINT

- **Statut** : gouvernance durcie LIVRÉE ; merge du joyau DIFFÉRÉ (0 pilier) ; rien en live.
- **Task 1** : registres migrés gitignoré→TRACKÉ `governance/` (crypto **1891**/fx 53/cross 5) + README + test anti-drift.
- **Task 2** : 5 règles durcies testées `s10_rules.py` (IC-delta, chance-correction, co-active ρ, leak-injection, cluster-null).
- **Task 3** : +4 négatifs S2 au ledger (36) ; AcquisitionMap multi-venue corrigée ; `funding_squeeze` → WEAKLY_SUPPORTED.
- **Task 4** : OOS-forward = précondition dure (`evaluate_with_forward`) ; gp_lock 4e condition, 0 candidat ; lead reversion en pré-reg.
- **Task 5** : spec gate du joyau `s10_jewel_gate.py` (codée+testée synthétique, NON lancée).
- **Task 6** : garde-fous provenance `s10_provenance.py` (stub funding ; formats symbole).
- **Tests** : 80 verts (42+19+10+9), ruff clean ; auto-revue adversariale → 4 MAJEURS corrigés (ZeroDiv eff-N, DSR sr_variance, governance/ committé, tags d1/cot over-block). **DÉCISION MALACHIE** : constantes barre PBO/DSR (§7 FINDINGS) = ma reco, à trancher.
