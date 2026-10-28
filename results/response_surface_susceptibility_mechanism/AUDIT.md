# AUDIT — Response Surface → Susceptibility → Mechanism Decomposition

Generated: 2026-10-04 16:15:11
Frozen records md5 `4035a19fd30fa4d14151f4c8055dfe59` (verified).

## Protocol adherence

- Frozen rec_structural_ablation.npz; seed-42 TSAR/PersA; no retraining.
- R = TSAR − PersA; per-basin; complete-case; unified records per part.
- Week-block bootstrap 1000, seed 42; BH-FDR per family per basin.
- Gated pipeline: ①→② requires R-tertile contrast direction-consistent in GBR+CARIB; ②→③ requires ΔROC or ΔPR CI-positive in both targets.
- Associative / representation-level language only.

## Gate results

- ①→② (response heterogeneity): PASS
- ②→③ (susceptibility established): PASS

## Mechanism verdicts (if run)

- SCS: trajectory=True, state_add=True, nonthermal_add=True
- GBR: trajectory=True, state_add=False, nonthermal_add=True
- CARIB: trajectory=True, state_add=True, nonthermal_add=True

## Interpretation guard

All statements are associative / representation-level. 'Susceptibility' means conditional association between the frozen residual and future severe thermal events at matched current DHW; it is not a causal or biological sensitivity claim.

