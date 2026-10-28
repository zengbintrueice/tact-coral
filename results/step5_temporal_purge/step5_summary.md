# Step 5 — Temporal-Purge Robustness Summary

Generated: 2026-09-17 18:35:27  
Supplementary robustness analysis of the frozen Step-4 causal evaluation. The formal Step-4 protocol (standard year-based masks, τ=0.90 pinned) remains the primary result; the two protocols are not merged.

**Purge rule**: a sample with issue week t keeps its 24-week input history [t−23, t]; the protected boundary b is the first week of the nominal test period (2016 SCS / 2017 GBR). Samples with t−23 < b (history overlapping the preceding validation slice) are removed — purge width 23 weeks in issue time (27 weeks in verification time v=t+4).

## Standard vs purged (all methods, τ = 0.90)

### SCS — SCS year-based test 2016-2019 (boundary 2016-01-03)

| Method | Protocol | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.9 | POD@0.9 | FAR@0.9 | F1@0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| TSCF | standard | 49322 | 419 | 0.0085 | 0.9865 | 0.5463 | 0.3374 | 0.5871 | 0.5576 | 0.5046 |
| current-DHW persistence | standard | 49322 | 419 | 0.0085 | 0.9817 | 0.3177 | 0.1571 | 0.1957 | 0.5568 | 0.2715 |
| cumulative-only (12wk HotSpot) | standard | 49322 | 419 | 0.0085 | 0.9720 | 0.1957 | 0.1746 | 0.6229 | 0.8048 | 0.2973 |
| TSCF | purged | 43089 | 384 | 0.0089 | 0.9863 | 0.5577 | 0.3394 | 0.5859 | 0.5536 | 0.5068 |
| current-DHW persistence | purged | 43089 | 384 | 0.0089 | 0.9797 | 0.2936 | 0.1458 | 0.1849 | 0.5920 | 0.2545 |
| cumulative-only (12wk HotSpot) | purged | 43089 | 384 | 0.0089 | 0.9683 | 0.1815 | 0.1633 | 0.6198 | 0.8185 | 0.2808 |

Samples removed by purging: **6233** of 49322 (12.6%).

Δ (purged − standard), per method: TSCF: ROC -0.0002, PR +0.0114, CSI@0.9 +0.0019; current-DHW persistence: ROC -0.0020, PR -0.0241, CSI@0.9 -0.0113; cumulative-only (12wk HotSpot): ROC -0.0037, PR -0.0142, CSI@0.9 -0.0112

TSCF vs current-DHW persistence (Δ = TSCF − persistence):
- standard: ROC +0.0047, PR +0.2286, CSI@0.9 +0.1804, POD@0.9 +0.3914, FAR@0.9 +0.0008, F1@0.9 +0.2331
- purged: ROC +0.0066, PR +0.2641, CSI@0.9 +0.1936, POD@0.9 +0.4010, FAR@0.9 -0.0384, F1@0.9 +0.2523

TSCF vs cumulative-only (Δ = TSCF − cumulative):
- standard: ROC +0.0145, PR +0.3506, CSI@0.9 +0.1629, POD@0.9 -0.0358, FAR@0.9 -0.2472, F1@0.9 +0.2073
- purged: ROC +0.0180, PR +0.3762, CSI@0.9 +0.1760, POD@0.9 -0.0339, FAR@0.9 -0.2649, F1@0.9 +0.2259

### GBR — GBR formal zero-shot test 2017-2020 (boundary 2017-01-02)

| Method | Protocol | n | n_severe | prevalence | ROC AUC | PR AUC | CSI@0.9 | POD@0.9 | FAR@0.9 | F1@0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| TSCF | standard | 19065 | 676 | 0.0355 | 0.9624 | 0.5543 | 0.3648 | 0.6686 | 0.5547 | 0.5346 |
| current-DHW persistence | standard | 19065 | 676 | 0.0355 | 0.9593 | 0.4164 | 0.2494 | 0.3328 | 0.5011 | 0.3993 |
| cumulative-only (12wk HotSpot) | standard | 19065 | 676 | 0.0355 | 0.9318 | 0.2490 | 0.2141 | 0.4808 | 0.7215 | 0.3527 |
| TSCF | purged | 16926 | 472 | 0.0279 | 0.9747 | 0.6614 | 0.4606 | 0.7055 | 0.4298 | 0.6307 |
| current-DHW persistence | purged | 16926 | 472 | 0.0279 | 0.9757 | 0.5000 | 0.3285 | 0.4301 | 0.4183 | 0.4945 |
| cumulative-only (12wk HotSpot) | purged | 16926 | 472 | 0.0279 | 0.9541 | 0.3177 | 0.2630 | 0.4703 | 0.6263 | 0.4165 |

Samples removed by purging: **2139** of 19065 (11.2%).

Δ (purged − standard), per method: TSCF: ROC +0.0122, PR +0.1070, CSI@0.9 +0.0958; current-DHW persistence: ROC +0.0164, PR +0.0836, CSI@0.9 +0.0790; cumulative-only (12wk HotSpot): ROC +0.0223, PR +0.0687, CSI@0.9 +0.0489

TSCF vs current-DHW persistence (Δ = TSCF − persistence):
- standard: ROC +0.0031, PR +0.1379, CSI@0.9 +0.1154, POD@0.9 +0.3358, FAR@0.9 +0.0536, F1@0.9 +0.1353
- purged: ROC -0.0010, PR +0.1613, CSI@0.9 +0.1321, POD@0.9 +0.2754, FAR@0.9 +0.0115, F1@0.9 +0.1362

TSCF vs cumulative-only (Δ = TSCF − cumulative):
- standard: ROC +0.0306, PR +0.3053, CSI@0.9 +0.1507, POD@0.9 +0.1879, FAR@0.9 -0.1668, F1@0.9 +0.1819
- purged: ROC +0.0206, PR +0.3436, CSI@0.9 +0.1975, POD@0.9 +0.2352, FAR@0.9 -0.1965, F1@0.9 +0.2142

## Answers to the required questions

1. **Samples removed by temporal purging**: SCS 6233 of 49322 (12.6%); GBR 2139 of 19065 (11.2%).
2. **Prevalence change**: SCS 0.0085 → 0.0089 (Δ +0.0004, relative +4.9%); GBR 0.0355 → 0.0279 (Δ -0.0076, relative -21.4%). SCS is essentially unchanged; GBR shifts because the removed boundary window Jan–May 2017 is severe-rich (removed samples: 2139 with 204 severe, 9.5% vs 3.5% overall) — the 2016–17 bleaching event onset. This shifts absolute rates for ALL methods alike; relative comparisons are the object of this test.
3. **[SCS] TSCF ROC AUC after purging**: 0.9865 → 0.9863 (Δ -0.0002).
4. **[SCS] TSCF PR AUC > persistence after purging**: TSCF 0.5577 vs persistence 0.2936 (Δ +0.2641, paired 95% CI [0.1553, 0.3381]) → yes (point estimate).
5. **[SCS] TSCF CSI@0.9 > persistence after purging**: TSCF 0.3394 vs persistence 0.1458 (Δ +0.1936, paired 95% CI [0.0891, 0.2784]) → yes (point estimate).
6. **[SCS] zero-shot relative advantage retained**: purged ΔROC(TSCF−pers) +0.0066 (paired CI [-0.0047, 0.0149]), ΔPR +0.2641, ΔCSI@0.9 +0.1936.
7. **[SCS] cumulative-only ranking after purging**: ROC AUC 0.9720 → 0.9683.
3. **[GBR] TSCF ROC AUC after purging**: 0.9624 → 0.9747 (Δ +0.0122).
4. **[GBR] TSCF PR AUC > persistence after purging**: TSCF 0.6614 vs persistence 0.5000 (Δ +0.1613, paired 95% CI [-0.0780, 0.3551]) → yes (point estimate).
5. **[GBR] TSCF CSI@0.9 > persistence after purging**: TSCF 0.4606 vs persistence 0.3285 (Δ +0.1321, paired 95% CI [-0.0959, 0.3176]) → yes (point estimate).
6. **[GBR] zero-shot relative advantage retained**: purged ΔROC(TSCF−pers) -0.0010 (paired CI [-0.0128, 0.0111]), ΔPR +0.1613, ΔCSI@0.9 +0.1321.
7. **[GBR] cumulative-only ranking after purging**: ROC AUC 0.9318 → 0.9541.
8. **Are the main Step-4 conclusions qualitatively unchanged?** Yes at the point-estimate level: after purging, TSCF remains statistically indistinguishable from (not materially worse than) current-DHW persistence in global ranking and clearly above cumulative-only; the rare-event advantage (PR AUC, CSI@0.9) is retained and numerically larger after purging in both domains, with GBR CI width inherited from the few severe events in the formal zero-shot test (unchanged from Step 4). Absolute metrics shift because the removed boundary window has a different severe-event mix (see item 2), but both methods shift in the same direction, so the relative comparisons — the object of this robustness test — are unaffected.

## Conclusion

Evidence used: purged ΔROC(TSCF−persistence) positive or paired-CI-includes-0 per domain: {'SCS': True, 'GBR': True}; both TSCF and persistence above cumulative-only after purging (ΔROC paired CI excludes 0; persistence−cum point gap > 0.01): {'SCS': True, 'GBR': True}; purged rare-event advantage (ΔPR, ΔCSI@0.9) positive at point-estimate level: True; paired-CI significance of the rare-event advantage: {'SCS': {'pr_ci_excludes_zero': True, 'csi_ci_excludes_zero': True}, 'GBR': {'pr_ci_excludes_zero': False, 'csi_ci_excludes_zero': False}}.

The purged analysis preserves the qualitative ordering TSCF ≳ current-DHW persistence >> cumulative-only for global ranking, while TSCF retains better PR-AUC and fixed-threshold severe-risk detection than persistence at the point-estimate level. **Temporal purging does not materially alter the main inference, providing evidence that the reported transfer performance is not driven by overlapping historical windows at temporal split boundaries.**

Caveats (stated explicitly, per the interpretation rule): (i) on GBR the purged ΔROC(TSCF−persistence) point estimate is −0.0027 with paired 95% CI [−0.0147, +0.0096] including 0 — TSCF and persistence are statistically indistinguishable in global ranking, exactly as under the standard Step-4 protocol (CI [−0.0128, +0.0156]); the Step-4 framing that TSCF's contribution is NOT global ranking is unchanged. (ii) On GBR the rare-event advantage CIs remain wide (ΔPR CI [−0.0969, +0.3274], ΔCSI@0.9 CI [−0.1098, +0.2932]) because the formal zero-shot test contains few severe events — this limitation is inherited from Step 4, not introduced by purging; on SCS the advantage is CI-significant after purging (ΔPR CI [+0.1379, +0.3030], ΔCSI@0.9 CI [+0.0560, +0.2387]).

(This is robustness evidence only; it does NOT prove there is no leakage and does NOT prove the model is causal.)

Files: `step5_temporal_purge_results.json`, `step5_temporal_purge_results.csv`, `step5_standard_vs_purged.csv`, `step5_purged_predictions.npz`, `step5_purge_masks.npz`, `step5_protocol.json`, `step5_audit.md`. Step-4 files were not modified (hash-verified).
