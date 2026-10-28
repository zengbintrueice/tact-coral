# domain_shift/ — which file feeds the paper

This directory contains TWO correlation-structure distance records that are
both labelled "Frobenius distance" but are computed on DIFFERENT feature
sets. They are not interchangeable.

| File | Feature set | SCS test | GBR test | CARIB test |
|---|---|---|---|---|
| `joint_shift.json` | 13 dynamic channels (the model input set) | 0.867 | 4.722 | 2.300 |
| `joint_structure.csv` / `joint_structure.md` | 16 features (13 channels + 3 extra columns used by the marginal analysis) | 1.066 | 4.863 | 2.372 |

**The manuscript and Supplementary Table 4(b) quote the `joint_shift.json`
values (0.867 / 4.722 / 2.300)**, i.e. the 13-channel record that matches the
model input space. `joint_structure.*` is retained because the marginal-shift
analysis (Supplementary Table 4(a)) uses the 16-feature framing.

Mean absolute off-diagonal |Δcorr| for the paper's record
(`joint_shift.json`): 0.050 (SCS test), 0.247 (GBR test), 0.135 (CARIB test).
