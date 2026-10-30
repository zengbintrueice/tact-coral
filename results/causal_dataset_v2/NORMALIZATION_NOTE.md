# Which normalization is actually applied?

**Short answer:** every model uses the per-channel mean/std in
`src/configs/config.py` (`SCS_CHANNEL_MEAN`, `SCS_CHANNEL_STD`), which equal
`results/infrastructure/train_normalization_stats.json`. These values were
estimated from SCS training weeks (2003–2010, 417 weeks).

## Evidence

`PIECFModel.__init__` builds
`ChannelNormalizer(SCS_CHANNEL_MEAN, SCS_CHANNEL_STD)` and registers the
values as buffers (`src/models/piecf_model.py:192–193`). Loading a checkpoint
restores those buffers. Verified in all shipped weights:

- `results/model_final_Btail3/model.pt`
- `results/physics_anchored/model.pt`
- `results/physics_anchored/model_seed_0..42.pt`
- `results/transfer_coordinate/weights/{GBR,CARIB}/{tscf,tsar}_model.pt`

All carry the same `channel_norm.mean/std` (SST mean 27.97262; SST-anomaly
mean −0.10284).

## `normalization_v2.json` is not read by any model

`results/causal_dataset_v2/normalization_v2.json` is written by
`build_causal_dataset_v2.py` as a record of the causal-v2 dataset's own
statistics. No training or evaluation script loads it.

It matches the applied values in 12 of 13 channels. The exception is
`sst_anomaly`:

| | applied (`config.py`) | `normalization_v2.json` |
|---|---:|---:|
| mean | −0.102840 | −0.003835 |
| std | 0.484529 | 0.457673 |

The difference is consistent with the channel being recomputed after the file
was generated (the config comment records an `sst_anomaly` unit/climatology
update). The paper's statement — normalization statistics are estimated from
SCS training data only and frozen — is correct for the applied values.

## If you need to reproduce bit-for-bit

Use `config.py` / `train_normalization_stats.json`. Do **not** substitute
`normalization_v2.json`; doing so would change one channel's z-score
(by ≈0.2 SD) and is not the protocol used for any reported result.
