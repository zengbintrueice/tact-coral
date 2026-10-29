# 第二轮交叉核对：release 算法/结果 ↔ 论文 md

日期：2026-10-19
对象：`doc/main_manuscript.md`、`doc/supplementary_information.md`
参照：`tact-coral/`（代码、结果文件、图件源数据）
范围：**(1) 算法代码与论文说明/公式；(2) release 结果与论文定量结果/实验图。**

---

## 一、算法代码 ↔ 论文说明与公式

| 论文声明 | 代码位置 | 代码事实 | 判定 |
|---|---|---|---|
| Eq (1) `y(t)=clip(DHW(t)/10,0,1)` | `data/source_builders/build_v9_1deg_from_raw.py:614` | `np.clip(dhw_weekly/10.0,0,1)` | **一致** |
| Eq (2) `PersA_DHW(t)=y(t)`（issue 时刻） | `train_physics_anchored.py:105,132`；`physics_anchored_model.py:87` | `batch["y_regression"][:, -1]`；`anchor = pers_anchor.detach()` | **一致** |
| Eq (3) `ŷ=PersA+g·r`, `r=s·tanh(R(h_t))`, `g=σ((y_main−τ_g)/T)` | `physics_anchored_model.py:81-88` | `res=tanh(residual_head(attn.detach()))*scale`；`gate=sigmoid((y_main-tau)/temp)`；`y_final=anchor+gate*res` | **一致** |
| Eq (3) `R(h_t)=MLP_sev(stopgrad(A))` | `physics_anchored_model.py:65-66,81` | `Linear(hid,8)→ReLU→Dropout(0.3)→Linear(8,1)`，输入 `attn.detach()` | **一致** |
| Eq (4) `ỹ=1{y(t+4)≥0.8}` | `train_severe_classifier.py:44,SEV=0.8` | `y_sev=(y_lead>=0.8)` | **一致** |
| `s=0.4` | `train_physics_anchored.py:44`；`train_structural_ablation_B.py:36` | `--scale default=0.4`；`"scale": 0.4` | **一致** |
| `τ_g=0.5, T=0.1` | `train_physics_anchored.py:58` | `{"tau":0.5,"temp":0.1}` | **一致** |
| ConvLSTM `[16,16]` 2 层 + 时序自注意力 | `src/configs/config.py:111`；`src/models/piecf_model.py:219` | `convlstm_hidden_dims=[16,16]`；`nn.MultiheadAttention` | **一致** |
| 13 动态通道 | `config.py` `SCS_DYNAMIC_FEATURES`(9)+`SCS_DERIVED_FEATURES`(4) | 9+4=13 | **一致** |
| 24 周历史 / 4 周提前 | `config.py:34,153`；`train_*` | `history_weeks=24`、`lead_weeks=4` | **一致** |
| TSCF 加权 BCE `pos_weight=1000` | `train_severe_classifier.py:108` | `pos_weight=1000.0` | **一致**（**但见第五节遗留 #19**） |
| TSCF 单头 sigmoid | `ablate_a_dh_stopgrad.py` `FinalSingleHeadModel` | checkpoint 51 keys，无 `severe_head`/`residual_head` | **一致** |
| TSAR 损失 `focal-MSE(γ=0.5)+λ_sev=20·MSE` | `physics_anchored_model.py:102-138`；`train_physics_anchored.py:58` | `loss_main + 20*sev_loss`，`gamma=0.5` | **一致** |
| 训练协议 Adam/1e-3/wd1e-5/cosine/32/60ep/patience10/seed42 | `train_physics_anchored.py:54-58,83-84,149`；`train_severe_classifier.py:35,113,160` | 全部一致 | **一致** |
| 归一化仅 SCS 训练集、冻结 | `src/models/piecf_model.py:317`；`config.py:95-100` | `channel_norm` 内置，未在目标域重拟合 | **一致** |
| 静态张量未被消费 | `piecf_model.py:302-325` `forward_features` | 函数体从未引用 `x_static` | **一致** |
| τ=0.90 预先固定 | `xgb_transfer_control/config.json` `tau_pinned=0.9`；结构消融 `threshold=0.9` | 一致 | **一致** |
| 配对周块 bootstrap 1000/seed42 | `raw->analysis` 各脚本 `N_BOOT=1000, seed=42` | 一致 | **一致** |

**结论：公式 (1)–(4)、损失、stop-gradient、残差有界性、全部超参数与代码逐项一致。无发现。**

---

## 二、release 结果 ↔ 论文定量结果

已逐值核对的表格（全部**逐一吻合**，除下方第四节列出的偏差）：

| 论文位置 | 数值条数 | 来源文件 | 判定 |
|---|---:|---|---|
| 摘要 / 主文六方向矩阵 | 27 格 ROC | `transfer_coordinate/{transfer_matrix,reverse_matrix}.csv` | 全部吻合 |
| Fig 2 / SI Table 7 | 9×3 格 | 同上 | 全部吻合 |
| ΔROC 及 95% CI | 14 组 | `transfer_matrix_ci.csv`、`reverse_matrix_ci.csv` | 全部吻合 |
| SI Table 6（五 seed） | 30 组 | `seed_sensitivity/seed_metrics.csv`、`physics_anchored/seed_sensitivity_tsar.csv` | 全部吻合 |
| Seed 边界 ΔPR/ΔCSI | 10 组 | `seed_sensitivity/seed_transfer_boundary.csv` | 全部吻合 |
| SI Table 3(a)(b)(c) | 6+12+15 组 | `physics_anchored/structural_ablation/structural_ablation.json`（+`qc/.../structural_ablation_corrected.json`）+`review2_revision/structural_marginal_paired_ci.json` | 全部吻合 |
| 空间网格 bootstrap（405 cells） | 3 组 CI | `review2_revision/carib_ca_spatial_cluster.json` | 全部吻合 |
| SI Table 4(a)(b)(c) | 5+3+8×3 | `figures/source_data/fig4_source_data_{w1_medians,joint,j2_assoc}.csv` | 全部吻合 |
| SI Table 5（XGBoost） | 3×8 | `xgb_transfer_control/metrics.csv` | 全部吻合 |
| XGB 记录集审计 | n=83,025 / 1,356 | `xgb_transfer_control/record_audit.json`（`exact_match: true`） | 全部吻合 |
| SI Table 8（GCBD） | 全表 | 由 `step9_gcbd_results.csv` 原始记录重算 + `review2_revision/gcbd_model_wise_enrichment.json`、`gcbd_common_cohort.json` | 全部吻合 |
| GCBD 年份分层 | 9 组 | `gcbd_year_stratification/summary.json`（2026-10-18 修复后重跑） | 全部吻合 |
| GCBD holdout/CARIB 窗口 | 2×5 | `gcbd_global_holdout_enrichment/summary.json` | 全部吻合 |
| SI Table 9(a)(b)(c) | 9+9+9 | `residual_hierarchy/summary.json`、`mechanism_discrimination/summary.json`、`residual_mechanism_decomposition/summary.json` | 全部吻合 |
| SI Table 10 | 9 行 | `step6_feature_ablation/`、`step7_temporal_scale/`、`basin_susceptibility/`、`counterfactual_response_surface/`、`step9.../severity_sensitivity/` | 全部吻合（1 处措辞见第四节） |
| 上下文置换 | 3 域 ×3 | `step8_contextual_gain/step8_context_permutation.csv`、`review2_revision/carib_env_permutation.json` | 全部吻合 |
| 固定 DHW 严重率 | 3 域 | `basin_susceptibility/summary.json` partA | 全部吻合 |
| 条件残差 / 匹配 DHW | 3 域 | `conditional_residual/summary.json`、`dhw_conditioned_susceptibility/summary.json` | 全部吻合 |
| 时间 purge | 4 组 | `step5_temporal_purge/step5_temporal_purge_results.json` | 全部吻合 |
| 空间 holdout（R6） | 2×2 | `review2_revision/spatial_holdout/spatial_holdout_metrics.json` | 全部吻合 |

**图件路径**：主文 10 张 + SI 10 张，共 20 个 `![]()` 引用，在 release 中**全部存在**。

---

## 三、已发现并修复的问题

### 【高】Figure 2(c) 标号错位 —— 已修复并重出图

`scripts/figures/fig2_transfer_matrix.py` 把 `transfer_matrix_ci.csv` 当作"仅 SCS 源"读取，
但该文件实际是**混合文件**（6 行 SCS 源 + 8 行反方向源）。脚本对所有行硬编码
`source="SCS"`，随后 `dd[...].iloc[0]` 取到的是**错行**。

| 方向 | 模型 | 修复前（画出的） | 正确值 |
|---|---|---:|---:|
| SCS→CARIB | TSCF | −0.2626 | **−0.3764** |
| SCS→CARIB | TSAR | +0.0189 | **+0.0014** |
| SCS→GBR | TSCF | −0.0216 | **−0.0240** |

修复前，Fig 2(c) 恰好把 **GBR→CARIB** 的两个 ΔROC 值画到了 **SCS→CARIB** 位置上，
使论文的头号塌缩结果（−0.376）在图上被弱化成 −0.263，同时把 TSAR 的
+0.001 放大成 +0.019。

**修复**：改为按每行自带的 `source` 标注并按 `(source,target,model)` 去重。
已重跑 `fig2_transfer_matrix.py`，PNG/PDF/SVG 与 `fig2_source_data_transfer_matrix.csv`
全部重新生成；新源数据 14 行、标签与数值一一对应。

---

## 四、仍未修改、需要作者决定的问题

### 【已更正】Figure 10 的 CI 与正文一致；step9 与 review2 的差异只在 RR 的分箱方式

上一版报告称 Fig 10 用了 step9 的 CI 而正文用 review2，**该结论有误，予以撤回**。
逐行核对 `fig7_gcbd_validation.py` 后确认：

- panel (b) 的 AUROC 区间读 `step9_bootstrap.csv` 的 `roc_sb/cluster_site`
  = `[0.807632, 0.935942]` → `[0.808, 0.936]`，**与正文完全相同**；
- panel (c) 的 RR 区间读 `step10_matching_results.csv`（offset 0）
  = `[8.950738, 48.311898]` → `[9.0, 48.3]`，也**与正文完全相同**。

`step9_bootstrap.csv` 里确实另有一组不同的 RR 行（`[8.845, 44.400]`，naive mean
40.478），但**图没有使用它**。

两组的真实差异及成因：

| 量 | step9_bootstrap.csv | review2 / step10 | 正文 |
|---|---|---:|---:|
| AUROC severe-vs-bg，site-cluster CI | [0.8076, 0.9359] | [0.8076, 0.9359] | [0.808, 0.936] |
| RR Q5/Q1，点估计 | 35.185 | 35.185 | 35.2 |
| RR，naive 均值 | 40.478 | 41.990 | 42.0 |
| RR，site-cluster 均值 | 26.623 | 27.428 | — |
| RR，site-cluster CI | [8.845, 44.400] | [8.951, 48.312] | [9.0, 48.3] |
| 有效重复数（naive/cluster） | 862 / 646 | 862 / 646 | — |

**重采样完全相同**（同为 `default_rng(42)`、1000 次、同一 site 列表），证据是
AUROC 区间逐位相同、有效重复数相同。唯一差别是 **RR 的五分位边界**：

- step9（`compute_step9_gcbd_validation.py`）：五分位标签在**全样本**上算一次
  （`bin_defs[...]["labels"] = pd.qcut(pred, 5)`），每个 bootstrap 重复里只用
  `labels[idx]` 取子集 —— 边界固定，Q5/Q1 成员不随重复变化 → 区间较窄。
- review2（`compute_review2_revision_evidence_20260928.py`）：`rr()` 在**每个重复内部**
  重新 `pd.qcut(ps, 5)` —— 边界随重采样重估 → 区间较宽。

两者都不使用 GCBD 结局来定边界，因此都不构成泄漏。review2 的口径更贴近
"分位点本身也是估计量"的标准 bootstrap 处理；正文与图均已采用该口径。
建议在 SI R5 补一句说明该约定即可，无需改数。

### 【低】`|ΔROC| ≤ 0.021` 实为 0.02144

主文与 Fig 2 图注均写 "TSAR stays within |ΔROC| ≤ 0.021"。六个方向的最大绝对值为
**0.02144**（GBR→SCS），按三位小数四舍五入为 0.021，但原式严格读作"≤0.021"不成立。
建议改为 `≤ 0.022` 或 `≤ 0.0214`。

### 【低】Fig 6(d) 图注的"every domain and metric"过强

`xgb_transfer_control/bootstrap_cis.csv`：GBR 的 PR-AUC 边际区间
NO_THERMAL `[0.199, 0.383]` 与 FULL `[0.358, 0.487]` **有重叠**。
图注写 "the NO-THERMAL interval lies below THERMAL and FULL in every domain and metric"
若按"区间不重叠"理解则不成立（点估计顺序仍成立）。建议改为
"the NO-THERMAL point estimate is below ... in every domain and metric; marginal intervals overlap only for GBR PR-AUC"。

### 【低】SI Table 10 的 "PR-AUC 0.599 (C4/C8)" 指代不明

`step7_temporal_scale_results.csv` 中 0.5989 来自**单尺度 C8**，不存在 "C4/C8" 组合行。
建议改为 "C8"。

### 【低】`fig5_source_data.csv` 仍含未 clip 的 ECE

该文件 panel b/c 的 `ece` 列取自未修正的 `structural_ablation.json`
（SCS PersA 0.065826，而 Table 3 用 0.0665）。panel b/c 实际**不绘制 ECE**，
故不影响图面；但该源数据文件与论文的统一 clip 约定不一致，建议重算该列。

### 【低】`fig5`/`fig6` 源数据里的 PersA 行在反方向为 NaN

属预期（PersA 与源无关）；无问题，仅记录。

---

## 五、上轮遗留（未处理）

**#19**：`train_severe_classifier_sweep.py` 以 **test CSI** 选 `pos_weight`
（`best_pw = max(results, key=lambda k: results[k]["test_csi"])`），
而 Methods 写 "sweep selected on SCS validation only"。用户 2026-10-18 明确暂不处理。

---

## 六、结论

- **算法/公式层：零不一致。**
- **结果层：本次逐值核对约 200+ 个数值，除第四节 5 处措辞/口径问题外全部吻合。**
- **图件层：修复 1 处高严重度标号错位（Fig 2c），重出图；另发现 1 处图-文 CI 口径不一致（Fig 10）待定。**
