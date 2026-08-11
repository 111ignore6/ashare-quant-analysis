# 现代模型调研与首批落地（2026-08-11）

> 模拟研究，仅用于数据分析与学习，不构成投资建议。

## 结论摘要

用 sciverse 论文库 + GitHub 调研了一轮 2021-2025 的选股/时序模型，选出三
个**能直接落进现有流水线**的快速胜利并已跑通全市场 walk-forward 评测：

| 模型 | 年化收益 | 夏普 | 最大回撤 | 胜率 | 平均IC | 定位 |
|---|---|---|---|---|---|---|
| **rank_lgb（LightGBM排序/LambdaRank）** | 57.1% | 2.47 | **-3.6%** | 80% | **0.097** | 直接优化横截面排序，IC 全场最高、回撤最低 |
| **xgb（XGBoost）** | 57.1% | 2.55 | -7.0% | 80% | 0.087 | 与 LGBM/HistGB 同族异构，集成多样性 |
| **rank_ensemble（自研：截面排名均值融合）** | **77.9%** | 2.36 | -8.6% | **86.7%** | - | 多模型预测转百分位排名再平均，ML 中年化最高 |
| lgbm（旧基线） | 65.9% | 2.75 | -8.0% | 86.7% | 0.096 | 现役主力 |
| histgb（旧基线） | 64.6% | 2.08 | -10.5% | 80% | 0.078 | 现役主力 |
| ensemble(lgbm+histgb+rf)（旧） | 72.3% | 2.39 | -11.1% | 80% | - | 现役融合 |
| lgbm+conformal_gate（旧） | 82.1% | **3.75** | -6.4% | 83.3% | - | 不确定门控（6 期） |

> 口径：全市场 5335 只 × 3 年日线（修正后密集数据），walk-forward
> （训练 15 月/验证 6 月/步进 6 月），月度调仓 Top-50 等权。

## 论文依据（sciverse 检索）

### 选股/因子方向（与"截面排序"任务最对口）

1. **Temporal Routing Adaptor（TRA, arXiv 2021）**：轻量路由模块，用最优传输
   把样本分派给多个"模式预测器"，在真实股票排序任务上把 IC 从 0.053 提到
   0.059（对比 Attention-LSTM/Transformer）。→ 可作我们集成上的轻量插件。
2. **AlphaNet 系列（华泰证券，Qlib 官方实现）**：LSTM/Transformer 因子挖掘，
   AlphaNetV4 用 Bi-LSTM + Transformer + Spearman Dropout 进一步改进
   （IJSRM 2024）。Qlib 模型库含 ALSTM/GRU/LSTM/GATs/TRA 等现成实现。
3. **FactorVAE（AAAI 2022）**：VAE 概率动态因子模型，预测横截面收益。
4. **UMI（KDD 2025）**：Transformer + 图注意力 + RankIC 损失，引入市场非理性
   因子，中美市场验证，A 股 5123 只。→ 方向新，但需要图/自监督预训练。
5. **QuantFormer（arXiv 2404.00424, ESWA 2026）**：Transformer 因子 + 情绪
   迁移学习，A 股 4601 只 2010-2023。→ 需要新闻情绪数据，暂缓。

### 现代时序骨干（可作特征编码器）

6. **iTransformer（ICLR 2024 Spotlight, 2.2k★）**：倒置注意力，多变量时序。
7. **PatchTST（ICLR 2023）**：patch 化 Transformer。
8. **TimesNet（ICLR 2023）**：1D→2D 周期建模。
9. **DLinear（AAAI 2023）**：线性分解，简单且强。
10. **Timer-XL / TimeMixer / TimeFuse（2024-2025）**：大模型时序/样本级自适应融合。

### 表格基础模型

11. **TabPFN v2（Nature 2025, priorlabs/tabpfn）**：表格基础模型，prior-fitting
    transformer，小样本强。→ 可无训练直接换 GBDT，但 4 万样本 × CPU 较慢。

## 可行性评级（针对本项目：15 特征 × 5335 股 × 3 年，CPU-only）

| 候选 | 评级 | 说明 |
|---|---|---|
| rank_lgb（LambdaRank） | ✅ 已落地 | LightGBM 自带，按日期分组直接优化排序，零新依赖 |
| xgb | ✅ 已落地 | 已安装，集成多样性 |
| rank_ensemble（排序融合） | ✅ 已落地 | 自研，多模型截面排名均值 |
| GRU/ALSTM（Qlib 式） | 🟡 可做 | torch 已装，CPU 慢；benchmark 已有 `--with-dl` GRU 骨架 |
| 保形门控加新模型 | 🟡 可做 | conformal 框架现成，换模型即可 |
| TabPFN v2 | 🟡 试装 | 需 pip 装 + 权重下载，CPU 推理慢，先小样本验证 |
| iTransformer/PatchTST | 🟠 较重构 | 需按 symbol×time×feature 张量化，先做 GRU 再升级 |
| TRA 路由 | 🟠 较重构 | Qlib 有实现可移植 |
| UMI / QuantFormer / RD-Agent | 🔴 暂缓 | 需图/情绪/多智能体，超本项目数据范围 |

## 推荐路线图

1. **短期（本周）**：把 `xgb`、`rank_lgb` 加入每日决策模型池，`rank_ensemble`
   作为候选融合方案再复验（换到更长验证窗口看是否过拟合）。
2. **中期**：`--with-dl` 的 GRU/ALSTM 在全市场子集（如沪深300）上评测；
   尝试 TabPFN 小样本对照；把保形门控套到 rank_lgb 上（低回撤 × 不确定过滤）。
3. **长期**：按 symbol×time×feature 张量实现 iTransformer/PatchTST 编码器 +
   截面排名头（RankIC loss，参考 UMI）；数据侧补齐行业/市值因子。

## 已改代码

- `ashare_quant/ml/models.py`：注册 `xgb`、`rank_lgb`（LambdaRank 包装，按日期
  分组、十分位 relevance）。
- `ashare_quant/ml/benchmark.py`：新增 `rank_ensemble_returns`（截面排名均值
  融合，成员 lgbm/histgb/rf/xgb/rank_lgb）。
- `dashboard.py`：新模型中文名映射。
- `tests/test_models.py`：新模型注册与可拟合回归测试。

> 注意：以上为研究评测，**每日实盘决策仍用原 6 模型**（lgbm/histgb/rf/svm/
> knn/linear）；是否把新模型纳入正式决策待你确认后执行。

## 第二轮：自研模型批量落地 + 实盘切换（2026-08-11 晚）

### 新增模型（全市场 walk-forward 复测）

| 模型 | 年化 | 夏普 | 最大回撤 | 胜率 | 平均IC | 说明 |
|---|---|---|---|---|---|---|
| **risk_aware_lgb（自研）** | 66.7% | **2.81** | -8.3% | 86.7% | 0.096 | 双头模型：收益÷(1+风险)，夏普全场 ML 最高 |
| xgb | 57.1% | 2.55 | -7.0% | 80% | 0.087 | 集成多样性 |
| rank_lgb（LambdaRank） | 57.1% | 2.47 | **-3.6%** | 80% | **0.097** | IC 最高、回撤最低 |
| temporal_decay_lgb（自研） | 62.4% | 2.42 | -8.8% | 80% | 0.092 | 时间衰减加权，适应状态漂移 |
| huber_lgb | **78.5%** | 2.34 | -6.6% | 73.3% | 0.095 | Huber 损失，年化 ML 最高 |
| rank_ensemble（自研融合） | 77.9% | 2.36 | -8.6% | 86.7% | - | 截面排名均值融合 |
| agreement_ensemble（自研） | 74.9% | 2.23 | -7.8% | 80% | - | 排名均值−分歧惩罚 |
| ic_rank_ensemble（自研） | 72.5% | 2.23 | -9.3% | 86.7% | - | 滚动 IC 加权排名融合 |
| rank_xgb | 64.6% | 1.69 | -11.4% | 73.3% | 0.010 | XGB pairwise 排序效果差，已排除 |
| mlp_deep / pls / enet | ≤1.06 夏普 | | | | | 数据量下深网/线性类不敌树模型 |
| lgbm（旧主力） | 65.9% | 2.75 | -8.0% | 86.7% | 0.096 | 保留 |

### 已切换实盘（config.yaml `models` 字段，新字段）

```yaml
models: [lgbm, xgb, rank_lgb, huber_lgb, risk_aware_lgb, temporal_decay_lgb]
```

- **决策引擎升级为"截面排名均值融合"**（`ml/decision.py`）：每个模型预测转
  当日横截面百分位排名再平均，量纲无关，rank_lgb 可安全参与；`score` 列
  改为各模型预测**中位数**（展示用），避免 rank 分数量纲污染"预期收益"。
- 已重训并生成 08-11 新决策（models 6 个，50 只）；账户累计仍 -1.40%
  （08-11 收盘口径，新持仓自明日开始体现收益）。
- 旧 6 模型（svm/knn/linear/mlp/histgb/rf 中的弱项）退出实盘但保留注册表，
  可在 `算法对比` 页继续查看。

### TabPFN v2：受阻

已安装，但 v2（tabpfn_3）是 HuggingFace **gated 模型**，需要
[接受许可](https://huggingface.co/Prior-Labs/tabpfn_3) + `hf auth login`
后才能下载权重。CPU 上 4 万样本推理也很慢。→ 需用户登录 HF 后另行评估。

### 深度时序（AlphaNet/iTransformer/PatchTST/TRA）

benchmark 已有 `--with-dl` 的 GRU 骨架（CPU 全市场偏慢）；iTransformer/
PatchTST 需按 symbol×time×feature 张量化重构，列为下一阶段（路线图不变）。
