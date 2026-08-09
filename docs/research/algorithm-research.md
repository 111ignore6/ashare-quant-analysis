# 现代量化算法研究报告（前沿选型与可行性）

> 模拟研究，仅用于数据分析与学习，不构成投资建议。

## 1. 背景

初版模型库以经典方法为主（反转/低波/动量/多因子 + GARCH/HMM/协整），数量少、方法偏传统。本报告通过 SciVerse 学术检索 MCP（`sciverse-mcp-server`）与公开论文库，筛选 2023~2026 年金融机器学习前沿方法，重新设计算法库（20+ 算法），并给出本机（无 GPU、Python+MATLAB R2025b）的落地可行性。

## 2. 检索方法

- 工具：SciVerse MCP（`scripts/sciverse_search.py`，stdio 直连 `~/.codex/config.toml` 中配置的 `sciverse-mcp-server`）；
- 检索主题：alpha 因子挖掘强化学习、时间序列基础模型、图神经网络选股、保形预测交易、金融大模型、深度学习 A 股预测；
- 关键结论：2023 年后方法重心从"单序列预测"转向"**横截面排序 + 关系建模 + 不确定性量化 + 自动化因子挖掘**"，与我们的"全市场选股"场景高度契合。

## 3. 现代算法库（20+）

### A. 自动化因子挖掘（取代手工因子）

| 算法 | 依据 | 说明 |
| --- | --- | --- |
| A1 强化学习公式化因子挖掘 | Generating Synergistic Formulaic Alpha Collections via RL（KDD 2023, DOI 10.1145/3580305.3599831） | RL 搜索因子表达式组合，自动生成"因子集合" |
| A2 卷积核 Alpha 挖掘 | Mining Profitable Alpha Factors via Convolution Kernel Learning（2023, DOI 10.1007/s10489-023-05014-4） | 用卷积核在价格序列上自动学因子 |
| A3 LLM+RL 因子筛选 | Alpha-R1（2025, arXiv:2512.23515） | 大模型推理 + RL 验证筛因子 |
| A4 分层强化学习因子挖掘 | Mining Intraday Risk Factor Collections via Hierarchical RL（2025, arXiv:2501.07274） | 期权/分层的风险因子自动挖掘 |
| A5 Alpha158/Alpha360 特征集 | 微软 Qlib | 158/360 维标准化量价特征，事实标准 |

### B. 现代时间序列/深度模型

| 算法 | 依据 | 说明 |
| --- | --- | --- |
| B1 iTransformer | ICLR 2024（thuml/iTransformer） | 倒置 Transformer：把"变量"当 token，适合多股横截面 |
| B2 PatchTST | ICLR 2023 | Patch 分块注意力，长序列效率高，已有金融组合应用 |
| B3 TFT 时序融合 Transformer | 经典但仍是解释性最强的 SOTA 之一 | 静态/动态协变量 + 注意力，可解释 |
| B4 MambaStock | arXiv:2402.18959（2024） | 状态空间模型（SSM），线性复杂度长序列 |
| B5 时序 KAN | pTKAN / TimeMixer-KAN（2024） | Kolmogorov-Arnold 网络，非线性拟合更强 |
| B6 时序基础模型 FinCast/Chronos/TimesFM | FinCast（2025, arXiv:2508.19609）；Chronos 金融多变量（2026, arXiv:2605.21504） | 预训练零样本迁移，小样本场景友好 |
| B7 市场引导 Transformer MASTER | AAAI 2024 | 多股瞬时/跨时相关 + 市场信息引导特征 |
| B8 LSTM/GRU | A 股 2013-2024 实证（Financial Innovation 2026） | 作为深度基线保留 |

### C. 关系/图模型（利用股票间关联）

| 算法 | 依据 | 说明 |
| --- | --- | --- |
| C1 TRRM 时序关系排序 | TOIS 2019 | 股票关系图的时序排序学习（经典基线） |
| C2 TD-HCN 趋势驱动超图卷积 | Neural Networks 2025 | 超图建模多股高阶关系 |
| C3 TSCGN 时序相似约束图网络 | IJCSE 2026（DOI 10.1504/IJCSE.2026.152471） | 相似性约束图 + 排序损失，选股提升明显 |
| C4 行业关联 GNN 波动预测 | 2025 | 行业图上的波动率传导 |

### D. 大模型 / 文本

| 算法 | 依据 | 说明 |
| --- | --- | --- |
| D1 FinGPT | arXiv:2306.06031（2023） | 金融领域 LLM，情感/预测，RLSP 收益对齐 |
| D2 ChatGPT 增强 GNN | arXiv:2306.03763（2023） | LLM 生成关系先验注入图模型 |
| D3 Alpha-R1（同 A3） | 2025 | LLM 推理参与因子筛选 |

### E. 不确定性量化与风控（现代重点）

| 算法 | 依据 | 说明 |
| --- | --- | --- |
| E1 保形预测交易门控 | Conformal Prediction for Reliable Stock Selections（2025）；Conformal Kelly（2026, arXiv:2608.01494） | 预测区间校准，区间窄且方向明确才交易；用区间宽度做仓位 |
| E2 在线保形 + 通用组合 | Online Conformal Prediction via Universal Portfolio Algorithms（2026, arXiv:2602.03168） | 在线校准，无需分布假设 |
| E3 深度集成/MC Dropout 不确定性 | 深度学习不确定性文献 | 多模型方差作为置信度 |

### F. 强化学习决策

| 算法 | 依据 | 说明 |
| --- | --- | --- |
| F1 FinRL DQN/PPO | AI4Finance FinRL / FinRL-Meta（2024） | 把"选股+仓位"建成 Gym 环境，策略学习 |
| F2 RLSP 收益对齐 | FinGPT（2023） | 用真实收益信号对齐预测 |

### G. 混合/集成（本项目的创新主线）

| 算法 | 思路 |
| --- | --- |
| G1 **状态感知混合路由** | HMM/GARCH 识别市场状态（平静/动荡/极端），按状态路由到反转/动量/低波/ML 子模型并加权——融合我们已有实证（动荡态收益为负） |
| G2 **保形不确定性门控集成** | 多模型打分 → 保形区间 → 仅当预测可靠时进入选股池，区间宽度决定仓位（Conformal Kelly） |
| G3 IC 衰减自适应因子权重 | 滚动 IC/ICIR 在线调整因子与模型权重（接续现有 rotation 机制） |
| G4 Stacking/Voting 集成 | 基学习器 = 传统 ML + 深度模型 + 因子模型（参考 FinRL Contest 2024 集成） |

## 4. 旧算法的新定位

不再当主角，转为"组件"：

- GARCH/EGARCH：产出波动率特征与仓位约束，喂给 ML/门控；
- HMM：市场状态识别（G1 的路由器）；
- 协整配对：作为独立的统计套利信号源之一；
- LightGBM/XGBoost：仍是结构化因子上最稳的基学习器（Qlib 事实标准），保留为 G4 的一员。

## 5. 本机实证结果（MATLAB R2025b + 真实数据）

基于沪深 300 日收益（截至 2026-08-07）：

- **GARCH(1,1)**：α=0.0745，β=0.9185，**波动持久性 0.993**——波动聚集极强，符合全市场 R2 结论；
- **HMM 2 状态**：平静态占 99.2%（日均 +0.022%），动荡态占 0.8%（日均 −0.097% 且波动更高）——**高波动期收益为负**，支持"状态路由 + 高波动防守"；
- **协整检验**（浦发 vs 兴业）：p=0.083，未达 5% 显著——配对池需扩大后再评估。

## 6. 分阶段实施路线

| 阶段 | 内容 | 主要技术 |
| --- | --- | --- |
| B 特征工程升级 | Alpha158 简化版（量价/波动/形态/状态特征） | Python |
| C1 因子挖掘 | A1/A5：RL 因子挖掘（简化版）+ Alpha158 | Python + LightGBM |
| C2 现代模型库 | B 类 6~8 个 + C 类 2 个 + E 类 2 个 | PyTorch（CPU 小规模）/ 可降采样 |
| C3 传统/统计组件 | GARCH/HMM/协整/Kalman 作为组件 | MATLAB + Python |
| C4 集成与门控 | G1~G4：状态路由 + 保形门控 + IC 自适应 + Stacking | Python |
| D 模拟投资决策引擎 | 每日信号→仓位→模拟成交→决策日志 | 现有回测引擎扩展 |
| E 数据展示 | Streamlit 仪表盘（净值/持仓/因子/状态/决策） | Streamlit |
| F 打包 | PyInstaller 桌面应用（后续） | PyInstaller |

## 7. 风险与限制

- 无 GPU：深度模型在全市场 5000+ 股票上需控制规模或降采样（先用沪深 300/中证 800 训练，全市场推理）；
- LLM 类（D 类）依赖外部模型与算力，列为可选研究项；
- 过拟合是金融 ML 第一风险：一律走滚动样本外验证 + 保形区间 + 参数敏感性检查；
- 因子挖掘类方法需防"数据挖掘偏差"，保留样本外复验环节；
- 本报告全部为模拟研究，不构成投资建议。
