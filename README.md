# A股量化研究·模拟分析系统

基于真实 A 股数据的历史数据研究项目。

> **数据源链以 `config.yaml` 为准**：主源 `akshare`（新浪），备源 `tencent` → `mootdx`。
> `baostock` 在注册表里但**不在**当前备源链中。

> 本项目为模拟研究，不构成任何投资建议。

## ⚠️ 先读这一条：本项目由 AI 维护

**这个仓库由 AI（编码 agent）在人类机主的指导下持续开发与维护**，不是人类手写的作品集。

这意味着几件事，请在阅读代码与结论前知悉：

- **代码、文档、提交信息、以及绝大多数"审查结论"都出自 AI**，人类机主负责方向与验收；
- **`AGENTS.md` 是 AI 的项目级长期记忆**（工程日志），它记录的是"踩过什么坑、
  哪条结论被后来的证据推翻了"——写法刻意保留原始证据强度，而不是成功叙事；
- **AI 会犯错，本项目选择把错误留在记录里而不是删掉**。文中的
  「未证实」「已推翻」「探针自己说谎」等字样都是真实发生过的事，
  详见 [`docs/HONESTY.md`](docs/HONESTY.md)；
- 因此**请对任何"漂亮"的数字保持默认怀疑**，并按 `docs/HONESTY.md`
  第八节的引用规范核对口径。

### 🔴 最重要的诚实结论（引用任何收益数字前必读）

> **本项目没有证明它具备选股能力。** 2026-09-16 的样本外 A/B 实测：
> Top-50 组合的**日均截面超额为负**，且按重叠样本校正后**统计上不显著**。
> 正确表述是「**未证实有超额**」，不是「已证伪超额」。
> 账户与回测里的高收益（年化 +18%~+39%）**绝大部分是市场 beta，不是 alpha**。

完整清单（含仍未修复的算法问题、未验证事项、以及本项目坚持的判据纪律）
见 **[`docs/HONESTY.md`](docs/HONESTY.md)**。

## 运行

```bash
pip install -r requirements.txt            # 或 pip install -e .（标准包安装）
python -m ashare_quant.cli fetch --universe csi300 --years 3   # 下载沪深300数据
python -m ashare_quant.cli research                            # 生成研究报告
```

## 一键启动（日常使用）

**最省事**：双击根目录 `启动系统.bat` —— 自动做每日增量更新 → 启动仪表盘 → 打开浏览器。
首次使用（还没有数据）会提示先用 `start.bat` 下载。

**推荐使用控制台**：运行 `python -m streamlit run dashboard.py`（或 `.\scripts\start.ps1 dashboard`），
打开 http://localhost:8501 —— 总览页直接显示数据状态、落后股票、模型/持仓状态，
点击按钮即可下载/更新数据（后台运行、输出实时显示），无需记命令。

命令行方式（自动化/计划任务）：

```powershell
.\scripts\start.ps1 daily        # 每日：增量更新 + 报告 + 决策
.\scripts\start.ps1 force        # 强制重算报告与决策
.\scripts\start.ps1 fetch        # 下载/更新全市场数据
.\scripts\start.ps1 simulate     # 模拟盘回测（含反馈调整）
.\scripts\start.ps1 research     # 生成历史数据研究报告
.\scripts\start.ps1 dashboard    # 启动仪表盘（http://localhost:8501）
.\scripts\start.ps1 all          # 完整一条龙：数据→模拟→决策→仪表盘
```

首次使用：全市场下载耗时取决于主源（`config.yaml` 的 `data_source`）。当前主源是
akshare（新浪），既有实测 4~7 只/s（12 并发）→ 全市场约 15~25 分钟；腾讯源快得多
（22 只/s，约 8-15 分钟）但高频会被 WAF 限流，所以当前只作备源。支持断点续传。

每日增量更新的耗时**取决于当天有没有新交易日**（下表为实测；16:05 那次生产的原始记录在
`data/tencent/update_stats.json`）：

| 情况 | 实测耗时 | 说明 |
|---|---|---|
| 无新交易日（指数日期未推进） | 约 2 秒 | `update_daily` 直接返回 `up_to_date=all`，不拉全市场列表 |
| 有新交易日（2026-09-16 修复前） | 阶段 1 `1382.8` s（`total_sec=1447.3`） | 逐只源链：每只至少 1 次 HTTP（akshare 实测 3 请求 / 90KiB / 1.26s） |
| 有新交易日（2026-09-16 之后） | **阶段 1 实测 36.8 s（全市场 5360 只 ≈ 145 只/s）** | 走腾讯批量报价端点（120 只/请求，取数 1.7s），且当日 bar 收盘后立即就有 |

> ⚠️ 口径：上表"修复前"是 16:05 收盘高峰的生产实跑，"修复后"是同日 18:20 空闲时段的
> scratch 实跑（**这一格为一次隔离会话的 scratch 测量，未做第三次独立复测**），两者**不同刻**；
> 另有一组相隔 10 分钟的 300 只**同刻 A/B** 是 **5.3×**（48.68s → 9.14s）。
> 所以不要把这个倍数当成"纯代码收益"——里面含源端发布时点的差异。
>
> 早先这里写「之后每日增量更新通常 1-3 秒」，那只覆盖了第一种情况；有新交易日时是分钟级。
> 还要记住 **`update_stats.json` 里的"成功"字样曾经会撒谎**：2026-09-16 那次它自述
> `updated=4989 / failed=0 / healthy=true / days_behind=0`，但逐文件读 parquet 的日期列，
> 全市场 5360 只里 **5140 只仍停在 09-15、只有 209 只真到 09-16**，当天决策就建立在
> "指数 09-16、多数个股 09-15"的混合日期面板上。根因是"源返回了非空、但不含目标交易日的表
> 也算 updated"（**已修**，见 `AGENTS.md`「关键机制」；修复后改走腾讯批量报价，不再依赖
> 新浪的当日可得性——但"**16:05 那一刻**源是否已发布"本身仍属**未验证**）。
> 判断是否真的更新成功，请直读 parquet 的日期分布，别只看自述指标。

计划任务仍可选用 `.\scripts\schedule_daily.ps1`（周一至五 16:05 自动运行；
改动脚本后要重新注册任务才会生效，见 `AGENTS.md`「已知问题」2）。

## 阶段

1. 数据底座与历史数据研究（已完成）
2. 候选模型构建与筛选（已完成）
3. 模拟盘与反馈调整（已完成）
4. 每日增量更新与自动执行（已完成）

## 每日更新

```powershell
python -m ashare_quant.cli daily          # 增量更新数据并生成当日 HTML 报告
.\scripts\schedule_daily.ps1              # 注册 Windows 计划任务（周一至五 16:05 自动运行）
```

## 全市场模式

```bash
python -m ashare_quant.cli fetch --universe all --data-root data/all --years 3
python -m ashare_quant.cli daily --data-root data/all
```

## 算法研究

```bash
python -m ashare_quant.cli benchmark --data-root data/all   # 算法表现对比（全市场）
python -m ashare_quant.cli decision --data-root data/all    # 训练模型并生成当日模拟持仓
```

## 控制台（仪表盘）

```bash
python -m streamlit run dashboard.py
```

外观由 `.streamlit/config.toml` 定义（2026-09-16 起）：浅灰蓝页面底色 + 纯白卡片、
主色深蓝 `#1d4ed8`、极简工具栏（隐藏 Deploy 等入口）、图表色板与
`dashboard.py::CHART_COLORS` 同步。**语义色按 A 股习惯：红=涨/正收益、绿=跌/负收益**
（与欧美默认相反）——它是靠 `dashboard.py` 自绘 KPI 卡片的 `up/down` 类实现的，
**不是**靠交换主题里的 `redColor/greenColor`（那样会把 `st.error` 也染成绿色）。

页签顺序：**总览 / 模拟盘 / 今日决策 / 账户 / 实时行情 / 算法对比 / 调整日志 / 数据状态**
（每个 tab 顶部都有状态药丸：数据截止 / 覆盖股票 / 落后股票 / 决策日期 / 每日自动更新 / 免责声明）。

- **总览**：状态卡片（股票数/数据截止/落后数/持仓）、一键操作（每日更新/强制重算/下载数据，后台运行实时输出）、今日持仓前 10；指标统一为等高 KPI 卡片，长段背景说明收进「口径说明」折叠区（文字未删）；
- **模拟盘 / 今日决策 / 算法对比**：净值曲线（含沪深300 与等权全市场基准）、预期收益、样本外夏普与算法净值。决策/持仓表不再显示 `model_scores` 原始 dict 列（那个列会撑爆宽度），改为 代码 / 排名得分 / 预期收益(20日) / 权重，模型明细在「模型预测明细」里按模型展开成列；
- **账户**：模拟账户净值曲线与绩效指标（总资产/总收益率/年化/夏普/最大回撤），
  自首个正式决策日（样本外）起逐日跟踪，支持 CSV 导出；
- **实时行情**：决策持仓的准实时快照（秒级延迟，腾讯/新浪源，自动刷新可选），
  并跟踪自决策日以来的涨跌；快照取不到时（非交易时段/限流）给**本地日线兜底表**并标注「非实时」；
- **调整日志**：原始 JSON dict 改为宽表「调整前 → 调整后（变化）」，另可查看全部 / 原始 JSON / 导出 CSV；
- **数据状态**：数据清单与失败清单，新增**数据完整率进度条**与**股票数据截止日分布图**（橙=与指数不同步）。

> ⚠️ **对比基准的口径 2026-09-16 修过一次**：账户页「等权全市场基准」原来按日取面板非空均值，
> 在"面板最后一行只有 209/5360 只有收盘价"的中间态下会算出 **-57% 的假暴跌**
> （当日均值 12.01 元 vs 前一日 28.27 元）。现在要求**当日有效成分股 ≥ 全市场 50%** 才纳入基准，
> 并在图上标注"已剔除 N 个覆盖不足的交易日"。**账户自身的净值/收益率/绩效指标一个都没动**，
> 只动了对比基准。回归测试：`tests/test_dashboard_smoke.py::test_equal_weight_bench_drops_sparse_dates`。
> 根因（面板在每日更新窗口内存在中间态）**尚未修复**，登记在 `AGENTS.md`「已知问题」。
>
> 复盘佐证（用事故当时的面板备份独立重算）：那 209 只在**前后两日都有值**，它们的均值实际只动
> **+0.77%** —— 也就是说 -57% 完全是"分母从 5349 只缩到 209 只"造成的假象，不是真跌。

> 实时行情为免费快照级（延迟数秒），非交易所级 tick；供盘中观察与持仓跟踪，
> 不改变月度调仓的决策逻辑。

## 扩展机制（便于后续开发维护）

系统提供三个可插拔注册表，新增算法/因子/数据源无需改动主流程：

- **模型**：`ashare_quant/ml/models.py` 用 `@register_model("name")` 注册工厂，
  第三方插件可通过 `[project.entry-points."ashare_quant.models"]` 自动发现；
- **因子**：`ashare_quant/research/factors.py` 用 `@register_factor("name")` 注册，
  签名 `(close, volume, **params)`，`compute_factors` 自动收集；
- **数据源**：`ashare_quant/fetchers/registry.py` 用 `register_source("name", module)`
  注册，`config.yaml` 里 `data_source` / `fallback_sources` 切换主备源
  （**主备顺序以 `config.yaml` 为准**；截至 2026-09-16 是主源 `akshare`（新浪）、
  备源 `tencent` → `mootdx`。腾讯直连实测并发 12 约 22 只/秒，是新浪源的 4 倍以上，
  但高频会被 WAF 限流；更新时主源失败自动切备源）。另可选 `mootdx`（通达信协议，
  1.06k★，单次 800 根 ~0.1s，自算前复权，与腾讯复权口径约有 1% 绝对价差、
  收益率影响可忽略）。

  > **实时行情已不再经 [easyquotation](https://github.com/shidenggui/easyquotation)**
  > （2026-09-18 修复，此前整块失效）：该库把两家接口硬编码成**明文 http**，
  > 两家都返回 400 空响应，而库又**静默返回空 dict** → 实时行情整块失效却看不出来。
  > 现在 `ashare_quant/realtime.py` **直连 https + 双源降级**（腾讯 → 新浪），
  > 全失败时显式抛 `RealtimeError`，绝不静默返回空。
  > 因此 `easyquotation` 已从依赖中移除（`pyproject.toml` / `requirements.txt` 均已不含）。

每日增量另有一条**腾讯批量报价端点**快路径（`ashare_quant/fetchers/tencent_quote.py`，
qt.gtimg.cn）：一次请求 120 只，全市场取数约 1.7~1.9s，收盘后当日 bar 立即就有
（新浪个股当日 bar 要滞后数小时）。它只在"每只仅缺最新一根 bar"时启用，其余情况
（缺多天 / 报价日期≠目标交易日 / 停牌）自动退回上面的逐只源链；**全量下载仍走主源 akshare**。
与 akshare 新浪 qfq 同日逐字段实测（150 只）：OHLC 相对误差 0、amount ≤3e-8、
volume 按手取整（绝对差中位 19 股 / 最大 50 股，均 ≤1 手）。

内置状态（注册表实测，2026-09-16）：模型 **16 个**（linear/rf/lgbm/histgb/svm/knn/mlp/
enet/pls/mlp_deep/xgb/rank_xgb/rank_lgb/huber_lgb/risk_aware_lgb/temporal_decay_lgb）、
因子 5 个（momentum/reversal/volatility/ma_deviation/volume_ratio）、数据源 **5 个**
（tencent/mootdx/akshare/baostock + 只做指数当日兜底的 index_snapshot；
后者只实现 `fetch_index_daily`，**不是个股源**）。注意**注册表里的模型数**与**每日决策实际用的模型池**
不是一回事：后者由 `config.yaml` 的 `models` 字段决定（当前 6 个）。

> 说明：账户净值曲线只统计正式决策（样本外）——首次生成时仅有当日一点，
> 随每日更新逐步累积；历史上曾尝试用全样本模型回填，属样本内（收益虚高），
> 故不作为账户曲线，历史策略表现请参考模拟盘/算法对比页。

## 成品使用（一条龙）

```powershell
.\scripts\run_all.ps1            # 数据 → 模拟盘 → 模型决策 → 仪表盘（首次全市场下载约20分钟）
.\scripts\start_dashboard.ps1    # 只启动仪表盘
python -m ashare_quant.cli daily --data-root data/all   # 每日：增量更新 + 报告 + 自动决策
```

## 性能优化（2026-08 重构）

全市场（5332 只 × 3 年日线，约 90MB 原始数据）此前全流程耗时约 3 分钟，经剖析确认
**瓶颈是代码开销而非硬件性能**（数据量很小，CPU/内存充足），已逐项向量化：

| 环节 | 优化前 | 优化后 | 手段 |
|---|---|---|---|
| 面板加载 build_panels | ~58s | ~10s（缓存命中 ~1.5s） | 每只股票只读一次 + 并行 IO + 数据未变时读合并缓存 |
| 横截面 IC（5 因子） | ~22s | ~5s | 逐日循环 → 一次排序同时产出 average/first 秩（自研 numpy 算子） |
| 分层收益（5 因子） | ~17s | ~5s | 逐日 qcut → 向量化分位边界公式 |
| 去极值 z-score（5 因子） | ~7s | ~3s | 逐行 apply → numpy nanquantile/nanmean/nanstd |
| 特征长表 build_dataset | ~25s | ~5.5s | 12 次 stack+concat → 一次 numpy reshape |
| daily 无新数据 | ~51s（含联网拉全市场列表） | ~2s | universe 本地缓存 + 指数日期相等时秒回 |
| daily 有新交易日 | ~23 分钟（1382.8s / 4989 只） | **~37 秒（5360 只，≈145 只/s）** | 腾讯批量报价端点一次多只（取数 1.7s）+ manifest 一次性写回 + 未覆盖目标日的源不再白跑 |
| daily --force 报告+决策 | ~180s | ~28s（含每月一次的重训） | 上述全部 + SVR 采样封顶 + thresholds 改按需（不再算校准集） |

数值一致性：向量化后的 IC/分层与旧逐日实现逐值对比，差异在机器精度
（~1e-16），研究报告结果完全可复现。

测试套件 **213 项**（`python -m pytest tests/` → `213 passed`，2026-09-18 实测）、
静态检查 `python -m ruff check .` 均须全绿再提交；项数以
`python -m pytest tests/ --collect-only -q` 的实测为准（会随功能增加而变化）。
已知偶发问题见 `AGENTS.md`「已知问题」一节。

**2026-09-18 实测的增量耗时**（与上表同机）：面板缓存命中时 `build_panels` **3.0s**（重建约 25s）、
一次 `--force`（含报告+决策，特征表命中）**47.0s**、补齐一个交易日的全市场（5352 只）**70.7s**。

## 稳定性与数据修复（2026-09-18）

这一轮把"数据错了但没人知道"的几类问题一起堵住（机制、判据与实测证据见 `AGENTS.md`）：

| 问题 | 现象 | 现状 |
|---|---|---|
| 实时行情整块失效 | 底层库把接口硬编码成 `http`，两家都返回 400 空响应，库又静默返回空 dict | `realtime.py` 直连 https + 双源降级；全失败显式抛 `RealtimeError` |
| 成交量量纲错 3 处 | 科创板 688/689 成交量被 ×100（K 线 / 报价 / 批量报价三条路径） | 统一走 `venues.volume_in_shares`；历史数据已用独立源对撞修回 |
| 面板横截面塌缩 | 指数到了 D、96% 个股停在 D-1，决策照样出 | `pipeline.panel_coverage` 实测覆盖率 → 塌缩面板不落盘、不出决策 |
| 当日指数取不到 | 唯一能当日给指数的端点被 WAF 拦（HTTP 501）→ 一整天不前进 | 指数链尾加"报价端点"兜底源；源全空时 `daily` 退出码 **2** |
| 缓存永不命中 | 每次运行都重写指数文件，把缓存判据自己刷失效 | 内容相同不写盘：`build_panels` 25s → **3.0s** |
| 标签泄漏 | 训练标签跨进校准期（实测 3.37% 训练行） | 训练/校准之间留 20 日 embargo |

```powershell
# 数据体检（换数据源、或怀疑行情不对时先跑这三条）
python scripts/fix_volume_unit.py            # 量纲体检：应始终报"需要修复 0 只"
python scripts/audit_volume_integrity.py     # 逐 (股票,日期) 对撞独立源（--apply 可修）
python scripts/audit_qfq_anchor.py           # 换主源后审计前复权锚点（--fix 整段重建）
```

⚠️ **16:05 计划任务的口径**：当日 bar 是否已发布由数据源决定；拿不到时 `daily` 会拒绝推进
并返回退出码 **2**（`Get-ScheduledTaskInfo` 的 `LastTaskResult` 随之非 0），
**不会**在旧数据上出决策 —— 看到非 0 就照 `logs\daily_scheduled.utf8.log` 末尾的 ‼️ 处理。

### 说明

- 决策模型每月重训一次（LGBM/HistGB 6 万样本，SVR 封顶 2 万样本）。
  ⚠️ **校准集抽样已不再是默认行为**：`train_and_save(compute_thresholds=False)` 是默认值
  （2026-09-18 起），`thresholds` 字段保留但默认 `{}` —— 全代码 0 处读取它，
  此前每轮为 6 个模型各跑一遍最多 3 万行预测是**纯浪费**（见 `docs/HONESTY.md` 问题⑥）。
  单次重训约 25s；SVR 全量 6 万样本训练需 ~3 分钟，故按模型差异化采样；
- benchmark 全量算法对比（7 个 ML 模型 × 4 折 walk-forward）因含逐折 SVR/MLP 训练，
  属于算法本身的计算成本，约 5-8 分钟，与日常数据流水线无关。
