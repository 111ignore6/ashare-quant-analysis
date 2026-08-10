# A股量化研究·模拟分析系统

基于真实 A 股数据（AKShare 主 / BaoStock 备）的历史数据研究项目。

> 本项目为模拟研究，不构成任何投资建议。

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

首次使用：下载全市场数据约 8-15 分钟（默认腾讯源，可断点续传）；
之后每日增量更新通常 1-3 秒。

计划任务仍可选用 `.\scripts\schedule_daily.ps1`（周一至五 16:05 自动运行）。

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

- **总览**：状态卡片（股票数/数据截止/落后数/持仓）、一键操作（每日更新/强制重算/下载数据，后台运行实时输出）、今日持仓前 10；
- **实时行情**：决策持仓的准实时快照（秒级延迟，腾讯/新浪源，自动刷新可选），
  并跟踪自决策日以来的涨跌；
- **模拟盘 / 今日决策 / 算法对比 / 调整日志 / 数据状态**：净值曲线（含沪深300 与等权全市场真实基准）、预期收益、样本外夏普与算法净值、权重调整日志、数据清单与失败清单。

> 实时行情为免费快照级（延迟数秒），非交易所级 tick；供盘中观察与持仓跟踪，
> 不改变月度调仓的决策逻辑。

## 扩展机制（便于后续开发维护）

系统提供三个可插拔注册表，新增算法/因子/数据源无需改动主流程：

- **模型**：`ashare_quant/ml/models.py` 用 `@register_model("name")` 注册工厂，
  第三方插件可通过 `[project.entry-points."ashare_quant.models"]` 自动发现；
- **因子**：`ashare_quant/research/factors.py` 用 `@register_factor("name")` 注册，
  签名 `(close, volume, **params)`，`compute_factors` 自动收集；
- **数据源**：`ashare_quant/fetchers/registry.py` 用 `register_source("name", module)`
  注册，`config.yaml` 里 `data_source` / `fallback_source` 切换主备源
  （默认 `tencent`：直连腾讯行情，实测并发 12 约 22 只/秒，是新浪源的 4 倍以上；
  更新时主源失败自动切备源）。另可选 `mootdx`（通达信协议，1.06k★，单次 800 根
  ~0.1s，自算前复权，与腾讯复权口径约有 1% 绝对价差、收益率影响可忽略）。
  实时行情用 [easyquotation](https://github.com/shidenggui/easyquotation)
  （5.3k★）腾讯/新浪快照。

内置状态：模型 7 个（linear/rf/lgbm/histgb/svm/knn/mlp）、因子 5 个
（momentum/reversal/volatility/ma_deviation/volume_ratio）、数据源 4 个
（tencent/mootdx/akshare/baostock）。

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
| daily --force 报告+决策 | ~180s | ~28s（含每月一次的重训） | 上述全部 + SVR 采样封顶/校准集抽样 |

数值一致性：向量化后的 IC/分层与旧逐日实现逐值对比，差异在机器精度
（~1e-16），研究报告结果完全可复现。测试套件 70 项全绿。

### 说明

- 决策模型每月重训一次（LGBM/HistGB 6 万样本，SVR 封顶 2 万样本、校准集抽样 3 万），
  单次重训约 25s；SVR 全量 6 万样本训练需 ~3 分钟，故按模型差异化采样；
- benchmark 全量算法对比（7 个 ML 模型 × 4 折 walk-forward）因含逐折 SVR/MLP 训练，
  属于算法本身的计算成本，约 5-8 分钟，与日常数据流水线无关。
