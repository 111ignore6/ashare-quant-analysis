# 仪表盘增强（自定义资金 + 参考主流量化平台功能）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 A 股量化模拟控制台补齐自定义资金入口，并按 qstock/聚宽/掘金/vn.py 等主流项目的常见功能，增强绩效指标、交易台账、K线详情、大盘速览与数据更新耗时统计。

**Architecture:** 逻辑能力下沉到 `ashare_quant/`（config 写回、绩效扩展、账户重算、交易台账、指数快照、更新统计），仪表盘 `dashboard.py` 只做展示与交互；全部数据复用本地缓存，不引入新依赖。

**Tech Stack:** Python 3.13 / Streamlit 1.61 / Plotly / pandas / easyquotation / pytest

---

## 文件结构

- 修改 `ashare_quant/config.py`：新增 `Config.to_dict()` 与 `update_config_yaml()`（合并写回）。
- 修改 `ashare_quant/backtest/metrics.py`：新增 `profit_loss_ratio`、`calmar`。
- 修改 `ashare_quant/portfolio.py`：新增 `monthly_returns_table()`、`recompute_account()`、`build_trade_ledger()`。
- 修改 `ashare_quant/realtime.py`：新增 `index_snapshot()`（四大指数实时快照）。
- 修改 `ashare_quant/cli.py`：`cmd_daily` 记录分阶段耗时，写 `update_stats.json`。
- 修改 `dashboard.py`：侧边栏模拟参数面板；账户页指标/热力图/对比卡/台账；决策页 K线+调仓对比；实时页大盘速览；数据状态页耗时统计。
- 测试：`tests/test_config.py`、`tests/test_metrics.py`、`tests/test_portfolio.py`、`tests/test_realtime.py`。

### Task 1: 配置写回

**Files:** `ashare_quant/config.py`、`tests/test_config.py`

- [ ] 在 `Config` 增加 `to_dict()`：`data_root` 转 str，其余字段原样。
- [ ] 新增 `update_config_yaml(path, **kwargs)`：读现有 yaml → 按 dataclass 字段做类型强转（float/int/bool/str/None）→ 合并写回。
- [ ] 测试：写临时 yaml → 更新 initial_capital/top_n/stop_loss=None → 断言其他字段保留、类型正确。

### Task 2: 绩效指标扩展

**Files:** `ashare_quant/backtest/metrics.py`、`tests/test_metrics.py`

- [ ] `metrics_from_returns` 新增 `profit_loss_ratio`（平均盈利/|平均亏损|，无亏损时 NaN）与 `calmar`（年化/|最大回撤|，无回撤时 NaN）。
- [ ] 测试：含正负收益的序列 → plr 数值正确；纯正收益 → plr/calmar 为 NaN。

### Task 3: 账户模块新增能力

**Files:** `ashare_quant/portfolio.py`、`tests/test_portfolio.py`

- [ ] `monthly_returns_table(returns)`：返回 年×月 复合收益 DataFrame（索引=年，列=1..12）。
- [ ] `recompute_account(history, close, capital)`：复用 `equity_curve`，返回 `(equity_series, metrics)`。
- [ ] `build_trade_ledger(history, close, capital)`：每次决策视为等权全换仓；输出 买入/卖出/持有 动作、数量、价格、金额、实现盈亏。
- [ ] 测试：构造两段决策历史，断言台账动作集合与实现盈亏。

### Task 4: 指数实时快照

**Files:** `ashare_quant/realtime.py`、`tests/test_realtime.py`

- [ ] `index_snapshot(source="tencent")`：拉取 上证/深成/创业板/沪深300，返回 代码/名称/现价/涨跌幅。
- [ ] 测试：monkeypatch easyquotation 返回四指数 → 断言列与顺序。

### Task 5: 每日更新耗时统计

**Files:** `ashare_quant/cli.py`

- [ ] `cmd_daily` 记录 phase1/2/3 耗时与更新数量，写入 `cfg.data_root/"update_stats.json"`。

### Task 6: 仪表盘接线

**Files:** `dashboard.py`

- [ ] 侧边栏：初始资金、选股数量、止损/止盈开关与阈值、数据源；「保存参数到配置」写回 yaml。
- [ ] 账户页：用侧边栏资金重算净值与指标；指标行补 胜率/盈亏比/年化波动/Calmar；月度热力图；账户 vs 基准对比卡；交易台账表 + CSV 下载。
- [ ] 决策页：持仓下拉 → 本地 K线（Candlestick + MA5/20/60 + 成交量）；与上一决策日 新增/卖出/持有 对比表。
- [ ] 实时页：顶部四大指数卡片。
- [ ] 数据状态页：显示 `update_stats.json` 与可用数据源列表。

### Task 7: 验证

- [ ] `pytest tests/ -q` 全绿（含既有 91 项）。
- [ ] `streamlit.testing.v1.AppTest` 冒烟：monkeypatch 行情后运行 `dashboard.py` 无异常。
- [ ] 提交。

---
