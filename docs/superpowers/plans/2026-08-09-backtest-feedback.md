# 模拟盘与反馈调整（M4）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现带 A 股真实约束（T+1、涨跌停、停牌、交易成本、次日开盘价成交）的模拟回测引擎，叠加"策略内调参 + 策略间轮动 + 调整日志"反馈闭环，并输出模拟盘对比报告。

**Architecture:** `ashare_quant/backtest/engine.py` 实现完整撮合；`ashare_quant/feedback/` 实现滚动绩效轮动与调整日志；CLI 增加 `simulate` 子命令，复用 M3 的模型与筛选结果（保留模型 + 对照组都跑，反馈层决定权重）。

**Tech Stack:** 沿用现有依赖，无新增。

---

## Task 1: 完整回测引擎

**Files:**
- Create: `ashare_quant/backtest/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_engine.py
import numpy as np
import pandas as pd
from ashare_quant.backtest.engine import run_backtest
from ashare_quant.backtest.simple import monthly_rebalance_dates


def _market(n_days=130, n_stocks=20, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    drift = np.linspace(0.0005, 0.002, n_stocks)
    rets = rng.normal(0, 0.01, (n_days, n_stocks)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(n_stocks)])
    open_ = close.shift(1) * (1 + rng.normal(0, 0.002, close.shape))
    open_.iloc[0] = close.iloc[0]
    return close, open_


def test_engine_returns_turnover_and_holdings():
    close, open_ = _market()
    score = close.pct_change(20, fill_method=None)
    dates = monthly_rebalance_dates(close.index)
    res = run_backtest(score, close, open_, dates, top_n=5)
    assert len(res.returns) > 2
    assert (res.turnover >= 0).all()
    assert len(res.holdings) == len(res.returns)


def test_engine_skips_limit_up_buy():
    close, open_ = _market(n_days=80, n_stocks=10, seed=3)
    # 让 S0000 在第一个成交日一字涨停：open 相对前收 +10%
    first_exec = close.index[1]
    prev = close.shift(1).loc[first_exec, "S0000"]
    open_.loc[first_exec, "S0000"] = prev * 1.10
    score = close.pct_change(20, fill_method=None)
    dates = monthly_rebalance_dates(close.index)
    res = run_backtest(score, close, open_, dates, top_n=3, limit=0.098)
    first_holdings = next(iter(res.holdings.values()))
    assert "S0000" not in first_holdings


def test_engine_costs_reduce_returns():
    close, open_ = _market(seed=5)
    score = close.pct_change(20, fill_method=None)
    dates = monthly_rebalance_dates(close.index)
    res_no = run_backtest(score, close, open_, dates, top_n=5, commission=0.0, stamp=0.0, slippage=0.0)
    res_yes = run_backtest(score, close, open_, dates, top_n=5)
    assert res_yes.returns.mean() < res_no.returns.mean()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_engine.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/backtest/engine.py
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    turnover: pd.Series
    holdings: dict = field(default_factory=dict)


def _next_trading_day(dates, d):
    pos = dates.get_loc(d)
    if pos + 1 >= len(dates):
        return None
    return dates[pos + 1]


def run_backtest(score: pd.DataFrame, close: pd.DataFrame, open_: pd.DataFrame,
                 rebalance_dates, top_n: int = 50, commission: float = 0.00025,
                 stamp: float = 0.0005, slippage: float = 0.001,
                 limit: float = 0.098) -> BacktestResult:
    """月度调仓：信号日选股，次日开盘成交；持仓数量记账，市值法算收益；T+1 由持有期隐含满足。"""
    signal_dates = [d for d in rebalance_dates if d in score.index]
    rets, turnovers, holdings = {}, {}, {}
    shares: dict[str, float] = {}
    prev_holdings: set[str] = set()
    prev_value: float | None = None
    for i in range(len(signal_dates) - 1):
        d, d2 = signal_dates[i], signal_dates[i + 1]
        exec_day = _next_trading_day(close.index, d)
        next_exec = _next_trading_day(close.index, d2)
        if exec_day is None or next_exec is None:
            continue
        picks = score.loc[d].dropna().nlargest(top_n).index.tolist()
        prev_close = close.shift(1).loc[exec_day]
        def px(symbol):
            p = open_.loc[exec_day, symbol]
            if pd.isna(p):
                p = prev_close[symbol]
            return p

        # 1) 期初市值：现金 + 持仓按开盘（停牌用前收）计值；可卖的先卖出变现
        cash = 0.0
        pos_value = 0.0
        for s in list(shares):
            price = px(s)
            if pd.isna(price):
                continue
            sellable = price > prev_close[s] * (1 - limit)
            if sellable:
                cash += shares[s] * price * (1 - slippage) * (1 - commission - stamp)
                del shares[s]
            else:
                pos_value += shares[s] * price
        value_now = cash + pos_value
        # 2) 记录期间收益
        if prev_value is not None and value_now > 0:
            rets[d] = float(value_now / prev_value - 1)
        # 3) 买入：现金等权分配给可买的新目标（非涨停、非停牌、未持有）
        buyable = [s for s in picks if s not in shares and not pd.isna(px(s))
                   and px(s) < prev_close[s] * (1 + limit)]
        if buyable and cash > 0:
            budget = cash / len(buyable)
            for s in buyable:
                buy_px = px(s) * (1 + slippage) * (1 + commission)
                if buy_px > 0:
                    shares[s] = budget / buy_px
                    cash -= budget
        # 4) 记录换手与持仓，更新期初基准值（交易后市值）
        held = set(shares)
        if prev_value is not None:
            turnovers[d] = len(prev_holdings.symmetric_difference(held)) / max(1, len(prev_holdings | held))
        prev_holdings = held
        prev_value = cash + sum(shares[s] * px(s) for s in shares if not pd.isna(px(s)))
        holdings[d] = sorted(held)
    r = pd.Series(rets)
    return BacktestResult(returns=r, equity=(1 + r).cumprod(), turnover=pd.Series(turnovers), holdings=holdings)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_engine.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/backtest/engine.py tests/test_engine.py
git commit -m "feat: 完整回测引擎（T+1/涨跌停/停牌/成本）"
```

---

## Task 2: 反馈调整（滚动轮动 + 调整日志）

**Files:**
- Create: `ashare_quant/feedback/__init__.py`
- Create: `ashare_quant/feedback/rotation.py`
- Create: `ashare_quant/feedback/log.py`
- Test: `tests/test_feedback.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_feedback.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ashare_quant.feedback.log import AdjustmentLog
from ashare_quant.feedback.rotation import rotate_weights


def _returns(n_models=3, n_periods=24, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-31", periods=n_periods, freq="ME")
    cols = {f"m{i}": rng.normal(0.008, 0.03, n_periods) for i in range(n_models)}
    return pd.DataFrame(cols, index=idx)


def test_rotation_weights_normalize_and_threshold():
    r = _returns()
    w = rotate_weights(r, window=6, min_weight=0.05, change_threshold=0.15)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert (w >= 0).all().all()


def test_rotation_no_churn_when_flat():
    r = pd.DataFrame({"m0": [0.01] * 24, "m1": [0.01] * 24})
    w = rotate_weights(r, window=6, change_threshold=0.5)
    assert (w.diff().abs().sum(axis=1) < 0.5).all()


def test_adjustment_log_roundtrip(tmp_path):
    log = AdjustmentLog(tmp_path / "adjust.jsonl")
    log.append(date="2026-08-07", trigger="rotation", action="weights", before={"m0": 0.5}, after={"m1": 0.5}, effect="expected")
    entries = log.read()
    assert len(entries) == 1
    assert entries[0]["trigger"] == "rotation"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_feedback.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/feedback/__init__.py
"""反馈监控与自适应调整（阶段4）。"""
```

```python
# ashare_quant/feedback/rotation.py
from __future__ import annotations

import numpy as np
import pandas as pd


def rotate_weights(model_returns: pd.DataFrame, window: int = 6, min_weight: float = 0.05,
                   change_threshold: float = 0.15) -> pd.DataFrame:
    """按滚动夏普分配权重：夏普<=0 视为 0；权重变化小于阈值时保持上次权重（防频繁切换）。"""
    rolling = model_returns.rolling(window).apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else 0.0, raw=True)
    pos = rolling.clip(lower=0)
    weights = pos.div(pos.sum(axis=1), axis=0).fillna(1 / len(model_returns.columns))
    weights = weights.clip(lower=min_weight)
    weights = weights.div(weights.sum(axis=1), axis=0)
    prev = weights.iloc[0]
    out = []
    for _, row in weights.iterrows():
        if (row - prev).abs().sum() < change_threshold:
            row = prev
        out.append(row)
        prev = row
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)
```

```python
# ashare_quant/feedback/log.py
from __future__ import annotations

import json
from pathlib import Path


class AdjustmentLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, **entry) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_feedback.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/feedback tests/test_feedback.py
git commit -m "feat: 反馈调整——滚动轮动与调整日志"
```

---

## Task 3: CLI simulate 与模拟盘对比报告

**Files:**
- Create: `ashare_quant/simulation.py`
- Modify: `ashare_quant/cli.py`
- Modify: `ashare_quant/research/report.py`
- Test: `tests/test_simulation.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_simulation.py
import numpy as np
import pandas as pd
from ashare_quant.models.candidates import MomentumModel, ReversalModel
from ashare_quant.simulation import run_simulation


def _market():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    drift = np.linspace(0.0003, 0.0012, 20)
    rets = rng.normal(0, 0.01, (300, 20)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(20)])
    open_ = close.shift(1).fillna(close)
    volume = pd.DataFrame(1000, index=idx, columns=close.columns)
    return close, open_, volume


def test_run_simulation_returns_models_and_rotation():
    close, open_, volume = _market()
    models = {"momentum": MomentumModel(20), "reversal": ReversalModel(20)}
    out = run_simulation(models, close, open_, volume, top_n=5)
    assert set(out["model_returns"].columns) == {"momentum", "reversal"}
    assert {"model", "sharpe", "annual_return", "max_drawdown"} <= set(out["summary"].columns)
    assert out["weights"] is not None
    assert len(out["log"].read()) >= 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_simulation.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/simulation.py
from __future__ import annotations

import pandas as pd

from .backtest.engine import run_backtest
from .backtest.metrics import metrics_from_returns
from .backtest.simple import monthly_rebalance_dates
from .feedback.log import AdjustmentLog
from .feedback.rotation import rotate_weights


def run_simulation(models: dict[str, object], close: pd.DataFrame, open_: pd.DataFrame,
                   volume: pd.DataFrame, top_n: int = 50,
                   log_path=None, window: int = 6, threshold: float = 0.15) -> dict:
    dates = monthly_rebalance_dates(close.index)
    model_returns = {}
    holdings = {}
    for name, model in models.items():
        score = model.score(close, volume)
        res = run_backtest(score, close, open_, dates, top_n=top_n)
        model_returns[name] = res.returns
        holdings[name] = res.holdings
    rets = pd.DataFrame(model_returns)
    weights = rotate_weights(rets, window=window, change_threshold=threshold)
    rot = (rets * weights.shift(1)).sum(axis=1, min_count=1).dropna()
    summary_rows = {}
    for name in rets.columns:
        summary_rows[name] = metrics_from_returns(rets[name], periods_per_year=12)
    summary_rows["rotation"] = metrics_from_returns(rot, periods_per_year=12)
    summary = pd.DataFrame(summary_rows).T.reset_index().rename(columns={"index": "model"})
    log = AdjustmentLog(log_path) if log_path else None
    if log is not None:
        first = weights.index[0]
        log.append(date=str(first.date()), trigger="rotation", action="weights",
                   before=dict(weights.iloc[0]), after=dict(weights.iloc[0]), effect="初始化")
        for i in range(1, len(weights)):
            before = dict(weights.iloc[i - 1])
            after = dict(weights.iloc[i])
            if before != after:
                log.append(date=str(weights.index[i].date()), trigger="rotation", action="weights",
                           before=before, after=after, effect="滚动夏普轮动")
    return {"model_returns": rets, "rotation_returns": rot, "weights": weights,
            "summary": summary, "holdings": holdings, "log": log}
```

在 `ashare_quant/research/report.py` 追加模拟盘报告：

```python
def simulation_to_markdown(summary: pd.DataFrame, log_entries: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 模拟盘对比报告（M4）", "",
             "> 模拟研究，仅用于数据分析与学习，不构成投资建议。", "",
             "## 绩效对比", "",
             "| 模型 | 年化收益 | 年化波动 | 夏普 | 最大回撤 | 胜率 |",
             "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['model']} | {r['annual_return']:.2%} | {r['annual_vol']:.2%} | "
                     f"{r['sharpe']:.2f} | {r['max_drawdown']:.2%} | {r['win_rate']:.2%} |")
    lines += ["", "## 调整日志", ""]
    for e in log_entries[-10:]:
        lines.append(f"- {e['date']} [{e['trigger']}] {e['action']}: {e['before']} -> {e['after']}（{e['effect']}）")
    path.write_text("\n".join(lines), encoding="utf-8")
```

在 `ashare_quant/cli.py` 追加：

```python
def cmd_simulate(args) -> None:
    import json

    from .models.candidates import LowVolModel, MomentumModel, MultiFactorModel, ReversalModel
    from .pipeline import build_panels
    from .research.report import simulation_to_markdown
    from .simulation import run_simulation

    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    panels = build_panels(store)
    close, volume = panels["close"], panels["volume"]
    open_ = pd.DataFrame({s: store.load(s)["open"] for s in close.columns}).sort_index()
    models = {"momentum": MomentumModel(60), "reversal": ReversalModel(60),
              "lowvol": LowVolModel(60), "multifactor": MultiFactorModel(
                  {"volume_ratio": 0.4, "ma_deviation": 0.2, "reversal60": 0.2, "lowvol": 0.2})}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "adjustments.jsonl"
    sim = run_simulation(models, close, open_, volume, top_n=cfg.top_n, log_path=log_path)
    simulation_to_markdown(sim["summary"], sim["log"].read(), out_dir / "simulation.md")
    (out_dir / "simulation.json").write_text(
        json.dumps({"summary": sim["summary"].to_dict(orient="records"),
                    "rotation_sharpe": float(metrics_from_returns(sim["rotation_returns"])["sharpe"])},
                   ensure_ascii=False, default=str), encoding="utf-8")
    sim["model_returns"].to_csv(out_dir / "model_returns.csv", encoding="utf-8-sig")
    print(sim["summary"].to_string(index=False))
    print(f"模拟盘报告已生成: {out_dir}")
```

`main()` 注册：

```python
    sm = sub.add_parser("simulate", help="运行模拟盘与反馈调整")
    sm.add_argument("--config", default="config.yaml")
    sm.add_argument("--data-root")
    sm.add_argument("--out-dir", default="docs/simulation")
    sm.set_defaults(func=cmd_simulate)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/ -q`
Expected: PASS（全部通过）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/simulation.py ashare_quant/cli.py ashare_quant/research/report.py tests/test_simulation.py
git commit -m "feat: 模拟盘运行与反馈调整闭环"
```

---

## Task 4: 端到端运行与计划自检

- [ ] **Step 1: 全量单元测试**

Run: `python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 2: 端到端模拟（真实数据 data/3y）**

Run: `python -m ashare_quant.cli simulate --data-root data/3y --out-dir docs/simulation`
Expected: 生成 `simulation.md`、`simulation.json`、`model_returns.csv`、`adjustments.jsonl`；输出对比表与轮动权重。

- [ ] **Step 3: 提交结果**

```bash
git add docs/simulation
git commit -m "docs: 端到端模拟盘与反馈调整结果"
```
