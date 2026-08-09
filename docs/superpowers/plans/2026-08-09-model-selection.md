# 候选模型构建与滚动筛选（M3）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 依据《数据研究报告》（60 日反转最强、有效因子 volume_ratio/ma_deviation/reversal/volatility、因子冗余、市场状态差异）实现 5 个候选模型，用滚动样本外（walk-forward）筛选出 2~4 个模型并记录淘汰理由。

**Architecture:** 在 `ashare_quant/models/` 放模型（统一 `score(close, volume) -> 日期×股票 评分面板`），`ashare_quant/backtest/` 放绩效指标与简化选股回测（M4 复用），`ashare_quant/screening.py` 做训练/验证划分、网格搜索、筛选决策；CLI 增加 `select` 子命令，输出模型筛选报告。

**Tech Stack:** 沿用现有 pandas/numpy/scipy；无新依赖。

---

## Task 1: 模型基类与四个候选模型

**Files:**
- Create: `ashare_quant/models/__init__.py`
- Create: `ashare_quant/models/base.py`
- Create: `ashare_quant/models/candidates.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_models.py
import numpy as np
import pandas as pd
from ashare_quant.models.candidates import LowVolModel, MomentumModel, MultiFactorModel, ReversalModel


def _panel(n_days=150, n_stocks=30, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    drift = np.linspace(-0.0005, 0.001, n_stocks)
    rets = rng.normal(0, 0.02, (n_days, n_stocks)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(n_stocks)])
    volume = pd.DataFrame(1000 + rng.normal(0, 100, close.shape), index=idx, columns=close.columns)
    return close, volume


def test_models_return_score_panels():
    close, volume = _panel()
    for model in [ReversalModel(60), LowVolModel(20), MomentumModel(20), MultiFactorModel()]:
        score = model.score(close, volume)
        assert isinstance(score, pd.DataFrame)
        assert score.shape == close.shape
        assert score.columns.tolist() == close.columns.tolist()


def test_reversal_scores_oversold_higher():
    close, volume = _panel()
    model = ReversalModel(20)
    score = model.score(close, volume)
    last = score.iloc[-1].dropna()
    worst = close.iloc[-1].pct_change(20).dropna().idxmin()
    assert last.idxmax() == worst
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_models.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/models/__init__.py
"""候选模型（阶段2）。"""
```

```python
# ashare_quant/models/base.py
from __future__ import annotations

import pandas as pd


class Model:
    name: str = "base"

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
```

```python
# ashare_quant/models/candidates.py
from __future__ import annotations

import pandas as pd

from ..research.factors import compute_factors, winsorize_zscore
from .base import Model


class ReversalModel(Model):
    """均值回归：过去 h 日跌得越多的股票得分越高。"""
    name = "reversal"

    def __init__(self, horizon: int = 60) -> None:
        self.horizon = horizon

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        return -close.pct_change(self.horizon, fill_method=None)


class LowVolModel(Model):
    """低波动防守：滚动波动率越低得分越高。"""
    name = "lowvol"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        vol = close.pct_change(fill_method=None).rolling(self.window).std()
        return -vol


class MomentumModel(Model):
    """动量趋势（对照组）：过去 h 日涨幅越高得分越高。"""
    name = "momentum"

    def __init__(self, horizon: int = 20) -> None:
        self.horizon = horizon

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        return close.pct_change(self.horizon, fill_method=None)


class MultiFactorModel(Model):
    """横截面多因子：有效因子 z-score 加权求和。"""
    name = "multifactor"

    FACTORS = ("volume_ratio", "ma_deviation", "reversal60", "lowvol")

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {f: 1 / len(self.FACTORS) for f in self.FACTORS}

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        base = compute_factors(close, volume)
        custom = {
            "volume_ratio": base["volume_ratio"],
            "ma_deviation": base["ma_deviation"],
            "reversal60": -close.pct_change(60, fill_method=None),
            "lowvol": -close.pct_change(fill_method=None).rolling(20).std(),
        }
        total = None
        for name, w in self.weights.items():
            z = winsorize_zscore(custom[name])
            total = w * z if total is None else total + w * z
        return total
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_models.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/models tests/test_models.py
git commit -m "feat: 候选模型（反转/低波/动量/多因子）"
```

---

## Task 2: 绩效指标模块

**Files:**
- Create: `ashare_quant/backtest/__init__.py`
- Create: `ashare_quant/backtest/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_metrics.py
import numpy as np
import pandas as pd
from ashare_quant.backtest.metrics import drawdown_series, metrics_from_returns


def _returns():
    rng = np.random.default_rng(4)
    return pd.Series(rng.normal(0.005, 0.03, 36), index=pd.date_range("2023-01-31", periods=36, freq="ME"))


def test_metrics_keys_and_drawdown():
    r = _returns()
    m = metrics_from_returns(r, periods_per_year=12)
    assert {"annual_return", "annual_vol", "sharpe", "max_drawdown", "win_rate"} <= set(m)
    dd = drawdown_series(r)
    assert (dd <= 0).all()
    assert abs(m["max_drawdown"] - dd.min()) < 1e-9


def test_metrics_of_constant_gain():
    r = pd.Series([0.01] * 12)
    m = metrics_from_returns(r, periods_per_year=12)
    assert m["sharpe"] > 0
    assert m["max_drawdown"] == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/backtest/__init__.py
"""回测引擎与绩效指标。"""
```

```python
# ashare_quant/backtest/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns.fillna(0)).cumprod()
    peak = equity.cummax()
    return equity / peak - 1


def metrics_from_returns(returns: pd.Series, periods_per_year: float = 12) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0}
    ann_return = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    return {
        "annual_return": float(ann_return),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown_series(r).min()),
        "win_rate": float((r > 0).mean()),
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/backtest tests/test_metrics.py
git commit -m "feat: 绩效指标与回撤计算"
```

---

## Task 3: 简化选股回测（筛选用）

**Files:**
- Create: `ashare_quant/backtest/simple.py`
- Test: `tests/test_simple_backtest.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_simple_backtest.py
import numpy as np
import pandas as pd
from ashare_quant.backtest.simple import monthly_rebalance_dates, simple_topn_returns


def _panel():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2023-01-02", periods=130, freq="B")
    drift = np.linspace(0.0005, 0.002, 20)
    rets = rng.normal(0, 0.01, (130, 20)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(20)])
    return close


def test_rebalance_dates_are_month_starts():
    close = _panel()
    dates = monthly_rebalance_dates(close.index)
    assert len(dates) > 3
    assert all(d.month != dates[i - 1].month for i, d in enumerate(dates[1:], start=1))


def test_topn_returns_positive_for_momentum():
    close = _panel()
    score = close.pct_change(20, fill_method=None)
    dates = monthly_rebalance_dates(close.index)
    rets = simple_topn_returns(score, close, dates, top_n=5)
    assert rets.mean() > 0.01
    assert set(rets.index).issubset(set(dates))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_simple_backtest.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/backtest/simple.py
from __future__ import annotations

import pandas as pd


def monthly_rebalance_dates(dates) -> pd.DatetimeIndex:
    """每月第一个交易日作为调仓日。"""
    s = pd.Series(dates)
    first = s.groupby(s.dt.to_period("M")).min()
    return pd.DatetimeIndex(first.values).sort_values()


def simple_topn_returns(score: pd.DataFrame, close: pd.DataFrame,
                        rebalance_dates, top_n: int = 50) -> pd.Series:
    """每个调仓日选评分 Top N 等权，持有到下一个调仓日，返回月度组合收益。"""
    dates = [d for d in rebalance_dates if d in close.index]
    out = {}
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        s = score.loc[d].dropna()
        if len(s) < top_n:
            continue
        picks = s.nlargest(top_n).index
        ret = close.loc[nxt, picks] / close.loc[d, picks] - 1
        out[d] = float(ret.mean())
    return pd.Series(out)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_simple_backtest.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/backtest/simple.py tests/test_simple_backtest.py
git commit -m "feat: 简化选股回测（月度调仓 TopN）"
```

---

## Task 4: 训练/验证划分、网格搜索与筛选决策

**Files:**
- Create: `ashare_quant/screening.py`
- Test: `tests/test_screening.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_screening.py
import numpy as np
import pandas as pd
from ashare_quant.models.candidates import MomentumModel, ReversalModel
from ashare_quant.screening import evaluate, grid_search, run_screening, split_dates


def _panel():
    rng = np.random.default_rng(8)
    idx = pd.date_range("2022-01-03", periods=500, freq="B")
    drift = np.linspace(0.0002, 0.0015, 30)
    rets = rng.normal(0, 0.01, (500, 30)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(30)])
    volume = pd.DataFrame(1000, index=idx, columns=close.columns)
    return close, volume


def test_split_dates():
    close, _ = _panel()
    tr, va = split_dates(close.index, train_frac=0.6)
    assert len(tr) > len(va)
    assert tr[-1] < va[0]


def test_grid_search_picks_best_train_sharpe():
    close, volume = _panel()
    tr, _ = split_dates(close.index)
    best = grid_search(ReversalModel, {"horizon": [20, 60]}, close, volume, tr)
    assert "horizon" in best and "sharpe" in best


def test_run_screening_returns_decisions():
    close, volume = _panel()
    bench = close.mean(axis=1)
    out = run_screening(close, volume, bench, top_n=10)
    assert {"model", "keep", "reason"} <= set(out.columns)
    assert "benchmark" in out["model"].tolist()


def test_evaluate_metrics():
    close, volume = _panel()
    tr, va = split_dates(close.index)
    m = evaluate(MomentumModel(20), close, volume, va, top_n=10)
    assert "sharpe" in m and "max_drawdown" in m
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_screening.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/screening.py
from __future__ import annotations

import pandas as pd

from .backtest.metrics import metrics_from_returns
from .backtest.simple import monthly_rebalance_dates, simple_topn_returns
from .models.candidates import LowVolModel, MomentumModel, MultiFactorModel, ReversalModel


def split_dates(dates, train_frac: float = 0.67):
    dates = pd.DatetimeIndex(sorted(dates))
    cut = int(len(dates) * train_frac)
    return dates[:cut], dates[cut:]


def evaluate(model, close: pd.DataFrame, volume: pd.DataFrame,
             dates, top_n: int = 50) -> dict:
    score = model.score(close, volume)
    rdates = monthly_rebalance_dates(dates)
    rets = simple_topn_returns(score, close, rdates, top_n=top_n)
    m = metrics_from_returns(rets, periods_per_year=12)
    m["model"] = model.name
    return m


def grid_search(model_cls, param_grid: dict, close: pd.DataFrame, volume: pd.DataFrame,
                dates, top_n: int = 50) -> dict:
    keys = list(param_grid)
    import itertools
    best = None
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(keys, combo))
        model = model_cls(**params)
        m = evaluate(model, close, volume, dates, top_n=top_n)
        if best is None or m["sharpe"] > best["sharpe"]:
            best = {**params, "sharpe": m["sharpe"]}
    return best or {}


CANDIDATES = [
    ("reversal", ReversalModel, {"horizon": [20, 60, 120]}),
    ("lowvol", LowVolModel, {"window": [20, 60]}),
    ("momentum", MomentumModel, {"horizon": [10, 20, 60]}),
    ("multifactor", MultiFactorModel, {"weights": [
        None,
        {"volume_ratio": 0.4, "ma_deviation": 0.2, "reversal60": 0.2, "lowvol": 0.2},
        {"volume_ratio": 0.1, "ma_deviation": 0.1, "reversal60": 0.4, "lowvol": 0.4},
    ]}),
]


def run_screening(close: pd.DataFrame, volume: pd.DataFrame,
                  benchmark_close: pd.Series, top_n: int = 50,
                  max_drawdown_floor: float = -0.35) -> pd.DataFrame:
    tr, va = split_dates(close.index)
    rows = []
    bench_close = benchmark_close.reindex(close.index)
    rdates = monthly_rebalance_dates(va)
    bench_monthly = bench_close.loc[rdates].pct_change(fill_method=None).dropna()
    bm = metrics_from_returns(bench_monthly, periods_per_year=12)
    rows.append({"model": "benchmark", "params": "-", "sharpe": bm["sharpe"],
                 "max_drawdown": bm["max_drawdown"], "keep": True,
                 "reason": "基准：沪深300买入持有"})
    for name, cls, grid in CANDIDATES:
        best = grid_search(cls, grid, close, volume, tr, top_n=top_n)
        m = evaluate(cls(**{k: v for k, v in best.items() if k != "sharpe"}),
                     close, volume, va, top_n=top_n)
        keep = m["sharpe"] > bm["sharpe"] and m["max_drawdown"] > max_drawdown_floor
        reason = ("样本外夏普 %.2f > 基准 %.2f 且回撤可控" % (m["sharpe"], bm["sharpe"])
                  if keep else "样本外夏普 %.2f <= 基准 %.2f 或回撤过深" % (m["sharpe"], bm["sharpe"]))
        rows.append({"model": name, "params": str(best), "sharpe": m["sharpe"],
                     "max_drawdown": m["max_drawdown"], "keep": keep, "reason": reason})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_screening.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/screening.py tests/test_screening.py
git commit -m "feat: 训练/验证划分、网格搜索与模型筛选"
```

---

## Task 5: CLI select 与模型筛选报告

**Files:**
- Modify: `ashare_quant/cli.py`
- Test: `tests/test_report.py`（追加 select 报告辅助函数测试）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_report.py 追加
def test_screening_markdown(tmp_path):
    from ashare_quant.research.report import screening_to_markdown
    import pandas as pd
    df = pd.DataFrame({"model": ["benchmark", "reversal"], "params": ["-", "{'horizon': 60}"],
                       "sharpe": [0.5, 0.8], "max_drawdown": [-0.2, -0.1],
                       "keep": [True, True], "reason": ["基准", "样本外胜出"]})
    out = tmp_path / "model-selection.md"
    screening_to_markdown(df, out)
    text = out.read_text(encoding="utf-8")
    assert "reversal" in text and "不构成投资建议" in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_report.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

在 `ashare_quant/research/report.py` 追加：

```python
def screening_to_markdown(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 候选模型筛选报告（M3）", "",
             "> 模拟研究，仅用于数据分析与学习，不构成投资建议。", "",
             "## 筛选结果", ""]
    for _, row in df.iterrows():
        mark = "保留" if row["keep"] else "淘汰"
        lines.append(f"- **{row['model']}**（{mark}）：夏普 {row['sharpe']:.2f}，"
                     f"最大回撤 {row['max_drawdown']:.2%}，参数 {row['params']}。{row['reason']}")
    lines += ["", "## 说明", "",
              "训练段用于网格搜索定参，验证段为样本外检验；筛选规则为样本外夏普高于基准且回撤可控。"]
    path.write_text("\n".join(lines), encoding="utf-8")
```

在 `ashare_quant/cli.py` 追加：

```python
def cmd_select(args) -> None:
    import json

    from .pipeline import build_panels
    from .research.report import screening_to_markdown
    from .screening import run_screening

    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    panels = build_panels(store)
    close, volume = panels["close"], panels["volume"]
    bench = panels["index_close"]
    if bench.empty:
        from .fetchers import akshare_fetcher
        idx_df = akshare_fetcher.fetch_index_daily("sh000300")
        store.save("sh000300", idx_df)
        bench = idx_df["close"]
    out = run_screening(close, volume, bench, top_n=cfg.top_n)
    target = Path(args.out)
    screening_to_markdown(out, target)
    (target.with_suffix(".json")).write_text(
        json.dumps(out.to_dict(orient="records"), ensure_ascii=False, default=str), encoding="utf-8")
    print(f"模型筛选报告已生成: {target}")
    print(out.to_string(index=False))
```

并在 `main()` 注册：

```python
    s = sub.add_parser("select", help="运行候选模型筛选")
    s.add_argument("--config", default="config.yaml")
    s.add_argument("--data-root")
    s.add_argument("--out", default="docs/research/model-selection.md")
    s.set_defaults(func=cmd_select)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/ -q`
Expected: PASS（全部通过）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/cli.py ashare_quant/research/report.py tests/test_report.py
git commit -m "feat: CLI select 与模型筛选报告"
```

---

## Task 6: 端到端运行与计划自检

- [ ] **Step 1: 全量单元测试**

Run: `python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 2: 端到端筛选（真实数据 data/3y）**

Run: `python -m ashare_quant.cli select --data-root data/3y --out docs/research/model-selection.md`
Expected: 输出保留 2~4 个模型；`docs/research/model-selection.md` 与 `.json` 生成。

- [ ] **Step 3: 提交结果**

```bash
git add docs/research
git commit -m "docs: 端到端模型筛选结果"
```
