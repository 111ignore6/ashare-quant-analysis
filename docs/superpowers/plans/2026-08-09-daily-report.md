# 每日增量更新、HTML 报告与自动执行（M5）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现每个交易日收盘后的增量数据更新（只拉最新交易日）、交互式 HTML 分析报告（Plotly），以及 Windows 计划任务自动执行脚本，让"每日跟进真实数据"可用。

**Architecture:** `ashare_quant/daily.py` 做增量更新（先更指数、再按 manifest 末尾逐股追加）；`ashare_quant/report/html_report.py` 用 Plotly 生成净值/回撤/因子热力图/调整日志 HTML；CLI 增加 `daily`（更新+报告）与 `report`（仅报告）；`scripts/schedule_daily.ps1` 注册 Windows 计划任务。

**Tech Stack:** 新增 plotly（已安装）。

---

## Task 1: 每日增量更新模块

**Files:**
- Create: `ashare_quant/daily.py`
- Test: `tests/test_daily.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_daily.py
import pandas as pd
from ashare_quant.cache import ParquetStore
from ashare_quant.config import Config
from ashare_quant.daily import last_trading_day, needs_update, update_daily


def _df(dates, close):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": close, "high": [c + 0.1 for c in close], "low": [c - 0.1 for c in close],
         "close": close, "volume": [1000] * len(idx), "amount": [1e6] * len(idx)},
        index=idx,
    )


def test_last_trading_day_and_needs_update(tmp_path):
    store = ParquetStore(tmp_path)
    assert last_trading_day(store) is None
    assert needs_update(store)
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000, 3010]))
    assert last_trading_day(store) == pd.Timestamp("2024-01-03")
    assert not needs_update(store)


def test_update_daily_appends_only_missing(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02", "2024-01-03"], [10, 10.5]))

    def fake_index():
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def fake_fetcher(code, start, end, adjust):
        assert code == "000001"
        assert start == "20240104"
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1})
    out = update_daily(["000001"], store, cfg, index_fetcher=fake_index, fetcher=fake_fetcher)
    assert out["new_index_date"] == "2024-01-04"
    assert out["updated"] == ["000001"]
    assert len(store.load("000001")) == 3
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_daily.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/daily.py
from __future__ import annotations

import pandas as pd

from .cache import ParquetStore
from .config import Config


def last_trading_day(store: ParquetStore, index_symbol: str = "sh000300"):
    idx = store.load(index_symbol)
    if idx is not None and len(idx):
        return idx.index.max()
    return None


def needs_update(store: ParquetStore, index_symbol: str = "sh000300") -> bool:
    return last_trading_day(store, index_symbol) is None


def update_daily(codes: list[str], store: ParquetStore, cfg: Config,
                 index_fetcher=None, fetcher=None, index_symbol: str = "sh000300") -> dict:
    if index_fetcher is None:
        from .fetchers import akshare_fetcher
        index_fetcher = akshare_fetcher.fetch_index_daily
    if fetcher is None:
        from .fetchers import akshare_fetcher
        fetcher = akshare_fetcher.fetch_daily
    idx_df = index_fetcher(index_symbol)
    store.append(index_symbol, idx_df)
    last = idx_df.index.max()
    updated, up_to_date, failed = [], [], []
    for code in codes:
        try:
            old = store.load(code)
            if old is None:
                start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=cfg.years)).strftime("%Y%m%d")
            else:
                end_ts = old.index.max()
                if end_ts >= last:
                    up_to_date.append(code)
                    continue
                start = (end_ts + pd.Timedelta(days=1)).strftime("%Y%m%d")
            df = fetcher(code, start, str(last).replace("-", ""), cfg.adjust)
            if not df.empty:
                store.append(code, df)
                updated.append(code)
            else:
                up_to_date.append(code)
        except Exception:
            failed.append(code)
    return {"new_index_date": str(last.date()), "updated": sorted(updated),
            "up_to_date": sorted(up_to_date), "failed": sorted(failed)}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_daily.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/daily.py tests/test_daily.py
git commit -m "feat: 每日增量更新模块"
```

---

## Task 2: Plotly HTML 报告

**Files:**
- Create: `ashare_quant/report/__init__.py`
- Create: `ashare_quant/report/html_report.py`
- Test: `tests/test_html_report.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_html_report.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ashare_quant.report.html_report import (build_html_report, drawdown_figure,
                                             equity_figure, factor_heatmap)


def _returns():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-31", periods=12, freq="ME")
    return pd.DataFrame({"momentum": rng.normal(0.01, 0.03, 12),
                         "rotation": rng.normal(0.008, 0.02, 12)}, index=idx)


def test_figures_are_plotly():
    r = _returns()
    assert isinstance(equity_figure(r), go.Figure)
    assert isinstance(drawdown_figure(r), go.Figure)
    assert isinstance(factor_heatmap(pd.DataFrame({"icir": [0.5, -0.3]}, index=["a", "b"])), go.Figure)


def test_build_html_report_writes_file(tmp_path):
    out = tmp_path / "report.html"
    build_html_report(equity_figure(_returns()), drawdown_figure(_returns()),
                      factor_heatmap(pd.DataFrame({"icir": [0.5]}, index=["a"])),
                      [{"date": "2026-08-07", "trigger": "rotation", "action": "weights",
                        "before": {}, "after": {}, "effect": "轮动"}],
                      data_through="2026-08-07", path=out)
    text = out.read_text(encoding="utf-8")
    assert "plotly" in text and "不构成投资建议" in text and "2026-08-07" in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_html_report.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/report/__init__.py
"""报告生成（Markdown 与 HTML）。"""
```

```python
# ashare_quant/report/html_report.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from ..backtest.metrics import drawdown_series


def equity_figure(model_returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    equity = (1 + model_returns.fillna(0)).cumprod()
    for col in equity.columns:
        fig.add_trace(go.Scatter(x=equity.index, y=equity[col], mode="lines", name=col))
    fig.update_layout(title="模型净值曲线（模拟）", xaxis_title="日期", yaxis_title="净值")
    return fig


def drawdown_figure(model_returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in model_returns.columns:
        dd = drawdown_series(model_returns[col])
        fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=col))
    fig.update_layout(title="回撤曲线（模拟）", xaxis_title="日期", yaxis_title="回撤")
    return fig


def factor_heatmap(ic_summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=ic_summary.T.values, x=ic_summary.index, y=ic_summary.columns,
        colorscale="RdBu", zmid=0))
    fig.update_layout(title="因子有效性热力图（ICIR 等）", xaxis_title="因子", yaxis_title="指标")
    return fig


def build_html_report(equity: go.Figure, drawdown: go.Figure, heatmap: go.Figure,
                      log_entries: list[dict], data_through: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figs = [equity, drawdown, heatmap]
    divs = "".join(f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
                   for i, f in enumerate(figs))
    rows = "".join(
        f"<tr><td>{e.get('date', '')}</td><td>{e.get('trigger', '')}</td>"
        f"<td>{e.get('action', '')}</td><td>{e.get('effect', '')}</td></tr>"
        for e in log_entries[-20:])
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>A股量化研究·模拟分析报告</title></head>
<body><h1>A股量化研究·模拟分析报告</h1>
<p>模拟研究，仅用于数据分析与学习，不构成投资建议。数据截止：{data_through}</p>
{divs}
<h2>调整日志</h2>
<table border="1"><tr><th>日期</th><th>触发</th><th>动作</th><th>说明</th></tr>{rows}</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_html_report.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/report tests/test_html_report.py
git commit -m "feat: Plotly 交互式 HTML 报告"
```

---

## Task 3: CLI daily 与 report

**Files:**
- Modify: `ashare_quant/cli.py`
- Test: `tests/test_cli.py`（新增）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_cli.py
import argparse
import pandas as pd
from ashare_quant.cli import main


def test_main_fetch_help():
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0


def test_main_daily_requires_no_args():
    try:
        main(["daily", "--help"])
    except SystemExit as e:
        assert e.code == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cli.py -q`
Expected: FAIL（daily 子命令不存在）

- [ ] **Step 3: 写实现**

在 `ashare_quant/cli.py` 追加（并在 `main()` 注册 `daily`、`report`）：

```python
def _build_html_report(cfg, store, out_dir) -> None:
    from .models.candidates import LowVolModel, MomentumModel, MultiFactorModel, ReversalModel
    from .pipeline import build_panels
    from .research.factor_stats import factor_report
    from .report.html_report import build_html_report, drawdown_figure, equity_figure, factor_heatmap
    from .simulation import run_simulation

    panels = build_panels(store)
    close, volume = panels["close"], panels["volume"]
    open_ = pd.DataFrame({s: store.load(s)["open"] for s in close.columns}).sort_index()
    models = {"momentum": MomentumModel(60), "reversal": ReversalModel(60),
              "lowvol": LowVolModel(60), "multifactor": MultiFactorModel(
                  {"volume_ratio": 0.4, "ma_deviation": 0.2, "reversal60": 0.2, "lowvol": 0.2})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = run_simulation(models, close, open_, volume, top_n=cfg.top_n,
                         log_path=out_dir / "adjustments.jsonl")
    all_returns = sim["model_returns"].copy()
    all_returns["rotation"] = sim["rotation_returns"]
    summary = factor_report(close, volume)["ic_summary"]
    build_html_report(equity_figure(all_returns), drawdown_figure(all_returns),
                      factor_heatmap(summary), sim["log"].read(),
                      data_through=str(close.index.max().date()), path=out_dir / "report.html")


def cmd_daily(args) -> None:
    from .daily import update_daily
    from .universe import load_universe

    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    codes = load_universe(cfg.universe_mode)
    out = update_daily(codes, store, cfg)
    print(f"指数截止={out['new_index_date']} 更新={len(out['updated'])} "
          f"已最新={len(out['up_to_date'])} 失败={len(out['failed'])}")
    if out["failed"]:
        print("failed:", ",".join(out["failed"][:20]))
    _build_html_report(cfg, store, args.out_dir)
    print(f"当日报告已生成: {args.out_dir}/report.html")


def cmd_report(args) -> None:
    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    _build_html_report(cfg, store, args.out_dir)
    print(f"报告已生成: {args.out_dir}/report.html")
```

`main()` 注册：

```python
    d = sub.add_parser("daily", help="每日增量更新并生成报告")
    d.add_argument("--config", default="config.yaml")
    d.add_argument("--data-root")
    d.add_argument("--out-dir", default="docs/simulation")
    d.set_defaults(func=cmd_daily)
    rep = sub.add_parser("report", help="仅重新生成 HTML 报告")
    rep.add_argument("--config", default="config.yaml")
    rep.add_argument("--data-root")
    rep.add_argument("--out-dir", default="docs/simulation")
    rep.set_defaults(func=cmd_report)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_cli.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/cli.py tests/test_cli.py
git commit -m "feat: CLI daily/report 子命令"
```

---

## Task 4: Windows 计划任务自动执行脚本

**Files:**
- Create: `scripts/schedule_daily.ps1`
- Test: `tests/test_scheduler_script.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_scheduler_script.py
from pathlib import Path


def test_schedule_script_exists():
    p = Path("scripts/schedule_daily.ps1")
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "AshareQuantDaily" in text
    assert "daily" in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_scheduler_script.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```powershell
# scripts/schedule_daily.ps1
param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$DataRoot = "$ProjectRoot\data\3y",
    [string]$OutDir = "$ProjectRoot\docs\simulation"
)

$taskName = "AshareQuantDaily"
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "计划任务 $taskName 已存在，跳过。"
    exit 0
}
$python = (Get-Command python).Source
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 16:05
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
$action = New-ScheduledTaskAction -Execute $python `
    -Argument "-m ashare_quant.cli daily --config `"$ProjectRoot\config.yaml`" --data-root `"$DataRoot`" --out-dir `"$OutDir`"" `
    -WorkingDirectory $ProjectRoot
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "A股量化研究：每个交易日收盘后增量更新数据并生成报告（模拟，不构成投资建议）"
Write-Output "已注册计划任务 $taskName（每周一至五 16:05）"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_scheduler_script.py -q`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add scripts/schedule_daily.ps1 tests/test_scheduler_script.py
git commit -m "feat: Windows 计划任务自动执行脚本"
```

---

## Task 5: 端到端运行与 README 更新

- [ ] **Step 1: 全量单元测试**

Run: `python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 2: 生成 HTML 报告（真实数据 data/3y）**

Run: `python -m ashare_quant.cli report --data-root data/3y --out-dir docs/simulation`
Expected: 生成 `docs/simulation/report.html`（包含 plotly 图与免责声明）。

- [ ] **Step 3: 验证 daily 幂等（无新交易日时不再重复拉取）**

Run: `python -m ashare_quant.cli daily --data-root data/3y --out-dir docs/simulation`
Expected: 输出 `已最新` 数量约等于股票数，报告重新生成。

- [ ] **Step 4: 更新 README 运行说明**

在 README 追加"每日更新"小节：

```markdown
## 每日更新

```powershell
python -m ashare_quant.cli daily          # 增量更新数据并生成当日 HTML 报告
.\scripts\schedule_daily.ps1              # 注册 Windows 计划任务（周一至五 16:05 自动运行）
```
```

- [ ] **Step 5: 提交**

```bash
git add docs/simulation README.md
git commit -m "docs: 端到端每日更新验证与 README"
```
