# 数据底座 + 历史数据研究（研究阶段）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成 A 股真实数据底座（AKShare 主 + BaoStock 备、本地缓存、校验），并完成"研究先行"阶段：对全市场历史数据做统计检验，产出一份带编号结论（R1..R6）的《数据研究报告》。

**Architecture:** 项目位于 `projects/ashare-quant-analysis/`，Python 包 `ashare_quant` 按职责拆分为 `data`（抓取/缓存/校验/流水线）、`research`（六类统计检验）、`report`（报告生成）、`cli`（命令入口）。面板数据统一为 `DataFrame(index=date, columns=symbol)`；行情缓存统一为带 `date` 索引的 Parquet。先做"沪深 300 快速验证模式"跑通全流程，再扩展全市场。

**Tech Stack:** Python 3.11+（本机 3.13）、pandas、numpy、scipy、statsmodels、scikit-learn、akshare、baostock、plotly、pyarrow、PyYAML、pytest。

**后续计划（本次不实现）：** 候选模型与筛选（M3）、模拟盘与反馈调整（M4）、每日增量更新与自动执行（M5）、全市场数据补齐（M6）。这些在《数据研究报告》产出后另行制定。

---

## Task 1: 项目脚手架与配置

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `config.yaml`
- Create: `ashare_quant/__init__.py`
- Create: `ashare_quant/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
from pathlib import Path
from ashare_quant.config import Config


def test_defaults_when_yaml_missing(tmp_path):
    cfg = Config.from_yaml(tmp_path / "nope.yaml")
    assert cfg.universe_mode == "csi300"
    assert cfg.years == 3
    assert cfg.adjust == "qfq"


def test_from_dict_overrides():
    cfg = Config.from_dict({"years": 5, "universe_mode": "all"})
    assert cfg.years == 5
    assert cfg.universe_mode == "all"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL（`ModuleNotFoundError: ashare_quant`）

- [ ] **Step 3: 安装依赖并写实现**

Run: `python -m pip install -r requirements.txt`

```text
# requirements.txt
pandas
numpy
scipy
statsmodels
scikit-learn
akshare
baostock
plotly
pyarrow
PyYAML
pytest
```

```text
# .gitignore
data/
__pycache__/
.pytest_cache/
*.pyc
```

```yaml
# config.yaml
data_root: data
universe_mode: csi300   # csi300 | all
years: 3
adjust: qfq
max_workers: 8
retry: 3
rate_limit_per_second: 2
top_n: 50
rebalance: M
```

```python
# ashare_quant/__init__.py
"""A股全市场量化研究·模拟分析系统（研究阶段）。"""
__version__ = "0.1.0"
```

```python
# ashare_quant/config.py
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    data_root: Path = Path("data")
    universe_mode: str = "csi300"
    years: int = 3
    adjust: str = "qfq"
    max_workers: int = 8
    retry: int = 3
    rate_limit_per_second: float = 2.0
    top_n: int = 50
    rebalance: str = "M"

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        if "data_root" in kwargs:
            kwargs["data_root"] = Path(kwargs["data_root"])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        path = Path(path)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return cls.from_dict(yaml.safe_load(f) or {})
        return cls()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add requirements.txt .gitignore config.yaml ashare_quant tests
git commit -m "chore: 项目脚手架与配置模块"
```

---

## Task 2: 交易日历

**Files:**
- Create: `ashare_quant/calendar.py`
- Test: `tests/test_calendar.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_calendar.py
import pandas as pd
from ashare_quant.calendar import TradingCalendar


def test_window_and_contains():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    cal = TradingCalendar.from_dates(dates)
    assert cal.count() == 4
    assert cal.contains("2024-01-03")
    assert not cal.contains("2024-01-06")
    assert len(cal.window("2024-01-03", "2024-01-05")) == 3


def test_dedup_and_sort():
    dates = pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-04"])
    cal = TradingCalendar.from_dates(dates)
    assert cal.all_dates.tolist() == pd.to_datetime(["2024-01-02", "2024-01-04"]).tolist()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_calendar.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写实现**

```python
# ashare_quant/calendar.py
from __future__ import annotations

import pandas as pd


class TradingCalendar:
    """由日期序列构造的交易日历（去重、升序）。"""

    def __init__(self, dates) -> None:
        self._dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates))))

    @classmethod
    def from_dates(cls, dates) -> "TradingCalendar":
        return cls(dates)

    @property
    def all_dates(self) -> pd.DatetimeIndex:
        return self._dates

    def contains(self, date) -> bool:
        return pd.Timestamp(date) in self._dates

    def window(self, start, end) -> pd.DatetimeIndex:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return self._dates[(self._dates >= s) & (self._dates <= e)]

    def count(self) -> int:
        return len(self._dates)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_calendar.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/calendar.py tests/test_calendar.py
git commit -m "feat: 交易日历工具"
```

---

## Task 3: 股票池

**Files:**
- Create: `ashare_quant/universe.py`
- Test: `tests/test_universe.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_universe.py
import pandas as pd
from ashare_quant.universe import filter_universe


def test_filter_universe_drops_st():
    raw = pd.DataFrame({"code": ["000001", "000002", "600001"], "name": ["平安银行", "*ST海投", "华夏银行"]})
    out = filter_universe(raw)
    assert out["code"].tolist() == ["000001", "600001"]


def test_filter_universe_keeps_all_when_disabled():
    raw = pd.DataFrame({"code": ["000001"], "name": ["*ST海投"]})
    out = filter_universe(raw, drop_st=False)
    assert len(out) == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_universe.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/universe.py
from __future__ import annotations

import pandas as pd


def filter_universe(frame: pd.DataFrame, drop_st: bool = True) -> pd.DataFrame:
    df = frame.copy()
    if drop_st and "name" in df.columns:
        mask = ~df["name"].str.upper().str.contains("ST|退", na=False)
        df = df[mask]
    return df.reset_index(drop=True)


def _csi300() -> list[str]:
    import akshare as ak

    cons = ak.index_stock_cons(symbol="000300")
    codes = cons["品种代码"].astype(str).str.zfill(6).tolist()
    return sorted(set(codes))


def _all_a() -> list[str]:
    import akshare as ak

    raw = ak.stock_info_a_code_name()
    codes = filter_universe(raw)["code"].astype(str).str.zfill(6).tolist()
    return sorted(set(codes))


def load_universe(mode: str = "csi300") -> list[str]:
    if mode == "csi300":
        return _csi300()
    return _all_a()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_universe.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/universe.py tests/test_universe.py
git commit -m "feat: 股票池获取与ST过滤"
```

---

## Task 4: Parquet 缓存层

**Files:**
- Create: `ashare_quant/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_cache.py
import pandas as pd
import pytest
from ashare_quant.cache import ParquetStore


def _df(dates, offset=0.0):
    idx = pd.to_datetime(dates)
    n = len(idx)
    return pd.DataFrame(
        {"open": [10 + offset] * n, "high": [11 + offset] * n, "low": [9 + offset] * n,
         "close": [10.5 + offset] * n, "volume": [1000] * n, "amount": [1e6] * n},
        index=idx,
    )


def test_roundtrip(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02", "2024-01-03"]))
    loaded = store.load("000001")
    assert list(loaded.index) == pd.to_datetime(["2024-01-02", "2024-01-03"]).tolist()
    assert store.symbols() == ["000001"]


def test_append_dedups(tmp_path):
    store = ParquetStore(tmp_path)
    store.append("000001", _df(["2024-01-02", "2024-01-03"]))
    store.append("000001", _df(["2024-01-03", "2024-01-04"]))
    loaded = store.load("000001")
    assert len(loaded) == 3
    m = store.read_manifest()
    assert m["000001"]["rows"] == 3


def test_missing_symbol_returns_none(tmp_path):
    store = ParquetStore(tmp_path)
    assert store.load("999999") is None
    assert not store.exists("999999")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cache.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/cache.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


class ParquetStore:
    """按 symbol 存 Parquet，manifest.json 记录区间与行数。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def _path(self, symbol: str) -> Path:
        return self.root / f"{symbol}.parquet"

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        out = df.copy()
        out.index.name = "date"
        out.to_parquet(self._path(symbol))
        self.update_manifest(symbol, out)

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self._path(symbol)
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def append(self, symbol: str, df: pd.DataFrame) -> None:
        old = self.load(symbol)
        merged = df if old is None else pd.concat([old, df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.save(symbol, merged)

    def exists(self, symbol: str) -> bool:
        return self._path(symbol).exists()

    def symbols(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.parquet"))

    def read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def update_manifest(self, symbol: str, df: pd.DataFrame) -> None:
        m = self.read_manifest()
        m[symbol] = {
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "rows": int(len(df)),
        }
        self.manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_cache.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/cache.py tests/test_cache.py
git commit -m "feat: Parquet 本地缓存与 manifest"
```

---

## Task 5: AKShare 行情抓取器

**Files:**
- Create: `ashare_quant/fetchers/__init__.py`
- Create: `ashare_quant/fetchers/akshare_fetcher.py`
- Test: `tests/test_akshare_fetcher.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_akshare_fetcher.py
import sys
import types

import pandas as pd
from ashare_quant.fetchers import akshare_fetcher


def test_normalize_maps_columns():
    raw = pd.DataFrame(
        {"日期": ["2024-01-02", "2024-01-03"], "开盘": [10, 10.2], "最高": [10.5, 10.6],
         "最低": [9.8, 10.0], "收盘": [10.3, 10.4], "成交量": [1000, 1200], "成交额": [1e6, 1.2e6]}
    )
    out = akshare_fetcher._normalize(raw)
    assert out.index.name == "date"
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert float(out.loc["2024-01-02", "close"]) == 10.3


def test_fetch_daily_uses_akshare(monkeypatch):
    class FakeAK:
        @staticmethod
        def stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
            assert symbol == "000001"
            assert start_date == "20240101"
            return pd.DataFrame({"日期": ["2024-01-02"], "开盘": [10], "最高": [11],
                                 "最低": [9], "收盘": [10.5], "成交量": [1000], "成交额": [1e6]})

    monkeypatch.setitem(sys.modules, "akshare", FakeAK())
    df = akshare_fetcher.fetch_daily("000001", "2024-01-01", "2024-01-31")
    assert len(df) == 1
    assert df.index[0] == pd.Timestamp("2024-01-02")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_akshare_fetcher.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/fetchers/__init__.py
"""行情数据抓取器（AKShare 主、BaoStock 备）。"""
```

```python
# ashare_quant/fetchers/akshare_fetcher.py
from __future__ import annotations

import pandas as pd

_RENAME = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
           "收盘": "close", "成交量": "volume", "成交额": "amount"}
_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", *_COLS]].set_index("date").sort_index()
    return df.astype({c: float for c in _COLS})


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """返回 date 索引、open/high/low/close/volume/amount 的标准面板。"""
    import akshare as ak

    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=str(start).replace("-", ""),
        end_date=str(end).replace("-", ""),
        adjust=adjust,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=_COLS)
    return _normalize(raw)


def fetch_index_daily(symbol: str = "sh000300") -> pd.DataFrame:
    """沪深指数日线（用于交易日历与市场状态研究）。"""
    import akshare as ak

    raw = ak.stock_zh_index_daily(symbol=symbol)
    raw = raw.rename(columns={"date": "date", "open": "open", "high": "high",
                              "low": "low", "close": "close", "volume": "volume"})
    out = raw[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["amount"] = 0.0
    return out.set_index("date")[["open", "high", "low", "close", "volume", "amount"]].sort_index()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_akshare_fetcher.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/fetchers tests/test_akshare_fetcher.py
git commit -m "feat: AKShare 行情与指数抓取器"
```

---

## Task 6: BaoStock 回退抓取器

**Files:**
- Create: `ashare_quant/fetchers/baostock_fetcher.py`
- Test: `tests/test_baostock_fetcher.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_baostock_fetcher.py
import pandas as pd
from ashare_quant.fetchers.baostock_fetcher import rows_to_frame, to_baostock_code


def test_code_mapping():
    assert to_baostock_code("600000") == "sh.600000"
    assert to_baostock_code("000001") == "sz.000001"
    assert to_baostock_code("688001") == "sh.688001"
    assert to_baostock_code("430047") == "bj.430047"


def test_rows_to_frame():
    fields = ["date", "open", "high", "low", "close", "volume", "amount"]
    rows = [["2024-01-02", "10.0", "11.0", "9.0", "10.5", "1000", "1000000"]]
    df = rows_to_frame(fields, rows)
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert float(df.loc[df.index[0], "close"]) == 10.5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_baostock_fetcher.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/fetchers/baostock_fetcher.py
from __future__ import annotations

import pandas as pd

_COLS = ["open", "high", "low", "close", "volume", "amount"]


def to_baostock_code(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith(("60", "68", "90")):
        return "sh." + s
    if s.startswith(("00", "30", "20")):
        return "sz." + s
    return "bj." + s


def rows_to_frame(fields: list[str], rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=fields)
    for c in _COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[_COLS].sort_index()


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    import baostock as bs

    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            to_baostock_code(symbol),
            "date,open,high,low,close,volume,amount",
            start_date=str(start),
            end_date=str(end),
            frequency="d",
            adjustflag="2" if adjust == "qfq" else "3",
        )
        rows: list[list] = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame(columns=_COLS)
        return rows_to_frame(rs.fields, rows)
    finally:
        bs.logout()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_baostock_fetcher.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/fetchers/baostock_fetcher.py tests/test_baostock_fetcher.py
git commit -m "feat: BaoStock 回退抓取器"
```

---

## Task 7: 数据校验

**Files:**
- Create: `ashare_quant/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_validation.py
import pandas as pd
from ashare_quant.calendar import TradingCalendar
from ashare_quant.validation import validate_symbol


def _df(dates, close):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": close, "high": [c + 0.1 for c in close], "low": [c - 0.1 for c in close],
         "close": close, "volume": [1000] * len(idx), "amount": [1e6] * len(idx)},
        index=idx,
    )


def test_duplicate_and_nonpositive():
    df = _df(["2024-01-02", "2024-01-02"], [10, 0])
    issues = validate_symbol("000001", df, None)
    rules = {i.rule for i in issues}
    assert "duplicate_dates" in rules
    assert "nonpositive_price" in rules


def test_missing_dates_exceeds_threshold():
    cal = TradingCalendar.from_dates(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
    df = _df(["2024-01-02", "2024-01-05"], [10, 11])
    issues = validate_symbol("000001", df, cal, missing_threshold=0.1)
    assert any(i.rule == "missing_dates" for i in issues)


def test_empty_flagged():
    issues = validate_symbol("000001", None, None)
    assert issues[0].rule == "empty"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_validation.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/validation.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationIssue:
    symbol: str
    rule: str
    detail: str


def validate_symbol(symbol: str, df: pd.DataFrame | None, calendar, missing_threshold: float = 0.05) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if df is None or df.empty:
        return [ValidationIssue(symbol, "empty", "no data")]
    if df.index.duplicated().any():
        issues.append(ValidationIssue(symbol, "duplicate_dates", str(int(df.index.duplicated().sum()))))
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        issues.append(ValidationIssue(symbol, "nonpositive_price", "price <= 0"))
    ret = df["close"].pct_change().dropna()
    if (ret.abs() > 0.21).any():
        issues.append(ValidationIssue(symbol, "limit_move", f"max |ret| {ret.abs().max():.3f}"))
    if calendar is not None:
        present = set(df.index)
        missing = [d for d in calendar.all_dates if d not in present]
        if missing and len(missing) / calendar.count() > missing_threshold:
            issues.append(ValidationIssue(symbol, "missing_dates", f"{len(missing)} missing"))
    return issues


def validate_panel(store, calendar, missing_threshold: float = 0.05) -> dict:
    report: dict[str, list[str]] = {}
    for symbol in store.symbols():
        issues = validate_symbol(symbol, store.load(symbol), calendar, missing_threshold)
        if issues:
            report[symbol] = [i.rule for i in issues]
    return report
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_validation.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/validation.py tests/test_validation.py
git commit -m "feat: 数据校验规则"
```

---

## Task 8: 数据流水线与 CLI fetch（含快速验证模式）

**Files:**
- Create: `ashare_quant/pipeline.py`
- Create: `ashare_quant/cli.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pipeline.py
import pandas as pd
from ashare_quant.cache import ParquetStore
from ashare_quant.config import Config
from ashare_quant.pipeline import build_panels, download_universe


def _df(dates, close):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": close, "high": [c + 0.1 for c in close], "low": [c - 0.1 for c in close],
         "close": close, "volume": [1000] * len(idx), "amount": [1e6] * len(idx)},
        index=idx,
    )


def test_download_universe_with_fake_fetcher(tmp_path):
    store = ParquetStore(tmp_path)
    cfg = Config.from_dict({"years": 1, "retry": 1})
    calls: list[str] = []

    def fake_fetcher(code, start, end, adjust):
        calls.append(code)
        return _df(["2024-01-02", "2024-01-03"], [10, 10.5])

    res = download_universe(["000001", "000002"], store, cfg, fetcher=fake_fetcher)
    assert set(calls) == {"000001", "000002"}
    assert res["ok"] == ["000001", "000002"]
    assert store.exists("000001")


def test_download_universe_skips_existing(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02"], [10]))
    cfg = Config.from_dict({"years": 1, "retry": 1})
    res = download_universe(["000001"], store, cfg, fetcher=lambda *a, **k: _df(["2024-01-03"], [11]))
    assert res["skipped"] == ["000001"]


def test_build_panels(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02", "2024-01-03"], [10, 10.5]))
    store.save("000002", _df(["2024-01-02", "2024-01-03"], [20, 19]))
    panels = build_panels(store)
    assert panels["close"].shape == (2, 2)
    assert panels["close"].columns.tolist() == ["000001", "000002"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/pipeline.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd

from .cache import ParquetStore
from .config import Config


def _fetch_one(code: str, cfg: Config, store: ParquetStore, fetcher) -> str:
    if store.exists(code):
        return "skipped"
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=cfg.years)).strftime("%Y%m%d")
    end = pd.Timestamp.today().normalize().strftime("%Y%m%d")
    for attempt in range(max(1, cfg.retry)):
        try:
            df = fetcher(code, start, end, cfg.adjust)
            if df.empty:
                return "no_data"
            store.append(code, df)
            return "ok"
        except Exception:
            if attempt == max(1, cfg.retry) - 1:
                return "failed"
            time.sleep(1)
    return "failed"


def download_universe(codes: list[str], store: ParquetStore, cfg: Config,
                      fetcher=None, universe_name: str = "csi300") -> dict:
    if fetcher is None:
        from .fetchers import akshare_fetcher
        fetcher = akshare_fetcher.fetch_daily
    counts = {"ok": [], "failed": [], "skipped": [], "no_data": []}
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as ex:
        futures = {ex.submit(_fetch_one, c, cfg, store, fetcher): c for c in codes}
        for fut in as_completed(futures):
            status = fut.result()
            counts.setdefault(status, []).append(futures[fut])
    result = {"universe": universe_name, **counts}
    result["rows"] = sum(len(store.load(c)) for c in codes if store.exists(c))
    return result


def build_panels(store: ParquetStore, index_symbol: str = "sh000300") -> dict:
    """把缓存拼成 日期×股票 的面板；指数单独作为 Series。"""
    symbols = [s for s in store.symbols() if s != index_symbol]
    frames = {s: store.load(s)["close"] for s in symbols if store.load(s) is not None}
    close = pd.DataFrame(frames).sort_index()
    volume = pd.DataFrame({s: store.load(s)["volume"] for s in symbols if store.load(s) is not None}).sort_index()
    idx = store.load(index_symbol)
    index_close = idx["close"].sort_index() if idx is not None else pd.Series(dtype=float)
    return {"close": close, "volume": volume, "index_close": index_close}
```

```python
# ashare_quant/cli.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cache import ParquetStore
from .calendar import TradingCalendar
from .config import Config
from .pipeline import download_universe
from .universe import load_universe


def _calendar(cfg: Config, store: ParquetStore) -> TradingCalendar:
    if store.symbols():
        dates = sorted({d for s in store.symbols() for d in store.load(s).index})
        return TradingCalendar.from_dates(dates)
    from .fetchers import akshare_fetcher
    df = akshare_fetcher.fetch_index_daily("sh000300")
    return TradingCalendar.from_dates(df.index)


def cmd_fetch(args) -> None:
    cfg = Config.from_yaml(Path(args.config))
    if args.universe:
        cfg.universe_mode = args.universe
    if args.years:
        cfg.years = args.years
    codes = load_universe(cfg.universe_mode)
    store = ParquetStore(cfg.data_root)
    res = download_universe(codes, store, cfg, universe_name=cfg.universe_mode)
    print(f"universe={res['universe']} ok={len(res['ok'])} skipped={len(res['skipped'])} "
          f"failed={len(res['failed'])} no_data={len(res['no_data'])} rows={res['rows']}")
    if res["failed"]:
        print("failed:", ",".join(res["failed"][:20]))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ashare_quant", description="A股量化研究·模拟分析（研究阶段）")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="下载行情到本地缓存")
    f.add_argument("--config", default="config.yaml")
    f.add_argument("--universe", choices=["csi300", "all"])
    f.add_argument("--years", type=int)
    f.set_defaults(func=cmd_fetch)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 快速验证模式冒烟（真实网络，沪深 300）**

Run: `python -m ashare_quant.cli fetch --universe csi300 --years 1`
Expected: 输出 `universe=csi300 ok=... failed=...`，`data/` 下出现约 300 个 parquet 文件。

- [ ] **Step 6: 提交**

```bash
git add ashare_quant/pipeline.py ashare_quant/cli.py tests/test_pipeline.py
git commit -m "feat: 数据下载流水线与 CLI（快速验证模式）"
```

---

## Task 9: 研究——收益分布与波动聚集

**Files:**
- Create: `ashare_quant/research/__init__.py`
- Create: `ashare_quant/research/stats.py`
- Test: `tests/test_research_stats.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_research_stats.py
import numpy as np
import pandas as pd
from ashare_quant.research.stats import distribution_stats, volatility_clustering


def _close_panel(n_days=1000, n_stocks=50, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    cols = {f"S{i:04d}": 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))) for i in range(n_stocks)}
    return pd.DataFrame(cols, index=idx)


def test_distribution_stats_reports_kurtosis():
    close = _close_panel()
    dist = distribution_stats(close)
    assert "kurtosis" in dist
    assert dist["skewness"] is not None
    assert dist["n_symbols"] == 50


def test_volatility_clustering_on_garch_like_series():
    rng = np.random.default_rng(7)
    n = 2000
    ret = np.zeros(n)
    sigma2 = np.ones(n) * 1e-4
    for t in range(1, n):
        sigma2[t] = 1e-5 + 0.9 * ret[t - 1] ** 2 + 0.08 * sigma2[t - 1]
        ret[t] = rng.normal(0, np.sqrt(sigma2[t]))
    out = volatility_clustering(pd.Series(ret))
    assert out["ljungbox_p"] < 0.05
    assert out["arch_p"] < 0.05
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_research_stats.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/research/__init__.py
"""阶段1：历史数据统计研究。"""
```

```python
# ashare_quant/research/stats.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def distribution_stats(close: pd.DataFrame) -> dict:
    """全市场日收益的分布画像：每只股票算偏度/峰度，再取横截面中位数。"""
    rets = close.pct_change().dropna(how="all")
    skews, kurts = [], []
    for col in rets.columns:
        s = rets[col].dropna()
        if len(s) < 30:
            continue
        skews.append(float(sp_stats.skew(s)))
        kurts.append(float(sp_stats.kurtosis(s)))
    return {
        "n_symbols": len(skews),
        "skewness": float(np.median(skews)) if skews else None,
        "kurtosis": float(np.median(kurts)) if kurts else None,
        "skew_mean": float(np.mean(skews)) if skews else None,
        "kurt_mean": float(np.mean(kurts)) if kurts else None,
    }


def volatility_clustering(returns: pd.Series) -> dict:
    """对指数/代表股票的日收益序列做波动聚集检验。"""
    r = returns.dropna()
    if len(r) < 100:
        return {"ljungbox_p": 1.0, "arch_p": 1.0}
    lb = acorr_ljungbox(r**2, lags=[10], return_df=True)
    arch = het_arch(r, nlags=5)
    return {"ljungbox_p": float(lb["lb_pvalue"].iloc[0]), "arch_p": float(arch[1])}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_research_stats.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/research tests/test_research_stats.py
git commit -m "feat: 研究——收益分布与波动聚集检验"
```

---

## Task 10: 研究——因子库与截面标准化

**Files:**
- Create: `ashare_quant/research/factors.py`
- Test: `tests/test_factors.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_factors.py
import numpy as np
import pandas as pd
from ashare_quant.research.factors import compute_factors, winsorize_zscore


def _panel(n_days=120, n_stocks=30, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    cols = {f"S{i:04d}": 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))) for i in range(n_stocks)}
    return pd.DataFrame(cols, index=idx)


def test_compute_factors_shapes():
    close = _panel()
    volume = pd.DataFrame(1000, index=close.index, columns=close.columns)
    factors = compute_factors(close, volume)
    assert set(factors) == {"momentum", "reversal", "volatility", "ma_deviation", "volume_ratio"}
    assert factors["momentum"].shape == close.shape


def test_winsorize_zscore_cross_sectional():
    df = pd.DataFrame({"a": [1.0, 2.0, 100.0], "b": [2.0, 4.0, 200.0]}, index=[0, 1, 2])
    z = winsorize_zscore(df)
    row0 = z.loc[0]
    assert abs(row0.mean()) < 1e-9
    assert abs(row0.std(ddof=0) - 1) < 1e-9
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_factors.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/research/factors.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factors(close: pd.DataFrame, volume: pd.DataFrame,
                    n_short: int = 5, n_long: int = 20, n_vol: int = 20) -> dict[str, pd.DataFrame]:
    """每天对每只股票计算技术因子（未标准化）。"""
    ret = close.pct_change()
    return {
        "momentum": close.pct_change(n_long),
        "reversal": -close.pct_change(n_short),
        "volatility": ret.rolling(n_vol).std(),
        "ma_deviation": close / close.rolling(n_long).mean() - 1,
        "volume_ratio": volume.rolling(n_short).mean() / volume.rolling(n_long).mean(),
    }


def winsorize_zscore(df: pd.DataFrame, clip: float = 0.01) -> pd.DataFrame:
    """逐日横截面：先去极值（分位截断），再转 z-score（总体标准差，保证每行均值0、标准差1）。"""
    def _row(row: pd.Series) -> pd.Series:
        lo, hi = row.quantile(clip), row.quantile(1 - clip)
        clipped = row.clip(lo, hi)
        mu, sd = clipped.mean(), clipped.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=row.index)
        return (clipped - mu) / sd

    return df.apply(_row, axis=1)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_factors.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/research/factors.py tests/test_factors.py
git commit -m "feat: 研究——因子库与截面标准化"
```

---

## Task 11: 研究——因子有效性（IC/ICIR/分层/稳定性）

**Files:**
- Create: `ashare_quant/research/factor_stats.py`
- Test: `tests/test_factor_stats.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_factor_stats.py
import numpy as np
import pandas as pd
from ashare_quant.research.factor_stats import (cross_sectional_ic, factor_report,
                                                forward_returns, layer_returns, stability_by_year)


def _momentum_panel():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2023-01-02", periods=250, freq="B")
    drift = np.linspace(0.0001, 0.002, 40)
    rets = rng.normal(0, 0.01, (250, 40)) + drift
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                        columns=[f"S{i:04d}" for i in range(40)])


def test_cross_sectional_ic_positive_for_momentum():
    close = _momentum_panel()
    factor = close.pct_change(20)
    target = forward_returns(close, 20)
    ic = cross_sectional_ic(factor, target)
    assert ic.mean() > 0.05


def test_layer_returns_monotonic():
    close = _momentum_panel()
    factor = close.pct_change(20)
    target = forward_returns(close, 20)
    layers = layer_returns(factor, target, k=5)
    assert len(layers) == 5
    assert layers.iloc[-1] > layers.iloc[0]


def test_stability_by_year_and_report():
    close = _momentum_panel()
    rep = factor_report(close, volume=pd.DataFrame(1000, index=close.index, columns=close.columns))
    assert "ic_summary" in rep and "ic_series" in rep
    assert "momentum" in rep["ic_summary"].index
    assert list(stability_by_year(rep["ic_series"]["momentum"]).columns) == ["mean_ic", "hit_rate", "count"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_factor_stats.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/research/factor_stats.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import compute_factors, winsorize_zscore


def forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame:
    return close.shift(-h) / close - 1


def cross_sectional_ic(factor_df: pd.DataFrame, target_df: pd.DataFrame,
                       method: str = "spearman") -> pd.Series:
    """逐日计算因子值与未来收益的截面相关系数。"""
    common = factor_df.index.intersection(target_df.index)
    ic_values = {}
    for d in common:
        f = factor_df.loc[d].dropna()
        t = target_df.loc[d].dropna()
        common_cols = f.index.intersection(t.index)
        if len(common_cols) < 10:
            continue
        ic_values[d] = f[common_cols].corr(t[common_cols], method=method)
    return pd.Series(ic_values, dtype=float).sort_index()


def icir(ic: pd.Series) -> float:
    s = ic.dropna()
    return float(s.mean() / s.std() * np.sqrt(252)) if len(s) > 2 and s.std() > 0 else 0.0


def layer_returns(factor_df: pd.DataFrame, target_df: pd.DataFrame, k: int = 5) -> pd.Series:
    """按因子排名分 k 层，输出每层平均未来收益（组1=最低，组k=最高）。"""
    common = factor_df.index.intersection(target_df.index)
    layers = {d: None for d in common}
    for d in common:
        f = factor_df.loc[d].dropna()
        t = target_df.loc[d].dropna()
        common_cols = f.index.intersection(t.index)
        if len(common_cols) < k * 5:
            continue
        ranks = f[common_cols].rank(method="first")
        buckets = pd.qcut(ranks, k, labels=False) + 1
        layers[d] = t[common_cols].groupby(buckets).mean()
    stacked = pd.DataFrame(layers).T
    return stacked.mean(axis=0)


def stability_by_year(ic: pd.Series) -> pd.DataFrame:
    df = ic.to_frame("ic")
    df["year"] = df.index.year
    grp = df.groupby("year")["ic"]
    out = pd.DataFrame({
        "mean_ic": grp.mean(),
        "hit_rate": grp.apply(lambda s: (s > 0).mean()),
        "count": grp.count(),
    })
    return out


def factor_report(close: pd.DataFrame, volume: pd.DataFrame, h: int = 20,
                  k: int = 5) -> dict:
    factors = compute_factors(close, volume)
    target = forward_returns(close, h)
    summary, series = {}, {}
    for name, raw in factors.items():
        z = winsorize_zscore(raw)
        ic = cross_sectional_ic(z, target)
        layers = layer_returns(z, target, k=k)
        summary[name] = {"mean_ic": float(ic.mean()), "icir": icir(ic),
                         "layer_spread": float(layers.iloc[-1] - layers.iloc[0])}
        series[name] = ic
    return {"ic_summary": pd.DataFrame(summary).T.sort_values("icir", ascending=False),
            "ic_series": series, "layer_returns": layers}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_factor_stats.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/research/factor_stats.py tests/test_factor_stats.py
git commit -m "feat: 研究——因子有效性统计（IC/分层/稳定性）"
```

---

## Task 12: 研究——动量/反转多尺度扫描

**Files:**
- Create: `ashare_quant/research/momentum.py`
- Test: `tests/test_momentum.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_momentum.py
import numpy as np
import pandas as pd
from ashare_quant.research.momentum import horizon_scan


def test_momentum_scan_detects_positive_ic():
    rng = np.random.default_rng(3)
    n_days, n_stocks = 300, 40
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    drift = np.linspace(0.0001, 0.002, n_stocks)  # 强者恒强：高漂移股票持续跑赢
    rets = rng.normal(0, 0.01, (n_days, n_stocks)) + drift
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(n_stocks)])
    out = horizon_scan(close, horizons=(5, 20, 60))
    assert list(out["horizon"]) == [5, 20, 60]
    assert out.loc[out["horizon"] == 60, "mean_ic"].iloc[0] > 0.05
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_momentum.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/research/momentum.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_stats import cross_sectional_ic, forward_returns


def horizon_scan(close: pd.DataFrame, horizons=(5, 10, 20, 60, 120)) -> pd.DataFrame:
    """扫描各持有期的动量/反转效应：过去 h 日收益 vs 未来 h 日收益的截面 IC。"""
    rows = []
    for h in horizons:
        factor = close.pct_change(h)
        target = forward_returns(close, h)
        ic = cross_sectional_ic(factor, target)
        rows.append({"horizon": h, "mean_ic": float(ic.mean()), "icir": float(_icir(ic)),
                     "fwd_mean": float(target.mean(axis=1).mean())})
    return pd.DataFrame(rows)


def _icir(ic: pd.Series) -> float:
    s = ic.dropna()
    return float(s.mean() / s.std() * np.sqrt(252)) if len(s) > 2 and s.std() > 0 else 0.0
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_momentum.py -q`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/research/momentum.py tests/test_momentum.py
git commit -m "feat: 研究——动量/反转多尺度扫描"
```

---

## Task 13: 研究——因子冗余分析（相关性与 PCA）

**Files:**
- Create: `ashare_quant/research/redundancy.py`
- Test: `tests/test_redundancy.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_redundancy.py
import numpy as np
import pandas as pd
from ashare_quant.research.factors import winsorize_zscore
from ashare_quant.research.redundancy import factor_correlation, pca_redundancy


def _duplicated_factors():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-02", periods=100, freq="B")
    base = pd.DataFrame(rng.normal(0, 1, (100, 20)), index=idx,
                        columns=[f"S{i:04d}" for i in range(20)])
    return {"a": base, "b": base.copy(), "c": base + rng.normal(0, 0.5, base.shape)}


def test_factor_correlation_high_for_copies():
    corr = factor_correlation(_duplicated_factors())
    assert corr.loc["a", "b"] > 0.99


def test_pca_first_component_dominant():
    out = pca_redundancy(_duplicated_factors())
    assert out["first_ratio"] > 0.8
    assert out["n_components_80"] == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_redundancy.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/research/redundancy.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .factors import winsorize_zscore


def _daily_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    vals = []
    for d in a.index.intersection(b.index):
        x, y = a.loc[d].dropna(), b.loc[d].dropna()
        common = x.index.intersection(y.index)
        if len(common) >= 10:
            vals.append(x[common].corr(y[common]))
    return float(np.nanmean(vals)) if vals else 0.0


def factor_correlation(factor_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = sorted(factor_dict)
    out = pd.DataFrame(1.0, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = _daily_corr(factor_dict[a], factor_dict[b])
            out.loc[a, b] = out.loc[b, a] = c
    return out


def pca_redundancy(factor_dict: dict[str, pd.DataFrame]) -> dict:
    z = {name: winsorize_zscore(df) for name, df in factor_dict.items()}
    names = sorted(z)
    joined = pd.concat([z[n].stack().rename(n) for n in names], axis=1).dropna()
    pca = PCA(n_components=min(len(names), joined.shape[0]))
    pca.fit(joined.values)
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    n_80 = int(np.searchsorted(cum, 0.8) + 1)
    return {"first_ratio": float(ratios[0]), "cum_ratios": [float(r) for r in cum],
            "n_components_80": n_80}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_redundancy.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/research/redundancy.py tests/test_redundancy.py
git commit -m "feat: 研究——因子冗余分析与PCA"
```

---

## Task 14: 研究——市场状态划分

**Files:**
- Create: `ashare_quant/research/regimes.py`
- Test: `tests/test_regimes.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_regimes.py
import numpy as np
import pandas as pd
from ashare_quant.research.regimes import state_forward_returns, volatility_regimes


def _two_regime_index():
    rng = np.random.default_rng(13)
    idx = pd.date_range("2023-01-02", periods=400, freq="B")
    ret = np.concatenate([rng.normal(0, 0.003, 200), rng.normal(0, 0.02, 200)])
    return pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)


def test_regimes_split_by_volatility():
    idx_close = _two_regime_index()
    reg = volatility_regimes(idx_close, n=10, k=2)
    assert set(reg["state"].dropna().unique()) <= {1, 2}
    assert reg["state"].iloc[:100].mode().iloc[0] == 1
    assert reg["state"].iloc[-100:].mode().iloc[0] == 2


def test_state_forward_returns_has_rows():
    idx_close = _two_regime_index()
    out = state_forward_returns(idx_close, h=10)
    assert len(out) == 2
    assert {"state", "mean_fwd", "count"} <= set(out.columns)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_regimes.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/research/regimes.py
from __future__ import annotations

import pandas as pd


def volatility_regimes(index_close: pd.Series, n: int = 20, k: int = 3) -> pd.DataFrame:
    """用滚动波动率分位把市场分成 k 个状态（1=低波 … k=高波）。"""
    vol = index_close.pct_change().rolling(n).std()
    ranks = vol.rank(method="first")
    state = pd.qcut(ranks, k, labels=False) + 1
    return pd.DataFrame({"vol": vol, "state": state.astype("Int64")})


def state_forward_returns(index_close: pd.Series, h: int = 20) -> pd.DataFrame:
    reg = volatility_regimes(index_close)
    fwd = index_close.shift(-h) / index_close - 1
    df = pd.DataFrame({"state": reg["state"], "fwd": fwd}).dropna()
    out = df.groupby("state")["fwd"].agg(mean_fwd="mean", count="count")
    return out.reset_index()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_regimes.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/research/regimes.py tests/test_regimes.py
git commit -m "feat: 研究——市场状态划分"
```

---

## Task 15: 研究报告生成与 CLI research

**Files:**
- Create: `ashare_quant/research/report.py`
- Modify: `ashare_quant/cli.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_report.py
import pandas as pd
from ashare_quant.research.report import build_report, conclusions


def _results():
    return {
        "distribution": {"kurtosis": 5.2, "skewness": -0.3, "n_symbols": 300},
        "volatility": {"ljungbox_p": 0.001, "arch_p": 0.002},
        "momentum": pd.DataFrame({"horizon": [5, 20, 60], "mean_ic": [-0.01, 0.02, 0.08]}),
        "factor_summary": pd.DataFrame({"icir": [0.5, 0.2]}, index=["momentum", "volume_ratio"]),
        "pca": {"first_ratio": 0.7, "n_components_80": 2},
        "regimes": pd.DataFrame({"state": [1, 2, 3], "mean_fwd": [0.01, 0.02, -0.03]}),
    }


def test_conclusions_numbered():
    conc = conclusions(_results())
    ids = {c.split(" ")[0] for c in conc}
    assert ids == {"R1", "R2", "R3", "R4", "R5", "R6"}


def test_build_report_writes_file(tmp_path):
    out = tmp_path / "research.md"
    build_report(_results(), out)
    text = out.read_text(encoding="utf-8")
    assert "R1" in text and "不构成投资建议" in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_report.py -q`
Expected: FAIL

- [ ] **Step 3: 写实现**

```python
# ashare_quant/research/report.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


def conclusions(results: dict) -> list[str]:
    conc: list[str] = []
    dist = results.get("distribution", {})
    kurt = dist.get("kurtosis") or 0.0
    conc.append(f"R1 收益分布{'呈现厚尾' if kurt > 3 else '接近正态'}（横截面中位峰度 {kurt:.2f}，样本 {dist.get('n_symbols', '?')} 只）")

    vol = results.get("volatility", {})
    p = vol.get("ljungbox_p", 1.0)
    conc.append(f"R2 波动率{'存在显著聚集效应' if p < 0.05 else '未发现显著聚集'}（Ljung-Box p={p:.4f}，ARCH p={vol.get('arch_p', 1.0):.4f}）")

    mom = results.get("momentum")
    if mom is not None and not mom.empty:
        best = mom.loc[mom["mean_ic"].abs().idxmax()]
        direction = "动量" if best["mean_ic"] > 0 else "反转"
        conc.append(f"R3 {direction}效应在 {int(best['horizon'])} 日尺度最强（mean_IC={best['mean_ic']:.3f}，ICIR={best['icir']:.2f}）")

    fac = results.get("factor_summary")
    if fac is not None and not fac.empty:
        strong = fac[fac["icir"].abs() > 0.3].index.tolist()
        conc.append(f"R4 稳定有效因子：{strong if strong else '暂无明显有效因子（|ICIR|>0.3）'}")

    pca = results.get("pca", {})
    r1 = pca.get("first_ratio", 0.0)
    conc.append(f"R5 因子冗余度：第一主成分解释 {r1:.0%}（{'因子高度冗余，需降维' if r1 > 0.5 else '因子相对独立'}）")

    reg = results.get("regimes")
    if reg is not None and not reg.empty and len(reg) > 1:
        spread = reg["mean_fwd"].max() - reg["mean_fwd"].min()
        conc.append(f"R6 不同市场状态下未来收益差异明显（状态间均值差 {spread:.3f}）")
    return conc


def build_report(results: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# A股历史数据研究报告（研究阶段）", "",
             "> 模拟研究，仅用于数据分析与学习，不构成任何投资建议。", "",
             "## 研究结论", ""]
    lines += [f"- {c}" for c in conclusions(results)]
    lines += ["", "## 明细数据", "", "```json", results.get("raw_json", "见同目录 results.json"), "```"]
    path.write_text("\n".join(lines), encoding="utf-8")
```

CLI 增加 research 子命令（在 `ashare_quant/cli.py` 的 `cmd_fetch` 之后追加）：

```python
def cmd_research(args) -> None:
    import json

    from .pipeline import build_panels
    from .research.factors import compute_factors, winsorize_zscore
    from .research import momentum, redundancy, regimes, stats
    from .research.factor_stats import factor_report
    from .research.report import build_report

    cfg = Config.from_yaml(Path(args.config))
    store = ParquetStore(cfg.data_root)
    panels = build_panels(store)
    close, volume, index_close = panels["close"], panels["volume"], panels["index_close"]
    raw_factors = compute_factors(close, volume)
    z_factors = {name: winsorize_zscore(f) for name, f in raw_factors.items()}
    results = {
        "distribution": stats.distribution_stats(close),
        "volatility": stats.volatility_clustering(index_close.pct_change().dropna()),
        "momentum": momentum.horizon_scan(close),
        "factor_report": factor_report(close, volume),
        "pca": redundancy.pca_redundancy(z_factors),
        "regimes": regimes.state_forward_returns(index_close),
    }
    results["factor_summary"] = results["factor_report"]["ic_summary"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results["raw_json"] = json.dumps({k: (v.to_dict() if hasattr(v, "to_dict") else v)
                                      for k, v in results.items()}, ensure_ascii=False, default=str)
    build_report(results, out)
    print(f"研究报告已生成: {out}")
```

并在 `main()` 中注册：

```python
    r = sub.add_parser("research", help="运行历史数据研究并生成报告")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--out", default="docs/research/data-research.md")
    r.set_defaults(func=cmd_research)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_report.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/research/report.py ashare_quant/cli.py tests/test_report.py
git commit -m "feat: 研究报告生成与 CLI research"
```

---

## Task 16: 端到端运行（快速验证模式）与 README

**Files:**
- Create: `README.md`

- [ ] **Step 1: 全量单元测试**

Run: `python -m pytest tests/ -q`
Expected: PASS（全部测试通过）

- [ ] **Step 2: 端到端：拉取沪深 300 数据并生成研究报告**

Run:

```bash
python -m ashare_quant.cli fetch --universe csi300 --years 3
python -m ashare_quant.cli research --out docs/research/2026-08-09-data-research.md
```

Expected: 第一步输出 `ok=...`；第二步生成 `docs/research/2026-08-09-data-research.md`，文件包含 R1~R6 结论与"不构成投资建议"声明。

- [ ] **Step 3: 校验报告数据截止日期与结论合理性**

打开报告，确认：数据截止日期为最近交易日；R1~R6 每一条结论与阶段 1 研究问题一一对应；如某检验因数据不足未执行，报告中明确标注。

- [ ] **Step 4: 写 README**

```markdown
# A股量化研究·模拟分析系统

基于真实 A 股数据（AKShare 主 / BaoStock 备）的历史数据研究项目。

> 本项目为模拟研究，不构成任何投资建议。

## 运行

```bash
pip install -r requirements.txt
python -m ashare_quant.cli fetch --universe csi300 --years 3   # 下载沪深300数据
python -m ashare_quant.cli research                            # 生成研究报告
```

## 阶段

1. 数据底座与历史数据研究（当前）
2. 候选模型构建与筛选（研究结论产出后制定）
3. 模拟盘与反馈调整
4. 每日增量更新与自动执行
```

- [ ] **Step 5: 提交**

```bash
git add README.md docs
git commit -m "docs: 端到端验证与README"
```

---

## 计划自检

- **Spec 覆盖**：数据层（Task 1-8）、阶段1六类检验（Task 9-14）、报告（Task 15）、快速验证模式（Task 8/16）全部对应设计文档第 4、5、10、11、12、14 节。
- **后续计划**：M3 候选模型与筛选、M4 模拟盘与反馈、M5 每日更新与自动执行、M6 全市场补齐在研究报告产出后另行制定。
