from __future__ import annotations

import itertools

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
        params = {k: v for k, v in best.items() if k != "sharpe"}
        model = cls(**params) if params else cls()
        m = evaluate(model, close, volume, va, top_n=top_n)
        keep = m["sharpe"] > bm["sharpe"] and m["max_drawdown"] > max_drawdown_floor
        reason = ("样本外夏普 %.2f > 基准 %.2f 且回撤可控" % (m["sharpe"], bm["sharpe"])
                  if keep else "样本外夏普 %.2f <= 基准 %.2f 或回撤过深" % (m["sharpe"], bm["sharpe"]))
        rows.append({"model": name, "params": str(best), "sharpe": m["sharpe"],
                     "max_drawdown": m["max_drawdown"], "keep": keep, "reason": reason})
    return pd.DataFrame(rows)
