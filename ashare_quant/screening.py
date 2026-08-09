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


def walk_forward_folds(dates, train_months: int = 18, valid_months: int = 6,
                       step_months: int = 6):
    """按自然月切出多折训练/样本外验证区间，滚动前进。"""
    dates = pd.DatetimeIndex(sorted(dates))
    months = sorted({d.to_period("M") for d in dates})
    folds = []
    i = 0
    while i + train_months + valid_months <= len(months):
        train_p = months[i:i + train_months]
        valid_p = months[i + train_months:i + train_months + valid_months]
        train_dates = dates[dates.to_period("M").isin(train_p)]
        valid_dates = dates[dates.to_period("M").isin(valid_p)]
        if len(train_dates) >= 120 and len(valid_dates) >= 20:
            folds.append((train_dates, valid_dates))
        i += step_months
    return folds


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


def walk_forward_evaluate(model_cls, param_grid: dict, close: pd.DataFrame,
                          volume: pd.DataFrame, folds, top_n: int = 50) -> tuple:
    """多折滚动验证：每折训练段定参、验证段检验，汇总所有验证段收益。"""
    all_rets, last_params = [], {}
    for train_dates, valid_dates in folds:
        best = grid_search(model_cls, param_grid, close, volume, train_dates, top_n=top_n)
        last_params = {k: v for k, v in best.items() if k != "sharpe"}
        model = model_cls(**last_params) if last_params else model_cls()
        rets = simple_topn_returns(model.score(close, volume), close,
                                   monthly_rebalance_dates(valid_dates), top_n=top_n)
        all_rets.append(rets)
    series = pd.concat(all_rets)
    return series, metrics_from_returns(series, periods_per_year=12), last_params


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
    folds = walk_forward_folds(close.index)
    rows = []
    bench_close = benchmark_close.reindex(close.index)
    valid_rdates = sorted({d for _, va in folds for d in monthly_rebalance_dates(va)})
    bench_monthly = bench_close.loc[valid_rdates].pct_change(fill_method=None).dropna()
    bm = metrics_from_returns(bench_monthly, periods_per_year=12)
    rows.append({"model": "benchmark", "params": "-", "sharpe": bm["sharpe"],
                 "max_drawdown": bm["max_drawdown"], "keep": True,
                 "reason": "基准：沪深300买入持有（walk-forward 验证段）"})
    for name, cls, grid in CANDIDATES:
        _, m, params = walk_forward_evaluate(cls, grid, close, volume, folds, top_n=top_n)
        keep = m["sharpe"] > bm["sharpe"] and m["max_drawdown"] > max_drawdown_floor
        reason = ("样本外夏普 %.2f > 基准 %.2f 且回撤可控" % (m["sharpe"], bm["sharpe"])
                  if keep else "样本外夏普 %.2f <= 基准 %.2f 或回撤过深" % (m["sharpe"], bm["sharpe"]))
        rows.append({"model": name, "params": str(params), "sharpe": m["sharpe"],
                     "max_drawdown": m["max_drawdown"], "keep": keep, "reason": reason})
    return pd.DataFrame(rows)
