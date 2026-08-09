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
