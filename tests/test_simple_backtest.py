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
