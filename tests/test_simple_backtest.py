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


def test_simple_topn_returns_charges_turnover_cost():
    """基准的 simple_topn_returns 支持按单边换手扣成本（净收益）。

    背景：它是全部 ML 基准数字的出处，此前一律零成本 —— 而毛收益会系统性
    偏向高换手模型，等于在奖励换手。
    """
    idx = pd.to_datetime(["2026-01-05", "2026-02-02", "2026-03-02"])
    close = pd.DataFrame({"A": [10.0, 10.0, 10.0], "B": [10.0, 10.0, 10.0],
                          "C": [10.0, 10.0, 10.0]}, index=idx)
    score = pd.DataFrame({"A": [3.0, 3.0], "B": [2.0, 2.0], "C": [1.0, 1.0]},
                         index=idx[:2])
    costs = {"commission": 0.00025, "stamp": 0.0005, "slippage": 0.001}
    gross = simple_topn_returns(score, close, idx, top_n=2)
    net = simple_topn_returns(score, close, idx, top_n=2, costs=costs)
    # 价格全平 → 毛收益恒为 0
    assert abs(float(gross.iloc[0])) < 1e-12
    # 首次建仓：全额买入，只付 佣金+滑点（不卖 → 无印花税）
    assert abs(float(net.iloc[0]) + (costs["commission"] + costs["slippage"])) < 1e-12
    # 第二期持仓不变（A,B 仍是前二）→ 换手 0 → 无成本
    assert abs(float(net.iloc[1])) < 1e-12
