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


def test_walk_forward_ml_evaluate_accepts_and_applies_costs():
    """ML 评估路径必须能扣成本 —— 否则基准表里两种口径混排。

    背景（2026-09-18 审查发现）：`benchmark.py` 的基准/集成/保形各行都传了
    `costs=BENCH_COSTS`（净），**唯独 ML 模型行经 `walk_forward_ml_evaluate` 是毛收益**
    —— 而报告抬头写的是"净收益"。这条测试钉住"该函数支持 costs 且真的生效"。

    还原旧行为（把 costs 参数删掉、调用不传）时本测试必红。
    """
    import inspect

    from ashare_quant.ml.evaluate import walk_forward_ml_evaluate

    sig = inspect.signature(walk_forward_ml_evaluate)
    assert "costs" in sig.parameters, (
        "walk_forward_ml_evaluate 必须接受 costs —— 否则 ML 行只能是毛收益，"
        "与同表内传了成本的基准/集成行口径不一致")

    # 端到端：构造一个"价格全平 + 每期换仓"的场景，净收益必须为负、毛收益必须为 0
    idx = pd.date_range("2023-01-02", periods=140, freq="B")
    syms = [f"S{i:03d}" for i in range(8)]
    close = pd.DataFrame(10.0, index=idx, columns=syms)
    # 特征：让每期排名都翻转（保证每期满额换手）
    rows = []
    for d_i, d in enumerate(idx):
        for s_i, s in enumerate(syms):
            rows.append((d, s, float((d_i + s_i) % 7)))
    X = pd.DataFrame(rows, columns=["date", "symbol", "f1"]).set_index(["date", "symbol"])
    y = pd.Series(0.0, index=X.index)

    class _Flip:
        def fit(self, X_, y_):  # noqa: N803
            return self

        def predict(self, X_):  # noqa: N803
            return X_["f1"].to_numpy()

    folds = [(idx[:100], idx[100:])]
    costs = {"commission": 0.00025, "stamp": 0.0005, "slippage": 0.001}
    _, gross = walk_forward_ml_evaluate("flip", _Flip, X, y, close,
                                        folds=folds, top_n=2, costs=None)
    _, net = walk_forward_ml_evaluate("flip", _Flip, X, y, close,
                                      folds=folds, top_n=2, costs=costs)
    assert len(gross) > 0, "构造的场景必须真的产生调仓期，否则这条测试是空转"
    assert abs(float(gross.mean())) < 1e-12, "价格全平 → 毛收益应为 0"
    assert float(net.mean()) < 0, (
        "传了 costs 之后净收益必须严格小于毛收益（本例应扣成负数）—— "
        "若仍为 0，说明 costs 没被真正传到 simple_topn_returns")

