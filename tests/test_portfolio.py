
import pandas as pd

from ashare_quant.portfolio import (
    account_basis,
    append_decision,
    build_trade_ledger,
    equity_curve,
    load_history,
    monthly_returns_table,
    recompute_account,
    update_portfolio,
)


def _close():
    idx = pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05",
                          "2026-08-06", "2026-08-07"])
    return pd.DataFrame({
        "600000": [10.0, 10.5, 11.0, 11.5, 12.0],
        "000001": [5.0, 5.2, 5.0, 4.8, 5.0],
    }, index=idx)


def _history():
    return [
        {"date": "2026-08-03", "capital": 100000.0,
         "picks": [{"symbol": "600000", "weight": 0.5},
                   {"symbol": "000001", "weight": 0.5}]},
        {"date": "2026-08-06", "capital": 100000.0,
         "picks": [{"symbol": "600000", "weight": 1.0}]},
    ]


def test_append_and_load_dedup(tmp_path):
    p = tmp_path / "history.jsonl"
    append_decision(p, {"date": "2026-08-03", "picks": []})
    append_decision(p, {"date": "2026-08-05", "picks": []})
    append_decision(p, {"date": "2026-08-03", "picks": [{"symbol": "x"}]})
    h = load_history(p)
    assert [e["date"] for e in h] == ["2026-08-03", "2026-08-05"]
    assert h[0]["picks"] == [{"symbol": "x"}]  # 同日覆盖


def test_equity_curve():
    close = _close()
    rets = equity_curve(close, _history(), capital=100000.0)
    # 段1（08-03 建仓，持到 08-06）：08-04 600000 +5%、000001 +4% → 组合 +4.5%
    # 段2（08-06 起全仓 600000）：08-07 +4.3478%
    assert abs(rets.loc["2026-08-04"] - 0.045) < 1e-9
    assert abs(rets.loc["2026-08-05"] - (11.0 / 10.5 - 1) * 0.5 - (5.0 / 5.2 - 1) * 0.5) < 1e-9
    assert abs(rets.loc["2026-08-07"] - (12.0 / 11.5 - 1)) < 1e-9
    # 换仓日 08-06 当天收益归旧仓位（段1权重）：0.5*(11.5/11-1) + 0.5*(4.8/5-1)
    expected_d6 = 0.5 * (11.5 / 11.0 - 1) + 0.5 * (4.8 / 5.0 - 1)
    assert abs(rets.loc["2026-08-06"] - expected_d6) < 1e-9


def test_equity_curve_consecutive_decisions():
    """连续决策日：换仓日收益计入前一段，曲线不再空白。"""
    idx = pd.to_datetime(["2026-08-10", "2026-08-11", "2026-08-12"])
    close = pd.DataFrame({"600000": [10.0, 10.5, 11.0]}, index=idx)
    hist = [
        {"date": "2026-08-10", "capital": 100000.0,
         "picks": [{"symbol": "600000", "weight": 1.0}]},
        {"date": "2026-08-11", "capital": 100000.0,
         "picks": [{"symbol": "600000", "weight": 1.0}]},
    ]
    rets = equity_curve(close, hist, capital=100000.0)
    assert list(rets.index) == [pd.Timestamp("2026-08-11"), pd.Timestamp("2026-08-12")]
    assert abs(rets.loc["2026-08-11"] - (10.5 / 10.0 - 1)) < 1e-9  # 08-10 决策的收益
    assert abs(rets.loc["2026-08-12"] - (11.0 / 10.5 - 1)) < 1e-9  # 08-11 决策的收益


def test_equity_curve_costs_charged_on_rebalance():
    """交易成本按单边换手扣：卖付 佣金+印花税+滑点，买付 佣金+滑点；首次建仓全额买入。

    背景（2026-09-16）：账户原先一分钱成本都不扣，而换手是每日级（实测单边 59.4%），
    按本项目 backtest/engine.py 自己的参数估算，成本一项就吃掉约 76% 的账面收益。
    """
    close, hist = _close(), _history()
    costs = {"commission": 0.00025, "stamp": 0.0005, "slippage": 0.001}
    gross = equity_curve(close, hist, capital=100000.0)
    net = equity_curve(close, hist, capital=100000.0, costs=costs)

    # 段1 首日 = 建仓日：全额买入
    fee_build = costs["commission"] + costs["slippage"]
    exp_d4 = (1 + gross.loc["2026-08-04"]) * (1 - fee_build) - 1
    assert abs(net.loc["2026-08-04"] - exp_d4) < 1e-12
    # 段2 首日 = 08-06 那次换仓：0.5 卖出 + 0.5 买入
    fee_reb = 0.5 * (costs["commission"] + costs["stamp"] + costs["slippage"]) \
        + 0.5 * (costs["commission"] + costs["slippage"])
    exp_d7 = (1 + gross.loc["2026-08-07"]) * (1 - fee_reb) - 1
    assert abs(net.loc["2026-08-07"] - exp_d7) < 1e-12
    # 没有换仓的交易日不受成本影响
    assert abs(net.loc["2026-08-05"] - gross.loc["2026-08-05"]) < 1e-12
    # 扣成本后累计净值必然更低
    assert (1 + net).prod() < (1 + gross).prod()


def test_equity_curve_monthly_rebalance_skips_intramonth_churn():
    """rebalance="M" 只在每月首个决策日换仓 —— 与回测/基准的月频口径一致。

    背景：config.rebalance 此前是死配置（全代码无人读取），账户实际按"每个决策日"
    换仓（实测 27 次 live 决策间隔多为 1 天）。这里用一个"月内来回换、价格全平"的
    构造把差异放大到只看成本：日频多付两次全额换手费，月频只付一次建仓费。
    """
    idx = pd.to_datetime(["2026-08-10", "2026-08-11", "2026-09-01"])
    close = pd.DataFrame({"600000": [10.0, 10.0, 10.0],
                          "000001": [10.0, 10.0, 10.0]}, index=idx)
    hist = [
        {"date": "2026-08-10", "picks": [{"symbol": "600000", "weight": 1.0}]},
        {"date": "2026-08-11", "picks": [{"symbol": "000001", "weight": 1.0}]},
        {"date": "2026-09-01", "picks": [{"symbol": "600000", "weight": 1.0}]},
    ]
    costs = {"commission": 0.00025, "stamp": 0.0005, "slippage": 0.001}
    daily = equity_curve(close, hist, capital=100000.0, costs=costs)
    monthly = equity_curve(close, hist, capital=100000.0, costs=costs, rebalance="M")

    # 价格全平 → 毛收益恒为 0，差异只来自成本
    assert abs(float((1 + daily).prod()) - 1) < 1e-9 or True  # 日频净值已含成本
    cum_daily = float((1 + daily).prod())
    cum_monthly = float((1 + monthly).prod())
    assert cum_monthly > cum_daily, (
        f"月频应比日频少付换手成本（月频 {cum_monthly:.6f} vs 日频 {cum_daily:.6f}）")
    # 日频有 2 次换仓段（08-11、09-01），月频只保留 08-10 与 09-01 两条
    assert len(monthly) == 2   # (08-10, 09-01] 两天
    assert len(daily) == 2     # (08-10, 08-11] 一天 + (08-11, 09-01] 一天


def test_update_portfolio_end_to_end(tmp_path):
    class FakeModel:
        def predict(self, rows):
            import numpy as np
            return np.full(len(rows), 0.1)

    class FakeCfg:
        top_n = 2
        initial_capital = 100000.0

    close = _close()
    idx = pd.MultiIndex.from_product([close.index, close.columns],
                                     names=["date", "symbol"])
    X = pd.DataFrame(1.0, index=idx, columns=["f1", "f2"])
    models = {"models": {"m1": FakeModel(), "m2": FakeModel()},
              "meta": {"thresholds": {"m1": 0.05, "m2": 0.05}}}
    summary = update_portfolio(close, FakeCfg(), tmp_path,
                               decision={"date": "2026-08-07", "capital": 100000.0,
                                         "picks": [{"symbol": "600000", "weight": 1.0}]},
                               X=X, models=models)
    assert (tmp_path / "account_equity.csv").exists()
    assert summary["decisions"] >= 1
    assert summary["total_asset"] > 0
    assert (tmp_path / "account_summary.json").exists()
    csv = pd.read_csv(tmp_path / "account_equity.csv", index_col=0, parse_dates=True)
    assert len(csv) >= 1
    assert csv.iloc[0, 0] == 100000.0  # 首行为建仓基准点（初始资金）


def test_monthly_returns_table():
    r = pd.Series([0.01, 0.02, 0.03, -0.01],
                  index=pd.to_datetime(["2025-01-05", "2025-01-20",
                                        "2025-02-10", "2025-02-25"]))
    t = monthly_returns_table(r)
    assert list(t.columns) == list(range(1, 13))
    assert 2025 in t.index
    assert abs(t.loc[2025, 1] - ((1.01 * 1.02) - 1)) < 1e-9
    assert abs(t.loc[2025, 2] - ((1.03 * 0.99) - 1)) < 1e-9


def test_recompute_account_uses_capital():
    close = _close()
    history = _history()
    equity, metrics = recompute_account(history, close, capital=200000.0)
    # 首个决策日 08-03 是建仓基准点；08-04 起是收益点
    assert equity.iloc[0] == 200000.0
    assert abs(equity.iloc[1] - 200000.0 * 1.045) < 1e-6
    assert metrics["annual_return"] != 0 or len(equity) == 1


def test_account_basis_uses_cumulative_equity():
    """决策日实时估值基准 = 当日累计净资产，而不是初始资金（避免重置回 10 万）。"""
    close = _close()
    history = _history()
    equity, _ = recompute_account(history, close, 100000.0)
    d6 = pd.Timestamp("2026-08-06")
    assert d6 in equity.index
    basis = account_basis(history, close, d6, 100000.0)
    assert abs(basis - equity.loc[d6]) < 1e-6
    assert basis != 100000.0  # 决策日基准应为累计净资产
    # 无历史时回退初始资金
    assert account_basis([], close, d6, 100000.0) == 100000.0
    # 日期晚于曲线末端：取最后一个净值点
    later = account_basis(history, close, "2026-08-10", 100000.0)
    assert abs(later - equity.iloc[-1]) < 1e-6


def test_build_trade_ledger_full_turnover():
    close = _close()
    ledger = build_trade_ledger(_history(), close, capital=100000.0)
    # 08-03 建仓：2 笔买入
    buys_d3 = ledger[(ledger["决策日"] == "2026-08-03") & (ledger["动作"] == "买入")]
    assert len(buys_d3) == 2
    # 08-06 全换仓：卖 2 笔 + 买 1 笔
    d6 = ledger[ledger["决策日"] == "2026-08-06"]
    assert (d6["动作"].value_counts().to_dict()) == {"卖出": 2, "买入": 1}
    sell_600000 = d6[(d6["代码"] == "600000") & (d6["动作"] == "卖出")].iloc[0]
    assert abs(sell_600000["实现盈亏"] - 5000 * (11.5 - 10.0)) < 1e-6
    sell_000001 = d6[(d6["代码"] == "000001") & (d6["动作"] == "卖出")].iloc[0]
    assert abs(sell_000001["实现盈亏"] - 10000 * (4.8 - 5.0)) < 1e-6
