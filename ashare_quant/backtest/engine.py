from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    turnover: pd.Series
    holdings: dict = field(default_factory=dict)


def _next_trading_day(dates, d):
    pos = dates.get_loc(d)
    if pos + 1 >= len(dates):
        return None
    return dates[pos + 1]


def run_backtest(score: pd.DataFrame, close: pd.DataFrame, open_: pd.DataFrame,
                 rebalance_dates, top_n: int = 50, commission: float = 0.00025,
                 stamp: float = 0.0005, slippage: float = 0.001,
                 limit: float = 0.098) -> BacktestResult:
    """月度调仓：信号日选股，次日开盘成交；持仓数量记账，市值法算收益；T+1 由持有期隐含满足。"""
    signal_dates = [d for d in rebalance_dates if d in score.index]
    rets, turnovers, holdings = {}, {}, {}
    shares: dict[str, float] = {}
    prev_holdings: set[str] = set()
    prev_value: float | None = None
    cash = 1.0  # 初始资金
    for i in range(len(signal_dates) - 1):
        d, d2 = signal_dates[i], signal_dates[i + 1]
        exec_day = _next_trading_day(close.index, d)
        next_exec = _next_trading_day(close.index, d2)
        if exec_day is None or next_exec is None:
            continue
        picks = score.loc[d].dropna().nlargest(top_n).index.tolist()
        prev_close = close.shift(1).loc[exec_day]

        def px(symbol):
            p = open_.loc[exec_day, symbol]
            if pd.isna(p):
                p = prev_close[symbol]
            return p

        # 1) 期初市值：现金 + 持仓按开盘（停牌用前收）计值；可卖的先卖出变现
        pos_value = 0.0
        for s in list(shares):
            price = px(s)
            if pd.isna(price):
                continue
            sellable = price > prev_close[s] * (1 - limit)
            if sellable:
                cash += shares[s] * price * (1 - slippage) * (1 - commission - stamp)
                del shares[s]
            else:
                pos_value += shares[s] * price
        value_now = cash + pos_value
        # 2) 记录期间收益
        if prev_value is not None and value_now > 0:
            rets[d] = float(value_now / prev_value - 1)
        # 3) 买入：现金等权分配给可买的新目标（非涨停、非停牌、未持有）
        buyable = [s for s in picks if s not in shares and not pd.isna(px(s))
                   and px(s) < prev_close[s] * (1 + limit)]
        if buyable and cash > 0:
            budget = cash / len(buyable)
            for s in buyable:
                buy_px = px(s) * (1 + slippage) * (1 + commission)
                if buy_px > 0:
                    shares[s] = budget / buy_px
                    cash -= budget
        # 4) 记录换手与持仓，更新期初基准值（交易后市值）
        held = set(shares)
        if prev_value is not None:
            turnovers[d] = len(prev_holdings.symmetric_difference(held)) / max(1, len(prev_holdings | held))
            holdings[d] = sorted(held)
        prev_holdings = held
        prev_value = cash + sum(shares[s] * px(s) for s in shares if not pd.isna(px(s)))
    r = pd.Series(rets)
    return BacktestResult(returns=r, equity=(1 + r).cumprod(), turnover=pd.Series(turnovers), holdings=holdings)
