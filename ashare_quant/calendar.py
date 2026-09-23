from __future__ import annotations

import pandas as pd


def market_session(now=None) -> str:
    """返回当前市场时段（按工作日时间判断，不区分节假日）：

    pre 盘前(<09:15) / am 上午盘中(09:15-11:30) / lunch 午间休市(11:30-13:00)
    pm 下午盘中(13:00-15:00) / post 收盘后(15:00+) / weekend 周末。

    用途：盘中当日日线未收盘确认，备源（新浪/akshare）当日数据要收盘后才有；
    此时应避免无意义的备源等待，提示用户以收盘后/16:05 自动更新为准。
    """
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    hm = now.hour * 60 + now.minute + now.second / 60.0
    if now.weekday() >= 5:
        return "weekend"
    if hm < 9 * 60 + 15:
        return "pre"
    if hm < 11 * 60 + 30:
        return "am"
    if hm < 13 * 60:
        return "lunch"
    if hm < 15 * 60:
        return "pm"
    return "post"


def drop_intraday_today(df: pd.DataFrame) -> pd.DataFrame:
    """盘中时段去掉"今天"的未收盘行（收盘后/周末原样返回）。

    防止把盘中价当"当日收盘"写入日线/决策/账户——08-11 的 +0.59% 假收益和
    08-12 中午"缺少决策日基准"都是这个根因：盘中更新把当日 bar 当成收盘，
    决策日期随之跳到当天，与本地数据/账户口径错位。收盘后（16:05 计划
    任务）再正式写入当日收盘。
    """
    if df is None or df.empty:
        return df
    if market_session() in ("pre", "am", "lunch", "pm"):
        today = pd.Timestamp.today().normalize()
        return df[df.index < today]
    return df


class TradingCalendar:
    """由日期序列构造的交易日历（去重、升序）。"""

    def __init__(self, dates) -> None:
        self._dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates))))

    @classmethod
    def from_dates(cls, dates) -> TradingCalendar:
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
