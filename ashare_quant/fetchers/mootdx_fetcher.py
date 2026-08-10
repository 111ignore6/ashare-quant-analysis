"""通达信协议数据源（mootdx，1.06k★）。

特点：单次最多返回 800 根日线（实测 ~0.1s），含除权除息信息可自算前复权；
指数走腾讯（通达信指数协议兼容性差）。服务器为公开行情服务器，稳定性次于
腾讯直连，作为可选冗余源。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tencent_fetcher import fetch_index_daily as _tencent_index

_COLS = ["open", "high", "low", "close", "volume", "amount"]
_client = None


def _get_client():
    global _client
    if _client is None:
        from mootdx.quotes import Quotes
        _client = Quotes.factory(market="std")
    return _client


def _to_tdx_code(symbol: str) -> str:
    s = str(symbol).zfill(6)
    return s


def _qfq_adjust(bars: pd.DataFrame, xdxr: pd.DataFrame) -> pd.DataFrame:
    """用除权除息信息做前复权（最新价=原价，历史价格按比例缩放）。

    除权事件：每 10 股分红 fenhong 元、送转 songzhuangu 股、配股 peigu 股
    （配股价 peigujia）。理论除权比例：
      ratio = (P - fenhong/10 + peigu/10*peigujia) / ((1 + songzhuangu/10 + peigu/10) * P)
    其中 P 为事件日前最近收盘价。
    """
    out = bars[["open", "high", "low", "close"]].copy()
    if xdxr is None or xdxr.empty:
        return out
    events = []
    for _, row in xdxr.iterrows():
        fh = float(row.get("fenhong") or 0)
        sz = float(row.get("songzhuangu") or 0)
        pg = float(row.get("peigu") or 0)
        if fh == 0 and sz == 0 and pg == 0:
            continue
        try:
            date = pd.Timestamp(year=int(row["year"]), month=int(row["month"]),
                                day=int(row["day"]))
        except (KeyError, ValueError, TypeError):
            continue
        events.append((date, fh, sz, pg, float(row.get("peigujia") or 0)))
    events.sort()
    close = bars["close"]
    pos_all = close.index.searchsorted([d for d, *_ in events])
    factor = np.ones(len(bars))
    for (_, fh, sz, pg, price), pos in zip(events, pos_all):
        if pos <= 0 or pos > len(bars):
            continue
        p_before = close.iloc[pos - 1]
        if not np.isfinite(p_before) or p_before <= 0:
            continue
        after = (p_before - fh / 10 + pg / 10 * price) / (1 + sz / 10 + pg / 10)
        if after <= 0:
            continue
        factor[:pos] *= after / p_before
    out = out.mul(factor, axis=0)
    out["volume"] = bars["volume"]
    out["amount"] = bars["amount"]
    return out[_COLS]


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """返回 date 索引、open/high/low/close/volume/amount 的标准面板。"""
    client = _get_client()
    bars = client.bars(symbol=_to_tdx_code(symbol), frequency=9, offset=800)
    if bars is None or bars.empty:
        return pd.DataFrame(columns=_COLS)
    bars = bars[["open", "high", "low", "close", "vol", "amount"]].copy()
    bars = bars.rename(columns={"vol": "volume"})
    bars.index = pd.to_datetime(bars.index).normalize()
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    # 按日期区间过滤
    bars = bars.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if bars.empty:
        return pd.DataFrame(columns=_COLS)
    if adjust == "qfq":
        xdxr = client.xdxr(symbol=_to_tdx_code(symbol))
        bars = _qfq_adjust(bars, xdxr)
    return bars


def fetch_index_daily(symbol: str = "sh000300", start: str | None = None) -> pd.DataFrame:
    """指数走腾讯源（通达信指数协议兼容性差）。"""
    return _tencent_index(symbol, start)
