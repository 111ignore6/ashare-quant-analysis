"""通达信协议数据源（mootdx，1.06k★）。

特点：单次最多返回 800 根日线（实测并发 ~40 只/s），含除权除息信息可自算
前复权；指数走通达信 index 接口。TCP 7709 协议直连公开行情服务器，不封 IP，
盘中即有当日日线，适合作为全市场主源。北交所（920 号段）标准服务器无数据，
由备源（akshare）兜底。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import threading

_COLS = ["open", "high", "low", "close", "volume", "amount"]
_tls = threading.local()
_XDXR_CACHE: dict[str, pd.DataFrame] = {}


def _get_client():
    """每线程独立通达信连接（共享单连接在并发下会被串行化，实测 4只/s→40只/s）。"""
    if not hasattr(_tls, "client"):
        from mootdx.quotes import Quotes
        _tls.client = Quotes.factory(market="std")
    return _tls.client


def _to_tdx_code(symbol: str) -> str:
    s = str(symbol).zfill(6)
    return s


def _bj_bars(client, code: str) -> pd.DataFrame:
    """北交所日线：通达信市场号 2（mootdx 的 get_stock_market 把 920 新码
    误判为沪市导致返回空，这里直接指定市场）。"""
    raw = client.client.get_security_bars(9, 2, code, 0, 800)
    if not raw:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame(raw)
    df = df.rename(columns={"vol": "volume"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume", "amount"]]
    df.index.name = "date"
    return df


def _qfq_adjust(bars: pd.DataFrame, xdxr: pd.DataFrame) -> pd.DataFrame:
    """用除权除息信息做前复权（最新价=原价，历史价格按比例缩放）。

    除权事件：每 10 股分红 fenhong 元、送转 songzhuangu 股、配股 peigu 股
    （配股价 peigujia）。理论除权比例：
      ratio = (P - fenhong/10 + peigu/10*peigujia) / ((1 + songzhuangu/10 + peigu/10) * P)
    其中 P 为事件日前最近收盘价。
    """
    out = bars[["open", "high", "low", "close"]].copy()
    if xdxr is None or xdxr.empty:
        out["volume"] = bars["volume"]
        out["amount"] = bars["amount"]
        return out
    events = []
    for _, row in xdxr.iterrows():
        # xdxr 里混有大量"公告日/股东大会日"行（分红送配字段全为 NaN），
        # 必须按 NaN→0 处理；若用 float(NaN or 0) 会得到 NaN（NaN 为真值），
        # 随后 after=NaN 会把该事件之前全部历史因子乘成 NaN。
        def _num(key: str) -> float:
            v = row.get(key)
            return float(v) if v is not None and pd.notna(v) else 0.0

        fh = _num("fenhong")
        sz = _num("songzhuangu")
        pg = _num("peigu")
        if fh == 0 and sz == 0 and pg == 0:
            continue
        try:
            date = pd.Timestamp(year=int(row["year"]), month=int(row["month"]),
                                day=int(row["day"]))
        except (KeyError, ValueError, TypeError):
            continue
        events.append((date, fh, sz, pg, _num("peigujia")))
    events.sort()
    close = bars["close"]
    pos_all = close.index.searchsorted([d for d, *_ in events])
    factor = np.ones(len(bars))
    for (_, fh, sz, pg, price), pos in zip(events, pos_all):
        # pos == len(bars) 表示除权事件日期晚于数据最后一天（未来预案未实施）：
        # 前复权以最新价为基准，不应缩放任何行（否则会把最新价也改掉）。
        if pos <= 0 or pos >= len(bars):
            continue
        p_before = close.iloc[pos - 1]
        if not np.isfinite(p_before) or p_before <= 0:
            continue
        after = (p_before - fh / 10 + pg / 10 * price) / (1 + sz / 10 + pg / 10)
        # NaN 因子（如配股价缺失导致 after 非有限）会污染 factor[:pos]，
        # 必须显式跳过，否则整个前复权历史变 NaN。
        if not np.isfinite(after) or after <= 0:
            continue
        factor[:pos] *= after / p_before
    out = out.mul(factor, axis=0)
    out["volume"] = bars["volume"]
    out["amount"] = bars["amount"]
    return out[_COLS]


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """返回 date 索引、open/high/low/close/volume/amount 的标准面板。"""
    client = _get_client()
    code = _to_tdx_code(symbol)
    if code.startswith("920"):
        # 北交所新代码段（mootdx 0.8.7+ 支持，须用 MARKET_BJ=2；
        # 旧 43/83/87 号段已迁移作废，不处理）
        bars = _bj_bars(client, code)
    else:
        bars = client.bars(symbol=code, frequency=9, offset=800)
        if bars is None or bars.empty:
            return pd.DataFrame(columns=_COLS)
        bars = bars[["open", "high", "low", "close", "vol", "amount"]].copy()
        bars = bars.rename(columns={"vol": "volume"})
    if bars.empty:
        return pd.DataFrame(columns=_COLS)
    bars.index = pd.to_datetime(bars.index).normalize()
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    # 按日期区间过滤
    bars = bars.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if bars.empty:
        return pd.DataFrame(columns=_COLS)
    if adjust == "qfq":
        xdxr = _XDXR_CACHE.get(symbol)
        if xdxr is None:
            try:
                xdxr = client.xdxr(symbol=code)
            except Exception:  # noqa: BLE001（北交所/复权信息缺失时按原价返回）
                xdxr = pd.DataFrame()
            _XDXR_CACHE[symbol] = xdxr
        bars = _qfq_adjust(bars, xdxr)
    return bars


def fetch_index_daily(symbol: str = "sh000300", start: str | None = None) -> pd.DataFrame:
    """指数日线（通达信 index 接口；sh000300 → 000300）。"""
    code = str(symbol).replace("sh", "").replace("sz", "").replace("bj", "")
    client = _get_client()
    bars = client.index(symbol=code, frequency=9, offset=800)
    if bars is None or bars.empty:
        return pd.DataFrame(columns=_COLS)
    bars = bars.copy()
    if "vol" in bars.columns and "volume" not in bars.columns:
        bars = bars.rename(columns={"vol": "volume"})
    bars = bars[["open", "high", "low", "close", "volume", "amount"]].copy()
    bars.index = pd.to_datetime(bars.index).normalize()
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    if start:
        bars = bars.loc[pd.Timestamp(start):]
    return bars[_COLS]
