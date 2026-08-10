from __future__ import annotations

import socket

import pandas as pd
import requests

_RENAME = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
           "收盘": "close", "成交量": "volume", "成交额": "amount"}
_COLS = ["open", "high", "low", "close", "volume", "amount"]

# akshare 内部请求不传 timeout，默认会无限挂起；这里统一注入默认超时。
_ORIG_SESSION_REQUEST = requests.sessions.Session.request


def _request_with_default_timeout(self, method, url, **kwargs):
    kwargs.setdefault("timeout", 12)
    return _ORIG_SESSION_REQUEST(self, method, url, **kwargs)


requests.sessions.Session.request = _request_with_default_timeout


def to_sina_code(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith(("60", "68", "90")):
        return "sh" + s
    if s.startswith(("00", "30", "20")):
        return "sz" + s
    return "bj" + s


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", *_COLS]].set_index("date").sort_index()
    return df.astype({c: float for c in _COLS})


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """返回 date 索引、open/high/low/close/volume/amount 的标准面板。"""
    import akshare as ak

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(8)
    try:
        raw = ak.stock_zh_a_daily(
            symbol=to_sina_code(symbol),
            start_date=str(start).replace("-", ""),
            end_date=str(end).replace("-", ""),
            adjust=adjust,
        )
    finally:
        socket.setdefaulttimeout(old_timeout)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=_COLS)
    return _normalize(raw)


def fetch_index_daily(symbol: str = "sh000300") -> pd.DataFrame:
    """沪深指数日线（用于交易日历与市场状态研究）。"""
    import akshare as ak

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        raw = ak.stock_zh_index_daily(symbol=symbol)
    finally:
        socket.setdefaulttimeout(old_timeout)
    raw = raw.rename(columns={"date": "date", "open": "open", "high": "high",
                              "low": "low", "close": "close", "volume": "volume"})
    out = raw[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["amount"] = 0.0
    return out.set_index("date")[["open", "high", "low", "close", "volume", "amount"]].sort_index()
