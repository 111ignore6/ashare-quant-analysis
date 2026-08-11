"""腾讯行情数据源（直连 web.ifzq.gtimg.cn）。

优点：速度快（并发 12 实测 ~22 只/s）、并发稳定、请求超时完全可控；
注意：单次最多返回 640 根 K 线，长区间按年分批拉取。
"""

from __future__ import annotations

import requests
import pandas as pd

_COLS = ["open", "high", "low", "close", "volume", "amount"]
_UA = {"User-Agent": "Mozilla/5.0"}
_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_REQUEST_TIMEOUT = 12


def to_tx_code(symbol: str) -> str:
    """6 位股票代码 → 腾讯带交易所前缀代码；指数代码原样返回。"""
    s = str(symbol).strip()
    if s.startswith(("sh", "sz", "bj")) and len(s) > 6:
        return s
    s = s.zfill(6)
    if s.startswith(("60", "68", "90")):
        return "sh" + s
    if s.startswith(("00", "30", "20")):
        return "sz" + s
    return "bj" + s


def _parse_kline(rows: list[list]) -> pd.DataFrame:
    """腾讯 K 线行 [date, open, close, high, low, volume] → 标准面板。"""
    if not rows:
        return pd.DataFrame(columns=_COLS)
    # 部分行带第 7 列分红信息 dict，只取前 6 列
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "close", "high", "low", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"] * 100  # 手 → 股
    df["amount"] = (df["volume"] * df["close"]).astype(float)  # 成交额估算
    return df.set_index("date")[_COLS].sort_index()


def _kline(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    url = f"{_KLINE_URL}?param={symbol},day,{start},{end},640,{adjust}"
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers=_UA)
    if resp.status_code == 501 or resp.status_code >= 500:
        # 腾讯 WAF/风控返回 501 反爬页；静默当"无数据"会把大批股票误判为停牌
        # 并写进失败冷却。改为抛异常：上游重试 + 备源确认，批量层触发退避。
        raise RuntimeError(f"腾讯行情源风控/服务器错误（HTTP {resp.status_code}）")
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or symbol not in data["data"]:
        return pd.DataFrame(columns=_COLS)
    data = data["data"][symbol]
    key = f"{adjust}day" if f"{adjust}day" in data else "day"
    return _parse_kline(data.get(key, []))


def _fetch_batched(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """按年分批拉取（腾讯单次最多 640 根）。"""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    chunks, cur = [], start_ts
    while cur <= end_ts:
        seg_end = min(cur + pd.DateOffset(years=1) - pd.Timedelta(days=1), end_ts)
        chunk = _kline(symbol, cur.strftime("%Y-%m-%d"), seg_end.strftime("%Y-%m-%d"), adjust)
        if not chunk.empty:
            chunks.append(chunk)
        cur = seg_end + pd.Timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=_COLS)
    out = pd.concat(chunks)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    # 腾讯长区间返回日终缓存（最新交易日滞后一天），短窗口（≤5 天）才含实时数据；
    # 末行落后于请求 end 时，用最近 5 天窗口补拉并合并。
    if not out.empty and out.index.max().date() < end_ts.date():
        recent = _kline(symbol,
                        (end_ts - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                        end_ts.strftime("%Y-%m-%d"), adjust)
        if not recent.empty:
            out = pd.concat([out, recent])
            out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """返回 date 索引、open/high/low/close/volume/amount 的标准面板。"""
    return _fetch_batched(to_tx_code(symbol), start, end, adjust)


def fetch_index_daily(symbol: str = "sh000300", start: str | None = None) -> pd.DataFrame:
    """指数日线（无复权）；start 指定时只拉该日期之后（增量更新更快）。"""
    start = start or "2000-01-01"
    end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    return _fetch_batched(symbol, start, end, "")
