"""腾讯批量报价端点（qt.gtimg.cn）——每日增量补齐"最新一根 bar"。

为什么单独一个模块：个股源链（akshare/腾讯 kline）**每只股票至少一次 HTTP**，
而"每日增量"每只只需要最新一根 bar。腾讯报价端点一次可带数百只代码，实测
310~340 只/s（4 线程），且**收盘后当日 bar 立即就有**——新浪个股当日 bar 要
滞后数小时，16:05 跑时只有 D-1，这是 2026-09-16 "updated=4989 但 96% 个股
仍停在 D-1"的直接原因。

口径（2026-09-16 实测：与 akshare 新浪 qfq 当日 bar 逐字段对比 20 只）
- open/high/low/close/amount 与 akshare **完全相等**（相对误差 0）；amount 取
  字段 35 的第三段（元，真实成交额），比腾讯 kline 的 volume×close 估算更准。
- volume：**单位按板块判**（`venues.volume_in_shares`）—— 科创板 688/689 腾讯
  直接给"股"，其余板块给"手"。⚠️ 本条最初写成"一律按手给、×100 得股"，
  样本 20 只里没有 688，于是 2026-09-18 把当日**全市场科创板**成交量放大 100 倍
  （688525 写成 1,889,345,500，真值 18,893,455）。新增任何解析成交量的模块，
  都必须先问 `volume_in_shares`。
- 前复权序列"最新一天 == 不复权最新一天"（qfq 锚点恒等式），因此追加当日原始
  报价与追加 akshare 当日 qfq 逐字段等价；历史部分不动，跨源复权锚点风险不变。
- 停牌股：open/high/low=0、volume=0 → 本模块直接丢弃（由调用方按"无当日 bar"
  处理，与现有 no_data 语义一致）。

只用于"最新一个交易日"：报价里只有当日一根，历史缺口必须走个股源链。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from ..venues import volume_in_shares

_COLS = ["open", "high", "low", "close", "volume", "amount"]
_QUOTE_URL = "https://qt.gtimg.cn/q="
_UA = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 12

# 单请求代码数：实测 400 只可用（URL 约 16KB），留足余量取 120
BATCH_SIZE = 120
WORKERS = 4

# 腾讯报价字段（'~' 分隔）下标
_F_CLOSE = 3      # 最新价（收盘后即当日收盘价）
_F_OPEN = 5       # 今开
_F_TS = 30        # 时间戳 yyyymmddHHMMSS
_F_HIGH = 33
_F_LOW = 34
_F_DETAIL = 35    # "最新价/成交量(手)/成交额(元)"

_tls = threading.local()


def _session() -> requests.Session:
    """线程本地 Session（复用连接；跨线程共享 requests.Session 不安全）。"""
    if not hasattr(_tls, "session"):
        s = requests.Session()
        s.headers.update(_UA)
        _tls.session = s
    return _tls.session


def to_quote_code(symbol: str) -> str:
    """6 位代码 → 腾讯带市场前缀代码（与 kline 源同一规则）。"""
    from .tencent_fetcher import to_tx_code

    return to_tx_code(symbol)


def _num(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def parse_quotes(text: str) -> dict[str, dict]:
    """解析 `v_sh600000="..."` 响应；返回 {代码: {date, ohlcv...}}（已丢弃停牌行）。

    停牌（volume=0 / 价格全 0）与解析失败的行一律不进结果——它们必须由调用方
    走个股源链确认为 no_data，不能在这里被当成"有当日 bar"。
    """
    out: dict[str, dict] = {}
    for line in text.split(";"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        code = key.strip().removeprefix("v_")
        fields = value.strip().strip('"').split("~")
        if len(fields) <= _F_DETAIL:
            continue
        detail = fields[_F_DETAIL].split("/")
        if len(detail) < 3:
            continue
        ts = fields[_F_TS].strip()
        if len(ts) < 8 or not ts[:8].isdigit():
            continue
        open_, high, low, close = (_num(fields[_F_OPEN]), _num(fields[_F_HIGH]),
                                   _num(fields[_F_LOW]), _num(fields[_F_CLOSE]))
        # 成交量单位**按板块**判：腾讯对科创板（688/689）直接给"股"，其余板块给"手"。
        # 无条件 ×100 会把 688 放大 100 倍 —— 2026-09-18 实测：688525 原始值
        # 18,893,455 == 新浪股数，而本地被这条批量路径写成了 1,889,345,500
        # （当日全市场 688 都被写坏）。这是「科创板成交量」那个坑的**第二次复发**，
        # 只是换了一个模块 —— 口径唯一事实来源是 ``venues.volume_in_shares``。
        raw_volume = _num(detail[1])
        volume = raw_volume if volume_in_shares(code) else raw_volume * 100  # 手 → 股
        amount = _num(detail[2])        # 元（真实成交额）
        if not all(v > 0 for v in (open_, high, low, close, volume)):
            continue  # 停牌/无成交：交给个股源链按 no_data 处理
        try:
            date = pd.Timestamp(f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}")
        except ValueError:
            continue
        out[code] = {"date": date, "open": open_, "high": high, "low": low,
                     "close": close, "volume": volume,
                     "amount": amount if amount > 0 else volume * close}
    return out


def _fetch_chunk(codes: list[str], tx_of: dict[str, str]) -> dict[str, pd.DataFrame]:
    resp = _session().get(_QUOTE_URL + ",".join(tx_of[c] for c in codes),
                          timeout=_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"腾讯报价源 HTTP {resp.status_code}")
    resp.encoding = "gbk"
    parsed = parse_quotes(resp.text)
    out: dict[str, pd.DataFrame] = {}
    for code in codes:
        row = parsed.get(tx_of[code])
        if row is None:
            continue
        out[code] = pd.DataFrame(
            [[row["open"], row["high"], row["low"], row["close"],
              row["volume"], row["amount"]]],
            columns=_COLS, index=pd.DatetimeIndex([row["date"]], name="date"))
    return out


def _safe_fetch_chunk(args):
    codes, tx_of = args
    try:
        return _fetch_chunk(codes, tx_of), None
    except Exception as exc:  # noqa: BLE001 单块失败不能带走整批（见下）
        return {}, exc


def fetch_quote_bars(codes, batch_size: int = BATCH_SIZE,
                     workers: int = WORKERS) -> dict[str, pd.DataFrame]:
    """批量抓取当日 bar；返回 {6 位代码: 单行 DataFrame（date 索引）}。

    只返回"看起来有当日 bar"的股票：缺失/停牌/解析失败/分块 HTTP 失败的代码
    都不在结果里，由调用方继续走原来的个股源链——批量路径失败只会退化成
    修复前的行为，不会静默写入错误数据。
    """
    codes = [str(c) for c in codes]
    if not codes:
        return {}
    tx_of = {c: to_quote_code(c) for c in codes}
    chunks = [(codes[i:i + batch_size], tx_of) for i in range(0, len(codes), batch_size)]
    out: dict[str, pd.DataFrame] = {}
    if workers <= 1 or len(chunks) == 1:
        for res, _ in map(_safe_fetch_chunk, chunks):
            out.update(res)
        return out
    with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as ex:
        for res, _ in ex.map(_safe_fetch_chunk, chunks):
            out.update(res)
    return out
