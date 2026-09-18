"""指数"当日 bar"的报价端点兜底源（2026-09-18 新增）。

**为什么需要它（两个交易日的数据就是这么丢的）**：指数的**历史 K 线端点**只有腾讯
能当日给到 —— 新浪（akshare `stock_zh_index_daily`）**滞后一个交易日**，通达信自
09-10 起不可用。而腾讯 K 线的 host（`web.ifzq.gtimg.cn`）会被 WAF 拦：实测
2026-09-18 16:07 返回 **HTTP 501**（15:41 同一 URL 还是 200），于是当天"最新交易日"
推不动，`update_daily` 正确地拒绝在旧日期上出决策，数据停在前一天
（09-17 16:05 与 09-18 16:05 连续两天如此）。

但**报价端点**是另外的 host，实测同一时刻（K 线 501 时）仍返回 **200 且带当日 OHLC**：

    https://qt.gtimg.cn/q=sh000300     → v_sh000300="1~沪深300~000300~4507.39~4460.16~4492.32~189924166~…~20260918160714~…~4523.09~4480.59~4507.39/189924166/537699359227~…"
    https://hq.sinajs.cn/list=sh000300 → var hq_str_sh000300="沪深300,4492.3233,4460.1557,4507.3926,4523.0915,4480.5856,…,189924166,537699359227,…,2026-09-18,15:35:44,00,"

本模块用它们在收盘后补出**当日**那根指数 bar。它是链尾兜底（`resolve_index_fetchers`
末尾追加），历史 K 线源正常时不会被用到。

**单位口径（实测，不是猜的）**：
- 本地指数库 `volume` 是**股**：本地 09-17 = `1.700256e10` ==
  `ak.stock_zh_index_daily("sh000300")` 的 `17002564200`（逐位相同）；
- 两个报价端点的指数成交量字段是**手**（两家数值都是 `189924166`），故 ``× 100``；
- `amount` 两家都给**元**（`537699359227`，也彼此相同）。akshare 指数行没有成交额
  （旧代码填 0）、腾讯 K 线行填的是 `volume × close` 估算值 —— 本模块写真实成交额。

**盘中安全**：当日未收盘时 `daily._call_index` 会用 `calendar.drop_intraday_today`
丢弃当日行，所以本模块盘中不会把未收盘价写进日线（与既有口径一致）。
"""

from __future__ import annotations

import re

import pandas as pd
import requests

TENCENT_URL = "https://qt.gtimg.cn/q="
SINA_URL = "https://hq.sinajs.cn/list="
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    # 新浪自 2021 起要求 Referer，否则 403
    "Referer": "https://finance.sina.com.cn/",
}
_TIMEOUT = 8
_COLS = ["open", "high", "low", "close", "volume", "amount"]
HAND_TO_SHARE = 100      # 报价端点的指数成交量是"手"，本地指数库是"股"
_QUOTED = re.compile(r'="([^"]*)"')


def _get(url: str) -> str:
    """HTTP 取回并按 GBK 解码（两家行情文本都是 GBK）。**测试以此处为接缝。**"""
    resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
    resp.raise_for_status()
    return resp.content.decode("gbk", errors="replace")


def _num(v) -> float | None:
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return f if f == f else None       # 过滤 NaN


def _normalize(symbol: str) -> str:
    s = str(symbol).strip().lower()
    if s[:2] in ("sh", "sz", "bj"):
        return s
    s = s.zfill(6)
    return ("sz" if s.startswith("399") else "sh") + s


def _row(date, open_, high, low, close, volume, amount) -> dict | None:
    if None in (open_, close, volume) or not close:
        return None
    o, c = float(open_), float(close)
    hi = float(high) if high is not None else max(o, c)
    lo = float(low) if low is not None else min(o, c)
    if hi < max(o, c) or lo > min(o, c):        # 端点偶尔给 0/缺列 → 用开收盘兜底
        hi, lo = max(o, c), min(o, c)
    return {"date": pd.Timestamp(date), "open": o, "high": hi, "low": lo, "close": c,
            "volume": float(volume) * HAND_TO_SHARE,
            "amount": float(amount) if amount is not None else 0.0}


def parse_tencent(text: str) -> dict | None:
    """解析 `v_sh000300="1~名称~代码~现价~昨收~今开~量(手)~…~YYYYMMDDHHMMSS~涨跌~涨跌幅~最高~最低~…"`。

    最高/最低按**时间戳字段的相对位置**取（不同板块的填充字段数不一样，写死下标会错位）。
    """
    m = _QUOTED.search(text or "")
    if not m:
        return None
    parts = m.group(1).split("~")
    ts = next((i for i, p in enumerate(parts) if re.fullmatch(r"\d{14}", p.strip())), None)
    if ts is None or len(parts) < ts + 5:
        return None
    amount = None
    comp = parts[ts + 5].split("/") if ts + 5 < len(parts) else []
    if len(comp) == 3:                          # "现价/量(手)/额(元)"
        amount = _num(comp[2])
    if amount is None and ts + 7 < len(parts):  # 兜底：单位是万元
        wan = _num(parts[ts + 7])
        amount = None if wan is None else wan * 1e4
    return _row(parts[ts][:8], _num(parts[5]), _num(parts[ts + 3]), _num(parts[ts + 4]),
                _num(parts[3]), _num(parts[6]), amount)


def parse_sina(text: str) -> dict | None:
    """解析 `var hq_str_sh000300="名称,今开,昨收,现价,最高,最低,…,量(手),额(元),…,YYYY-MM-DD,HH:MM:SS,…"`。"""
    m = _QUOTED.search(text or "")
    if not m:
        return None
    parts = m.group(1).split(",")
    if len(parts) < 10:
        return None
    date = next((p.strip() for p in parts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.strip())), None)
    if date is None:
        return None
    return _row(date, _num(parts[1]), _num(parts[4]), _num(parts[5]),
                _num(parts[3]), _num(parts[8]), _num(parts[9]))


def fetch_index_daily(symbol: str = "sh000300", start: str | None = None) -> pd.DataFrame:
    """指数当日 bar（1 行）；两家报价端点都失败则返回空表（由上层判"源不可用"）。

    `start` 只为兼容 `daily._call_index` 的调用签名，实际不过滤（当日那根必然在 start 之后）。
    """
    sym = _normalize(symbol)
    for _name, url, parser in (("tencent", TENCENT_URL + sym, parse_tencent),
                               ("sina", SINA_URL + sym, parse_sina)):
        try:
            row = parser(_get(url))
        except Exception:  # noqa: BLE001 - 兜底源：失败就换下一家，绝不阻断主链
            row = None
        if row:
            df = pd.DataFrame([{k: v for k, v in row.items() if k != "date"}],
                              index=[row["date"]])
            return df[_COLS].sort_index()
    return pd.DataFrame(columns=_COLS)


def fetch_daily(*_args, **_kwargs):
    """本模块只做指数兜底；被误配成个股源时要响亮地失败，而不是返回空。"""
    raise RuntimeError(
        "index_snapshot 只提供指数兜底（fetch_index_daily），不能作为个股 data_source；"
        "个股请用 akshare / tencent / mootdx")
