"""准实时行情快照（秒级延迟，免费源：腾讯/新浪）。

说明：免费接口为快照级行情（延迟数秒），非交易所级 tick 数据；
用于盘中观察与持仓跟踪，不改变月度调仓决策逻辑。

⚠️ 为什么不用 easyquotation 抓这两家行情（2026-09-18 实测）：
`easyquotation` 0.7.7 把两家接口都硬编码成**明文 http**
（`tencent.py:17` 的 `http://qt.gtimg.cn/q=`、`sina.py:26` 的
`http://hq.sinajs.cn/...`），而这两家现在对 http 一律返回 **400 空响应**
（同参数 https 为 200 且有数据）。更危险的是该库把空响应当成"字段数不足"
静默 `continue`，于是**不抛异常、返回空 dict** —— 仪表盘只能显示
"可能非交易时段或接口限流"，把"接口已失效"误报成"休市"。
本模块直接走 https，主源失败自动降级到备源，**全部源失败时显式抛
`RealtimeError`**（休市时源仍会返回最后一份快照，所以"空"确实等于"坏了"）。

成交量口径：对外列名是 `成交量(手)`，值就是**手**。
- 腾讯对**科创板（688/689）报价的成交量给的是"股"**，其余板块是"手"，
  故按 ``venues.volume_in_shares`` 归一（否则 688 虚高 100 倍）；
- 新浪成交量为"股"，统一 ÷100；
- 早期经 easyquotation 的实现把"手×100=股"塞进了这一列（名实不符），
  仪表盘再 ×100 拼当日 K 线 bar 时会放大 100 倍，故此处一并纠正。
"""

from __future__ import annotations

import pandas as pd
import requests

from .venues import volume_in_shares

INDEX_CODES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000300": "沪深300",
}

TENCENT_URL = "https://qt.gtimg.cn/q="
SINA_URL = "https://hq.sinajs.cn/list="
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    # 新浪自 2021 起要求 Referer，否则 403
    "Referer": "https://finance.sina.com.cn/",
}
_TIMEOUT = 8
_TENCENT_BATCH = 60   # 腾讯单请求代码数上限（easyquotation 同值）
_SINA_BATCH = 400     # 新浪上限 800，留余量
_FALLBACK = {"tencent": "sina", "sina": "tencent"}

COLUMNS = ["代码", "名称", "现价", "涨跌幅", "今开", "最高", "最低", "昨收", "成交量(手)"]


class RealtimeError(RuntimeError):
    """实时行情源全部不可用（区别于"休市无行情"）。"""


def _prefix(code: str) -> str:
    """代码 → 带市场前缀（sh/sz/bj）；已带前缀的原样返回。

    规则与 easyquotation.helpers.get_stock_type 一致：43/83/87/92 → bj
    （北交所新号段 920xxx 必须走 bj，否则拿不到数据），
    5/6/7/9 开头 → sh，其余 → sz。
    """
    c = str(code).strip().lower()
    if c[:2] in ("sh", "sz", "bj"):
        return c
    c = c.zfill(6)
    if c.startswith(("43", "83", "87", "92")):
        return "bj" + c
    if c.startswith(("5", "6", "7", "9")):
        return "sh" + c
    return "sz" + c


def _num(v) -> float | None:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _at(fields: list[str], idx: int):
    return fields[idx] if idx < len(fields) else None


def _to_hand(volume, code) -> float | None:
    """成交量归一到"手"。

    腾讯对**科创板（688/689）报价的成交量单位是"股"**，其余板块是"手"
    （实测 40/40 vs 90/90，见 ``ashare_quant.venues``）；
    新浪成交量为股、统一 ÷100，两家因此对齐。
    """
    if volume is None:
        return None
    return volume / 100 if volume_in_shares(code) else volume


def _get(url: str) -> str:
    """取行情原文（腾讯/新浪均为 GBK）。非 200 直接抛错，不静默吞掉。"""
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    if resp.status_code != 200:
        raise RealtimeError(f"HTTP {resp.status_code}（{url.split('=')[0]}）")
    return resp.content.decode("gbk", errors="replace")


def _parse_tencent(text: str) -> list[dict]:
    rows = []
    for seg in text.split(";"):
        f = seg.split("~")
        # 字段数不足 = 无此代码/返回空页；腾讯正常个股与指数均为 88 字段
        if len(f) <= 49:
            continue
        code, now, prev = _at(f, 2), _num(_at(f, 3)), _num(_at(f, 4))
        if not code or now is None or prev is None or prev <= 0:
            continue
        rows.append({
            "代码": str(code).strip(),
            "名称": (_at(f, 1) or "").strip(),
            "现价": now,
            "涨跌幅": (now - prev) / prev,
            "今开": _num(_at(f, 5)),
            "最高": _num(_at(f, 33)),
            "最低": _num(_at(f, 34)),
            "昨收": prev,
            # 本表成交量统一用"手"：腾讯对科创板（688/689）给的是"股"
            "成交量(手)": _to_hand(_num(_at(f, 6)), code),
        })
    return rows


def _parse_sina(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if '="' not in line:
            continue
        head, _, payload = line.partition('="')
        f = payload.rstrip('";').split(",")
        if len(f) < 10:
            continue
        code = head.split("_")[-1].strip()[-6:]
        name = f[0].strip()
        open_, prev, now = _num(f[1]), _num(f[2]), _num(f[3])
        high, low, vol_share = _num(f[4]), _num(f[5]), _num(f[8])
        if not code or now is None or prev is None or prev <= 0:
            continue
        rows.append({
            "代码": code,
            "名称": name,
            "现价": now,
            "涨跌幅": (now - prev) / prev,
            "今开": open_,
            "最高": high,
            "最低": low,
            "昨收": prev,
            # 新浪成交量单位是股 → 统一成"手"
            "成交量(手)": None if vol_share is None else vol_share / 100,
        })
    return rows


_FETCHERS = {
    "tencent": (TENCENT_URL, _TENCENT_BATCH, _parse_tencent),
    "sina": (SINA_URL, _SINA_BATCH, _parse_sina),
}


def _fetch(source: str, codes: list[str]) -> list[dict]:
    url, batch, parse = _FETCHERS[source]
    rows: list[dict] = []
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        rows.extend(parse(_get(url + ",".join(chunk))))
    return rows


def _fetch_any(codes: list[str], source: str) -> tuple[list[dict], list[str]]:
    """主源 → 备源依次尝试；返回（行情行, 失败原因）。"""
    errors: list[str] = []
    for src in (source, _FALLBACK.get(source)):
        if not src or src not in _FETCHERS:
            continue
        try:
            rows = _fetch(src, codes)
        except Exception as exc:  # noqa: BLE001 - 汇聚为一次可读的报错
            errors.append(f"{src}: {type(exc).__name__} {exc}")
            continue
        if rows:
            return rows, errors
        errors.append(f"{src}: 返回空（代码不存在或源已失效）")
    return [], errors


def snapshot(symbols, source: str = "tencent") -> pd.DataFrame:
    """拉取一批股票的快照行情，返回标准表格。

    列：代码/名称/现价/涨跌幅/今开/最高/最低/昨收/成交量(手)（成交量为手）。
    主源不可用时自动降级备源；全部失败抛 ``RealtimeError``。
    """
    codes = [_prefix(s) for s in symbols]
    if not codes:
        return pd.DataFrame(columns=COLUMNS)
    rows, errors = _fetch_any(codes, source)
    if not rows:
        raise RealtimeError("实时行情源全部不可用：" + "；".join(errors))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["代码"] = df["代码"].astype(str).str[-6:]
    # 按请求顺序去重（源可能多返/少返）
    df = df.drop_duplicates("代码", keep="first")
    order = {c[-6:]: i for i, c in enumerate(codes)}
    return df.sort_values("代码", key=lambda s: s.map(lambda c: order.get(c, 1e9))) \
             .reset_index(drop=True)


def index_snapshot(source: str = "tencent") -> pd.DataFrame:
    """四大指数实时快照（上证/深成/创业板/沪深300）。"""
    codes = list(INDEX_CODES)
    rows, errors = _fetch_any(codes, source)
    if not rows:
        raise RealtimeError("指数行情源全部不可用：" + "；".join(errors))
    df = pd.DataFrame(rows, columns=COLUMNS)
    lookup = {c[-6:]: c for c in codes}
    df["代码"] = df["代码"].astype(str).str[-6:].map(lambda c: lookup.get(c, c))
    df = df[df["代码"].isin(codes)].drop_duplicates("代码", keep="first")
    df["名称"] = [n or INDEX_CODES.get(c, c)
                  for c, n in zip(df["代码"], df["名称"])]
    return df[["代码", "名称", "现价", "涨跌幅"]].reset_index(drop=True)
