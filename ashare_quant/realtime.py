"""准实时行情快照（秒级延迟，免费源：腾讯/新浪）。

说明：免费接口为快照级行情（延迟数秒），非交易所级 tick 数据；
用于盘中观察与持仓跟踪，不改变月度调仓决策逻辑。
"""

from __future__ import annotations

import pandas as pd


def snapshot(symbols, source: str = "tencent") -> pd.DataFrame:
    """拉取一批股票的快照行情，返回标准表格。

    列：代码/名称/现价/涨跌幅/今开/最高/最低/昨收/成交量。
    """
    import easyquotation

    codes = [str(s).zfill(6) for s in symbols]
    eq = easyquotation.use(source)
    raw = eq.stocks(codes)
    rows = []
    for code, q in raw.items():
        now = q.get("now")
        prev_close = q.get("close")
        if now is None or prev_close in (None, 0):
            continue
        rows.append({
            "代码": code,
            "名称": q.get("name"),
            "现价": float(now),
            "涨跌幅": (float(now) - float(prev_close)) / float(prev_close),
            "今开": q.get("open"),
            "最高": q.get("high"),
            "最低": q.get("low"),
            "昨收": float(prev_close),
            "成交量(手)": q.get("volume"),
        })
    return pd.DataFrame(rows)
