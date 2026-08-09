from __future__ import annotations

import socket

import pandas as pd

_COLS = ["open", "high", "low", "close", "volume", "amount"]


def to_baostock_code(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith(("60", "68", "90")):
        return "sh." + s
    if s.startswith(("00", "30", "20")):
        return "sz." + s
    return "bj." + s


def rows_to_frame(fields: list[str], rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=fields)
    for c in _COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[_COLS].sort_index()


def _fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def fetch_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    import baostock as bs

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            to_baostock_code(symbol),
            "date,open,high,low,close,volume,amount",
            start_date=_fmt_date(start),
            end_date=_fmt_date(end),
            frequency="d",
            adjustflag="2" if adjust == "qfq" else "3",
        )
        rows: list[list] = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame(columns=_COLS)
        return rows_to_frame(rs.fields, rows)
    finally:
        try:
            bs.logout()
        finally:
            socket.setdefaulttimeout(old_timeout)
