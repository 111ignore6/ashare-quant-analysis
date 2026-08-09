from __future__ import annotations

import pandas as pd


def filter_universe(frame: pd.DataFrame, drop_st: bool = True) -> pd.DataFrame:
    df = frame.copy()
    if drop_st and "name" in df.columns:
        mask = ~df["name"].str.upper().str.contains("ST|退", na=False)
        df = df[mask]
    return df.reset_index(drop=True)


def _csi300() -> list[str]:
    import akshare as ak

    cons = ak.index_stock_cons(symbol="000300")
    codes = cons["品种代码"].astype(str).str.zfill(6).tolist()
    return sorted(set(codes))


def _all_a() -> list[str]:
    import akshare as ak

    raw = ak.stock_info_a_code_name()
    codes = filter_universe(raw)["code"].astype(str).str.zfill(6).tolist()
    return sorted(set(codes))


def load_universe(mode: str = "csi300") -> list[str]:
    if mode == "csi300":
        return _csi300()
    return _all_a()
