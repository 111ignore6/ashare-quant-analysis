from __future__ import annotations

import json
from pathlib import Path

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


def load_universe_cached(mode: str = "csi300", cache_path=None,
                         max_age_days: int = 7, extra=None) -> list[str]:
    """加载股票池；优先读本地缓存（避免每次联网拉全市场列表）。

    cache_path 指向 JSON 缓存；超过 max_age_days 才重新联网刷新。
    extra 提供额外的本地符号集合（如缓存目录里已下载的股票），
    两者取并集，避免缓存过期期间漏掉已有标的。
    """
    cache_path = Path(cache_path) if cache_path else None
    codes: list[str] = []
    if cache_path is not None and cache_path.exists():
        try:
            meta = json.loads(cache_path.read_text(encoding="utf-8"))
            fetched = pd.Timestamp(meta.get("fetched_at"))
            age_days = (pd.Timestamp.today().normalize() - fetched.normalize()).days
            if meta.get("mode") == mode and age_days <= max_age_days:
                codes = list(meta.get("codes", []))
        except (ValueError, KeyError, OSError):
            codes = []
    if not codes:
        codes = load_universe(mode)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "mode": mode,
                "fetched_at": str(pd.Timestamp.today().normalize().date()),
                "codes": codes,
            }, ensure_ascii=False), encoding="utf-8")
    if extra:
        codes = sorted(set(codes) | {str(c) for c in extra})
    return codes
