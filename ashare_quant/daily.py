from __future__ import annotations

import pandas as pd

from .cache import ParquetStore
from .config import Config


def last_trading_day(store: ParquetStore, index_symbol: str = "sh000300"):
    idx = store.load(index_symbol)
    if idx is not None and len(idx):
        return idx.index.max()
    return None


def needs_update(store: ParquetStore, index_symbol: str = "sh000300") -> bool:
    return last_trading_day(store, index_symbol) is None


def update_daily(codes: list[str], store: ParquetStore, cfg: Config,
                 index_fetcher=None, fetcher=None, index_symbol: str = "sh000300") -> dict:
    if index_fetcher is None:
        from .fetchers import akshare_fetcher
        index_fetcher = akshare_fetcher.fetch_index_daily
    if fetcher is None:
        from .fetchers import akshare_fetcher
        fetcher = akshare_fetcher.fetch_daily
    manifest = store.read_manifest()
    prev_index_end = manifest.get(index_symbol, {}).get("end")
    idx_df = index_fetcher(index_symbol)
    store.append(index_symbol, idx_df)
    last = idx_df.index.max()
    if prev_index_end and str(last.date()) == prev_index_end:
        return {"new_index_date": str(last.date()), "updated": [], "up_to_date": "all",
                "failed": [], "new_data": False}
    updated, up_to_date, failed = [], [], []
    for code in codes:
        try:
            entry = manifest.get(code, {})
            end_ts = pd.Timestamp(entry["end"]) if entry.get("end") else None
            if end_ts is None:
                old = store.load(code)
                end_ts = old.index.max() if old is not None else None
            if end_ts is None:
                start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=cfg.years)).strftime("%Y%m%d")
            else:
                if end_ts >= last:
                    up_to_date.append(code)
                    continue
                start = (end_ts + pd.Timedelta(days=1)).strftime("%Y%m%d")
            df = fetcher(code, start, str(last).replace("-", ""), cfg.adjust)
            if not df.empty:
                store.append(code, df)
                updated.append(code)
            else:
                up_to_date.append(code)
        except Exception:
            failed.append(code)
    return {"new_index_date": str(last.date()), "updated": sorted(updated),
            "up_to_date": sorted(up_to_date), "failed": sorted(failed), "new_data": True}
