from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
        from .fetchers import resolve_fetchers
        _, index_fetcher, _ = resolve_fetchers(cfg)
    if fetcher is None:
        from .fetchers import resolve_fetchers
        fetcher, _, _ = resolve_fetchers(cfg)
    manifest = store.read_manifest()
    prev_index_end = manifest.get(index_symbol, {}).get("end")
    idx_df = index_fetcher(index_symbol)
    store.append(index_symbol, idx_df)
    last = idx_df.index.max()
    if prev_index_end and str(last.date()) == prev_index_end:
        return {"new_index_date": str(last.date()), "updated": [], "up_to_date": "all",
                "failed": [], "new_data": False}

    def _update_one(code: str) -> tuple[str, str]:
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
                    return "up_to_date", code
                start = (end_ts + pd.Timedelta(days=1)).strftime("%Y%m%d")
            df = fetcher(code, start, str(last).replace("-", ""), cfg.adjust)
            if not df.empty:
                store.append(code, df)
                return "updated", code
            return "up_to_date", code
        except Exception:
            return "failed", code

    updated, up_to_date, failed = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as ex:
        for status, code in ex.map(_update_one, codes):
            {"updated": updated, "up_to_date": up_to_date, "failed": failed}[status].append(code)
    return {"new_index_date": str(last.date()), "updated": sorted(updated),
            "up_to_date": sorted(up_to_date), "failed": sorted(failed), "new_data": True}
