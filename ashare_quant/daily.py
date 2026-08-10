from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd
from tqdm import tqdm

from .cache import ParquetStore
from .config import Config


def last_trading_day(store: ParquetStore, index_symbol: str = "sh000300"):
    idx = store.load(index_symbol)
    if idx is not None and len(idx):
        return idx.index.max()
    return None


def needs_update(store: ParquetStore, index_symbol: str = "sh000300") -> bool:
    return last_trading_day(store, index_symbol) is None


def _symbol_end(manifest: dict, store: ParquetStore, code: str):
    """股票的本地数据截止日（manifest 缺失时回退读文件）。"""
    entry = manifest.get(code, {})
    if entry.get("end"):
        return pd.Timestamp(entry["end"])
    old = store.load(code)
    return old.index.max() if old is not None and len(old) else None


def _load_failed_cache(store: ParquetStore) -> dict:
    """读取当日失败冷却表（code -> 失败日期字符串）。"""
    p = store.root / "update_failed.json"
    if not p.exists():
        return {}
    try:
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _save_failed_cache(store: ParquetStore, failed: dict) -> None:
    import json
    p = store.root / "update_failed.json"
    try:
        p.write_text(json.dumps(failed, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def update_daily(codes: list[str], store: ParquetStore, cfg: Config,
                 index_fetcher=None, fetcher=None, index_symbol: str = "sh000300") -> dict:
    fallback_fetcher = None
    if index_fetcher is None:
        from .fetchers import resolve_fetchers
        _, index_fetcher, fallback_fetcher = resolve_fetchers(cfg)
    if fetcher is None:
        from .fetchers import resolve_fetchers
        fetcher, _, fallback_fetcher = resolve_fetchers(cfg)
    manifest = store.read_manifest()
    prev_index_end = manifest.get(index_symbol, {}).get("end")
    idx_start = (pd.Timestamp(prev_index_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
        if prev_index_end else None
    try:
        idx_df = index_fetcher(index_symbol, idx_start)
    except TypeError:
        # 兼容只接受 symbol 的旧签名数据源
        idx_df = index_fetcher(index_symbol)
    if idx_df is not None and not idx_df.empty:
        store.append(index_symbol, idx_df)
        last = idx_df.index.max()
    elif prev_index_end:
        last = pd.Timestamp(prev_index_end)
    else:
        return {"new_index_date": None, "updated": [], "up_to_date": [],
                "failed": sorted(codes), "no_data": [], "new_data": False,
                "stale": 0, "error": "指数数据获取失败"}
    if prev_index_end and str(last.date()) == prev_index_end:
        # 指数无新交易日，但上次更新可能中断：检查股票是否落后，落后则补齐
        stale = [c for c in codes
                 if (end := _symbol_end(manifest, store, c)) is None or end < last]
        if not stale:
            return {"new_index_date": str(last.date()), "updated": [], "up_to_date": "all",
                    "failed": [], "new_data": False, "stale": 0}
        codes_to_update = stale
    else:
        codes_to_update = codes

    # 失败冷却：当天已失败过的股票不再反复重试（停牌/接口异常），次日自动重试
    today = str(pd.Timestamp.today().normalize().date())
    failed_cache = _load_failed_cache(store)
    cooldown = {c for c, d in failed_cache.items() if d == today}
    codes_to_update = [c for c in codes_to_update if c not in cooldown]
    if prev_index_end and str(last.date()) == prev_index_end and not codes_to_update:
        return {"new_index_date": str(last.date()), "updated": [], "up_to_date": "all",
                "failed": [], "new_data": False, "stale": len(stale)}

    def _update_one(code: str) -> tuple[str, str]:
        def _fetch_attempt(f) -> str:
            end_ts = _symbol_end(manifest, store, code)
            if end_ts is None:
                start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=cfg.years)).strftime("%Y%m%d")
            else:
                if end_ts >= last:
                    return "up_to_date"
                start = (end_ts + pd.Timedelta(days=1)).strftime("%Y%m%d")
            df = f(code, start, str(last).replace("-", ""), cfg.adjust)
            if not df.empty:
                store.append(code, df)
                return "updated"
            return "no_data"

        sources = [fetcher] + ([fallback_fetcher] if fallback_fetcher else [])
        for f in sources:
            for attempt in range(max(1, cfg.retry)):
                try:
                    return _fetch_attempt(f), code
                except Exception:
                    if attempt == max(1, cfg.retry) - 1:
                        break
                    time.sleep(0.5)
        return "failed", code

    updated, up_to_date, failed, no_data = [], [], [], []
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as ex:
        for status, code in tqdm(ex.map(_update_one, codes_to_update),
                                 total=len(codes_to_update),
                                 desc="增量更新", unit="只",
                                 disable=len(codes_to_update) < 50):
            {"updated": updated, "up_to_date": up_to_date, "failed": failed,
             "no_data": no_data}[status].append(code)
    if failed or no_data:
        failed_cache.update({c: today for c in failed + no_data})
        _save_failed_cache(store, failed_cache)
    return {"new_index_date": str(last.date()), "updated": sorted(updated),
            "up_to_date": sorted(up_to_date), "failed": sorted(failed),
            "no_data": sorted(no_data), "new_data": True,
            "stale": len(codes_to_update)}
