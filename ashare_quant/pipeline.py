from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd

from .cache import ParquetStore
from .config import Config


def _fetch_one(code: str, cfg: Config, store: ParquetStore, fetcher) -> str:
    if store.exists(code):
        return "skipped"
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=cfg.years)).strftime("%Y%m%d")
    end = pd.Timestamp.today().normalize().strftime("%Y%m%d")
    for attempt in range(max(1, cfg.retry)):
        try:
            df = fetcher(code, start, end, cfg.adjust)
            if df.empty:
                return "no_data"
            store.append(code, df)
            return "ok"
        except Exception:
            if attempt == max(1, cfg.retry) - 1:
                return "failed"
            time.sleep(1)
    return "failed"


def download_universe(codes: list[str], store: ParquetStore, cfg: Config,
                      fetcher=None, universe_name: str = "csi300",
                      progress_every: int = 500) -> dict:
    if fetcher is None:
        from .fetchers import akshare_fetcher
        fetcher = akshare_fetcher.fetch_daily
    counts = {"ok": [], "failed": [], "skipped": [], "no_data": []}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as ex:
        futures = {ex.submit(_fetch_one, c, cfg, store, fetcher): c for c in codes}
        for fut in as_completed(futures):
            done += 1
            if done % progress_every == 0:
                print(f"progress {done}/{len(codes)}")
            status = fut.result()
            counts.setdefault(status, []).append(futures[fut])
    result = {"universe": universe_name, **counts}
    for key in ("ok", "failed", "skipped", "no_data"):
        result[key] = sorted(result[key])
    result["rows"] = sum(len(store.load(c)) for c in codes if store.exists(c))
    return result


def build_panels(store: ParquetStore, index_symbol: str = "sh000300") -> dict:
    """把缓存拼成 日期×股票 的面板；指数单独作为 Series。"""
    symbols = [s for s in store.symbols() if s != index_symbol]
    frames = {s: store.load(s)["close"] for s in symbols if store.load(s) is not None}
    close = pd.DataFrame(frames).sort_index()
    volume = pd.DataFrame({s: store.load(s)["volume"] for s in symbols if store.load(s) is not None}).sort_index()
    idx = store.load(index_symbol)
    index_close = idx["close"].sort_index() if idx is not None else pd.Series(dtype=float)
    return {"close": close, "volume": volume, "index_close": index_close}
