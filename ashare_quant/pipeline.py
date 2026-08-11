from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time

import pandas as pd

from .cache import ParquetStore
from .config import Config


def _fetch_with_fallback(code: str, cfg: Config, fetcher, fallback=None) -> pd.DataFrame:
    last_err: Exception | None = None
    for f in (fetcher, fallback):
        if f is None:
            continue
        try:
            df = f(code, start_date_for(cfg), end_date_for(cfg), cfg.adjust)
            if not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.DataFrame()


def start_date_for(cfg: Config) -> str:
    return (pd.Timestamp.today().normalize() - pd.DateOffset(years=cfg.years)).strftime("%Y%m%d")


def end_date_for(cfg: Config) -> str:
    return pd.Timestamp.today().normalize().strftime("%Y%m%d")


def _fetch_one(code: str, cfg: Config, store: ParquetStore, fetcher, fallback=None,
               update_manifest: bool = True) -> str:
    if store.exists(code):
        return "skipped"
    for attempt in range(max(1, cfg.retry)):
        try:
            df = _fetch_with_fallback(code, cfg, fetcher, fallback)
            if df.empty:
                return "no_data"
            store.append(code, df, update_manifest=update_manifest)
            return "ok"
        except Exception:
            if attempt == max(1, cfg.retry) - 1:
                return "failed"
            time.sleep(1)
    return "failed"


def download_universe(codes: list[str], store: ParquetStore, cfg: Config,
                      fetcher=None, universe_name: str = "csi300",
                      progress_every: int = 100, fallback_fetcher=None) -> dict:
    if fetcher is None:
        from .fetchers import resolve_fetchers
        fetcher, _, fallback = resolve_fetchers(cfg)
        if fallback_fetcher is not None:
            fallback = fallback_fetcher
    else:
        fallback = fallback_fetcher
    counts = {"ok": [], "failed": [], "skipped": [], "no_data": []}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as ex:
        futures = {ex.submit(_fetch_one, c, cfg, store, fetcher, fallback,
                             False): c for c in codes}
        for fut in as_completed(futures):
            done += 1
            if done % progress_every == 0:
                print(f"progress {done}/{len(codes)}", flush=True)
            status = fut.result()
            counts.setdefault(status, []).append(futures[fut])
    manifest = store.rebuild_manifest()
    result = {"universe": universe_name, **counts}
    for key in ("ok", "failed", "skipped", "no_data"):
        result[key] = sorted(result[key])
    # 用 manifest 的行数统计，避免下载后重读全部缓存文件
    result["rows"] = sum(m.get("rows", 0) for c, m in manifest.items() if c in codes)
    return result


def _manifest_fingerprint(manifest: dict) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _load_panel_cache(store: ParquetStore, index_symbol: str):
    """数据未变化时直接读合并缓存，秒级返回。"""
    cache_dir = store.root / "panels"
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if meta.get("index_symbol") != index_symbol:
        return None
    if meta.get("fingerprint") != _manifest_fingerprint(store.read_manifest()):
        return None
    def _read(name: str) -> pd.DataFrame:
        return pd.read_parquet(cache_dir / f"{name}.parquet")
    try:
        return {
            "close": _read("close"),
            "volume": _read("volume"),
            "open": _read("open"),
            "index_close": _read("index_close")["close"],
        }
    except (FileNotFoundError, ValueError, OSError):
        return None


def _save_panel_cache(store: ParquetStore, index_symbol: str, panels: dict) -> None:
    cache_dir = store.root / "panels"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("close", "volume", "open"):
        panels[name].to_parquet(cache_dir / f"{name}.parquet")
    index_close = panels["index_close"].to_frame("close")
    index_close.to_parquet(cache_dir / "index_close.parquet")
    meta = {
        "index_symbol": index_symbol,
        "fingerprint": _manifest_fingerprint(store.read_manifest()),
    }
    (cache_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def build_panels(store: ParquetStore, index_symbol: str = "sh000300",
                 max_workers: int = 8, use_cache: bool = True) -> dict:
    """把缓存拼成 日期×股票 的面板；指数单独作为 Series。

    每只股票只读一次（并行 IO），同时取出 open/close/volume，
    避免下游再逐只重读缓存文件；数据未变化时直接读合并缓存。
    """
    if use_cache:
        cached = _load_panel_cache(store, index_symbol)
        if cached is not None:
            return cached
    symbols = [s for s in store.symbols() if s != index_symbol]
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        for s, df in zip(symbols, ex.map(store.load, symbols)):
            if df is not None:
                closes[s] = df["close"]
                volumes[s] = df["volume"]
                opens[s] = df["open"]
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).sort_index()
    open_ = pd.DataFrame(opens).sort_index()
    idx = store.load(index_symbol)
    index_close = idx["close"].sort_index() if idx is not None else pd.Series(dtype=float)
    panels = {"close": close, "volume": volume, "open": open_,
              "index_close": index_close}
    if use_cache and len(symbols) >= 10:
        try:
            _save_panel_cache(store, index_symbol, panels)
        except OSError:
            pass
    return panels
