from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd
from tqdm import tqdm

from .cache import ParquetStore
from .calendar import market_session
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
                 index_fetcher=None, fetcher=None, fallback_fetcher=None,
                 index_symbol: str = "sh000300") -> dict:
    if index_fetcher is None:
        from .fetchers import resolve_fetchers
        _, index_fetcher, _ = resolve_fetchers(cfg)
    if fetcher is None:
        from .fetchers import resolve_fetchers
        fetcher, _, _ = resolve_fetchers(cfg)
    if fallback_fetcher is None:
        from .fetchers import resolve_fallback_fetchers
        fallback_fetcher = resolve_fallback_fetchers(cfg)
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
                "failed": [], "new_data": False, "stale": len(stale),
                "cooldown_skipped": True}

    def _update_one(code: str) -> tuple[str, str]:
        def _fetch_attempt(f) -> str:
            end_ts = _symbol_end(manifest, store, code)
            if end_ts is None:
                start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=cfg.years)).strftime("%Y%m%d")
            else:
                if end_ts >= last:
                    return "up_to_date"
                start = (end_ts + pd.Timedelta(days=1)).strftime("%Y%m%d")
            # end 用干净日期串（str(Timestamp) 会带 " 00:00:00" 时间部分，部分源容错差）
            df = f(code, start, last.strftime("%Y%m%d"), cfg.adjust)
            if not df.empty:
                store.append(code, df)
                return "updated"
            return "no_data"

        fallbacks = list(fallback_fetcher) if isinstance(fallback_fetcher, (list, tuple)) \
            else ([fallback_fetcher] if fallback_fetcher else [])
        sources = [fetcher] + [f for f in fallbacks if f is not fetcher]
        intraday = market_session() in ("am", "lunch", "pm")
        final = "failed"
        for f in sources:
            is_last_source = f is sources[-1]
            for attempt in range(max(1, cfg.retry)):
                try:
                    status = _fetch_attempt(f)
                except Exception:
                    status = "error"
                    if attempt < max(1, cfg.retry) - 1:
                        time.sleep(0.5)
                        continue
                if status in ("updated", "up_to_date"):
                    return status, code
                if status == "no_data" and not is_last_source:
                    if intraday:
                        # 盘中：备源（新浪/akshare）当日日线尚未生成，等待只会拖慢
                        # 整批更新；直接记为 no_data，收盘后（15:00 后）再走备源。
                        return "no_data", code
                    # 收盘后主源仍返回空：可能是限流或真停牌，交给备源确认；
                    # 真停牌时备源同样返回空，结果仍为 no_data。
                    break
                final = "failed" if status == "error" else status
                break
        return final, code

    updated, up_to_date, failed, no_data = [], [], [], []
    consecutive_errors = 0
    # 非终端环境（重定向/仪表盘后台）下 tqdm 会卡住批量迭代，改用定期打印进度
    use_progress = len(codes_to_update) >= 50 and sys.stderr.isatty()
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as ex:
        mapped = (tqdm(ex.map(_update_one, codes_to_update),
                       total=len(codes_to_update), desc="增量更新", unit="只")
                  if use_progress else ex.map(_update_one, codes_to_update))
        for i, (status, code) in enumerate(mapped):
            if status == "failed":
                # 仅"异常失败"（主备源均报错，如风控/断连）触发退避；
                # no_data（停牌/盘中未生成）是正常结果，不参与，避免整批被拖慢。
                consecutive_errors += 1
                if consecutive_errors >= 25:
                    print("连续 25 只更新失败，疑似行情源风控/断连，"
                          "暂停 60 秒再继续…", flush=True)
                    time.sleep(60)
                    consecutive_errors = 0
            else:
                consecutive_errors = 0
            {"updated": updated, "up_to_date": up_to_date, "failed": failed,
             "no_data": no_data}[status].append(code)
            if not use_progress and (i + 1) % 200 == 0:
                print(f"…增量更新 {i + 1}/{len(codes_to_update)}："
                      f"已更新 {len(updated)}，失败 {len(failed)}，无数据 {len(no_data)}",
                      flush=True)
    if failed or no_data:
        # failed（风控/断连）总是冷却防反复；no_data 只在收盘后写冷却——
        # 盘中/盘前的 no_data 多为"当日日线尚未生成"，冷却会挡住收盘后的正常重试。
        session = market_session()
        failed_cache.update({c: today for c in failed})
        if session not in ("pre", "am", "lunch", "pm"):
            failed_cache.update({c: today for c in no_data})
        if failed_cache:
            _save_failed_cache(store, failed_cache)
    return {"new_index_date": str(last.date()), "updated": sorted(updated),
            "up_to_date": sorted(up_to_date), "failed": sorted(failed),
            "no_data": sorted(no_data), "new_data": True,
            "stale": len(codes_to_update)}
