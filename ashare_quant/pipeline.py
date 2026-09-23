from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time

import pandas as pd

from .cache import ParquetStore
from .calendar import drop_intraday_today
from .config import Config


def _fetch_with_fallback(code: str, cfg: Config, fetcher, fallbacks=None) -> pd.DataFrame:
    last_err: Exception | None = None
    chain = [fetcher] + (list(fallbacks) if fallbacks else [])
    for f in chain:
        try:
            df = f(code, start_date_for(cfg), end_date_for(cfg), cfg.adjust)
            if not df.empty:
                # 盘中：去掉"今天"的未收盘行，避免盘中价当收盘写入缓存
                df = drop_intraday_today(df)
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
        from .fetchers import resolve_fetchers, resolve_fallback_fetchers
        fetcher, _, _ = resolve_fetchers(cfg)
        fallback = resolve_fallback_fetchers(cfg)
    else:
        fallback = fallback_fetcher
    if fallback_fetcher is not None:
        fallback = fallback_fetcher
    if not isinstance(fallback, (list, tuple)):
        fallback = [fallback] if fallback else []
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


# 面板最后一日有效成分股 < 近期常态的该比例 ⇒ 判定为"横截面塌缩"（数据只到一半）
PANEL_MIN_COVERAGE_RATIO = 0.5


def panel_coverage(panels: dict, lookback: int = 10) -> dict:
    """实测面板横截面覆盖率：最后一日有效股票数 vs 近 ``lookback`` 日的中位数。

    为什么必须实测面板（而不是信 ``update_daily`` 的 ``completeness``）：
    2026-09-16 的塌缩是"指数已到 09-16、96% 个股停在 09-15"，而每天的放行判据
    用的是 update_daily **自述**的 completeness(≥0.5)，**从没校验过真正被决策与
    仪表盘消费的面板**；两者在"文件被重写但数据没前进一天"这类场景下会背离
    （项目的工程约定「自述指标必须与实测对撞，不能自证」正是这条）。
    """
    close = panels.get("close")
    empty = {"last_date": None, "last_count": 0, "normal_count": 0,
             "ratio": None, "collapsed": False, "has_norm": False}
    if close is None or close.empty:
        return empty
    counts = close.notna().sum(axis=1)
    last = int(counts.iloc[-1])
    prev = counts.iloc[-(lookback + 1):-1]
    # 历史不足 5 天（新库/小样本）时不判定，避免误伤
    if len(prev) < 5:
        return {**empty, "last_date": str(close.index[-1].date()), "last_count": last}
    normal = int(prev.median())
    ratio = (last / normal) if normal > 0 else None
    # 判定条件（缺一不可，避免误伤）：
    #   1) 有足够历史建立"常态"（≥5 天，见上）；
    #   2) 常态规模 ≥ 20 只 —— 更小的新建库/自选股实验不做判定；
    #   3) 最后一日不足常态的一半，且绝对缺口 ≥ 10 只 —— 既排除小样本抖动，
    #      也排除"少数真停牌/未上市"（那通常只有个位数）。
    collapsed = bool(normal >= 20 and (normal - last) >= 10
                     and last < PANEL_MIN_COVERAGE_RATIO * normal)
    return {"last_date": str(close.index[-1].date()), "last_count": last,
            "normal_count": normal,
            "ratio": None if ratio is None else round(ratio, 4),
            "collapsed": collapsed, "has_norm": True}


def _source_signature(store: ParquetStore) -> str:
    """源数据签名 = manifest 指纹 + 每个个股 parquet 的 (mtime_ns, size)。

    为什么 manifest 指纹不够：manifest 只记 start/end/rows，**值被改写而日期不变**
    时指纹完全相同。2026-09-18 修复科创板 volume 时实测：改了 604 个 parquet 的
    数值、日期一行没变 → manifest 指纹与面板 last_date 都没变 → 面板缓存被判有效
    → `daily --force --retrain` 实际用的仍是被污染的特征（features 指纹一字未变）。
    加 mtime/size 后，任何"文件被重写"（哪怕日期不变）都会让缓存失效。
    代价：5000+ 次 os.stat，实测约 0.6 秒（面板重建是 25 秒量级）。
    """
    h = hashlib.sha256()
    h.update(_manifest_fingerprint(store.read_manifest()).encode())
    for sym in store.symbols():
        try:
            st = (store.root / f"{sym}.parquet").stat()
        except OSError:
            continue
        h.update(f"{sym}:{st.st_mtime_ns}:{st.st_size};".encode())
    return h.hexdigest()


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
    # 指纹只含 manifest 元数据（起止/行数），数据回滚等场景可能指纹相同但
    # 面板内容滞后（08-12 曾复用到 08-11 旧面板导致决策卡住）。直接校验
    # 缓存面板的实际最后日期 == manifest 指数截止，不一致即重建。
    idx_end = store.read_manifest().get(index_symbol, {}).get("end")
    if not idx_end or str(meta.get("last_date", "")) != str(idx_end):
        return None
    # 值被改写而日期不变（手工数据修复）时上面两条都拦不住 → 比对源数据签名。
    # 旧 meta 没有该字段 → 一律视为失效（fail-safe：宁可重建 25 秒，
    # 也不能拿滞后/被污染的面板出决策）。
    if meta.get("source_signature") != _source_signature(store):
        return None
    def _read(name: str) -> pd.DataFrame:
        return pd.read_parquet(cache_dir / f"{name}.parquet")
    try:
        cached = {
            "close": _read("close"),
            "volume": _read("volume"),
            "open": _read("open"),
            "index_close": _read("index_close")["close"],
        }
    except (FileNotFoundError, ValueError, OSError):
        return None
    # 第二道：即便缓存是历史遗留/手工放的，也要确认它的最后一日没有塌缩
    # （写入侧已拒绝塌缩面板，这里防的是"更早版本写下的坏缓存"）。
    coverage = panel_coverage(cached)
    if coverage["collapsed"]:
        return None
    cached["_coverage"] = coverage
    return cached


def _save_panel_cache(store: ParquetStore, index_symbol: str, panels: dict) -> dict:
    """写面板缓存；**塌缩的面板不落盘**（返回 coverage 供调用方判断）。

    塌缩面板一旦落盘，仪表盘（直接读 ``panels/close.parquet``）会拿它显示一整天，
    任何"按日取全市场均值"的下游都会算出假暴跌 —— 所以宁可不缓存（代价是下次
    重建 ~25s），也不把坏面板留给消费方。返回的 ``coverage`` 再交给调用方做门禁。
    """
    coverage = panel_coverage(panels)
    if coverage["collapsed"]:
        return coverage
    cache_dir = store.root / "panels"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("close", "volume", "open"):
        panels[name].to_parquet(cache_dir / f"{name}.parquet")
    index_close = panels["index_close"].to_frame("close")
    index_close.to_parquet(cache_dir / "index_close.parquet")
    meta = {
        "index_symbol": index_symbol,
        "fingerprint": _manifest_fingerprint(store.read_manifest()),
        "last_date": str(panels["close"].index.max().date()),
        "source_signature": _source_signature(store),
        "coverage": coverage,
    }
    (cache_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return coverage


def build_panels(store: ParquetStore, index_symbol: str = "sh000300",
                 max_workers: int = 8, use_cache: bool = True) -> dict:
    """把缓存拼成 日期×股票 的面板；指数单独作为 Series。

    每只股票只读一次（并行 IO），同时取出 open/close/volume，
    避免下游再逐只重读缓存文件；数据未变化时直接读合并缓存。

    返回值里额外带一个私有键 ``_coverage``（``panel_coverage`` 的实测结果），
    调用方据此判断"最后一日横截面是否塌缩"；其余键与原先一致。
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
    coverage = panel_coverage(panels)
    panels["_coverage"] = coverage
    if use_cache and len(symbols) >= 10:
        try:
            _save_panel_cache(store, index_symbol, panels)   # 塌缩面板不会落盘
        except OSError:
            pass
    return panels
