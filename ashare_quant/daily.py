from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd
from tqdm import tqdm

from .cache import ParquetStore
from .calendar import drop_intraday_today, market_session
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


def _append_merged(store: ParquetStore, code: str, df: pd.DataFrame) -> pd.DataFrame:
    """追加并返回合并后的整表（批量路径据此一次性写回 manifest）。"""
    old = store.load(code)
    merged = df if old is None else pd.concat([old, df])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    store.save(code, merged, update_manifest=False)
    return merged


def _update_manifest_many(store: ParquetStore, frames: dict) -> None:
    """一次性写回多条 manifest 条目。

    逐只走 `ParquetStore.update_manifest` 要 read+write 整个 manifest.json
    （实测 8.5ms/只，全市场约 45 秒，且被锁串行化）；批量路径先落 parquet、
    最后合并写一次。格式与 cache.py 保持一致，避免无谓 diff。
    """
    import json
    m = store.read_manifest()
    for code, df in frames.items():
        m[code] = {"start": str(df.index.min().date()),
                   "end": str(df.index.max().date()),
                   "rows": int(len(df))}
    try:
        store.manifest_path.write_text(
            json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _resolve_batch_fetcher(fetcher, fallback_fetcher):
    """批量报价快路径的取数函数；腾讯不在解析出的源链里时返回 None。

    用"解析后的可调用对象"判断而不是读配置字符串：测试注入假源时不会联网，
    用户从 config.yaml 去掉 tencent 后批量路径也随之关闭。
    """
    from .fetchers import tencent_fetcher
    from .fetchers.tencent_quote import fetch_quote_bars

    chain = [fetcher]
    if isinstance(fallback_fetcher, (list, tuple)):
        chain += list(fallback_fetcher)
    elif fallback_fetcher is not None:
        chain.append(fallback_fetcher)
    return fetch_quote_bars if tencent_fetcher.fetch_daily in chain else None


def _fill_latest_bar_via_quotes(codes: list[str], store: ParquetStore, manifest: dict,
                                last, prev_trading_day, fetch_bars, workers: int,
                                updated: list) -> list[str]:
    """用批量报价补齐"只缺最新一根 bar"的股票，返回仍需逐只抓取的代码。

    只对 `本地末行 == 上一个交易日` 的股票生效：报价里只有当日一根，若该股缺
    更多天，单补最后一根会在中间留洞（面板/特征按交易日对齐，会错），必须走
    个股源链。报价日期 != 目标交易日（盘中、源未发布、指数滞后）时整批放弃。
    批量取数或写盘失败时原样返回，退化成修复前的逐只路径。
    """
    eligible = [c for c in codes if _symbol_end(manifest, store, c) == prev_trading_day]
    if not eligible:
        return codes
    try:
        bars = fetch_bars(eligible)
    except Exception:  # noqa: BLE001 批量路径失败只退化为逐只路径
        return codes
    hits = {c: df for c, df in (bars or {}).items()
            if df is not None and not df.empty and df.index.max() == last}
    if not hits:
        return codes
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            frames = dict(ex.map(
                lambda kv: (kv[0], _append_merged(store, kv[0], kv[1])), hits.items()))
        _update_manifest_many(store, frames)
    except Exception:  # noqa: BLE001 落盘失败：交给逐只路径重写（幂等）
        return codes
    updated.extend(sorted(hits))
    return [c for c in codes if c not in hits]


def _stock_health(codes: list[str], store: ParquetStore, manifest: dict, target,
                  expected: int = 0) -> dict:
    """个股口径健康度：更新后仍未走到 `target` 的股票数（供 update_stats/仪表盘）。

    `expected` = 本来就不可能到达 target 的只数（真停牌 no_data、今日冷却中的
    代码）——它们不算故障，否则每天的停牌股都会变成假警报。

    **每个出口都要带上这组字段**：只写"正常路径"的出口会让"当天第二次运行/
    收盘后补跑"退回只看得懂指数的旧口径，而那正是 2026-09-16 说谎的 healthy。
    """
    total = len(codes)
    behind = [c for c in codes
              if (e := _symbol_end(manifest, store, c)) is None or e < target]
    return {"stocks_total": total, "stocks_behind": len(behind),
            "stocks_behind_expected": expected,
            "stocks_behind_unexpected": max(0, len(behind) - expected),
            "completeness": round((total - len(behind)) / total, 4) if total else 1.0}


def _call_index(fetch, symbol: str, start):
    """兼容 (symbol, start) 与 (symbol) 两种签名调指数源；返回去盘中后的非空表或 None。

    单源异常/空返回都收敛为 None，由上层走下一级备源——不能让一个挂掉的源
    把整批更新带走，也不能让它伪装成"没有新交易日"。
    """
    try:
        try:
            df = fetch(symbol, start)
        except TypeError:
            df = fetch(symbol)  # 兼容只接受 symbol 的旧签名数据源
    except Exception:  # noqa: BLE001 源故障与"无新数据"必须区分，见 _fetch_index
        return None
    if df is None or df.empty:
        return None
    # 有的源（akshare/通达信）不实现 start 过滤，会直接给全量历史：不裁剪就会
    # 把指数缓存从 3 年撑成 20 年，改变下游交易日历/面板的口径。
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
        if df.empty:
            return None
    # 盘中：当日 bar 未收盘确认，不推进"最新交易日"（否则盘中价被当
    # 成收盘写入，污染日线/决策/账户；等收盘后 16:05 正式更新）
    df = drop_intraday_today(df)
    return None if df is None or df.empty else df


def _fetch_index(cfg: Config, index_symbol: str, index_fetcher, probe_start: str | None):
    """沿"主源 → 各级备源"抓指数，返回 (合并表, 生效源列表, 主源是否空返回)。

    所有源各拉一次并取最靠后的一天：备源对当日 bar 的发布时点不同
    （新浪指数滞后一天、腾讯当天即有），只用第一个非空结果会整体滞后。
    起点用"上一个交易日"而不是"次日"：存活的源至少会返回上一交易日那根
    bar，因此"全部空返回"只可能是源不可用，而不是"今天没有新交易日"。
    """
    if index_fetcher is not None:
        chain: list[tuple[str, object]] = [("custom", index_fetcher)]
    else:
        from .fetchers import resolve_index_fetchers
        chain = resolve_index_fetchers(cfg)
    frames: list[tuple[str, pd.DataFrame]] = []
    for name, fn in chain:
        df = _call_index(fn, index_symbol, probe_start)
        if df is not None:
            frames.append((name, df))
    if not frames:
        return None, [n for n, _ in chain], bool(chain)
    frames.sort(key=lambda kv: kv[1].index.max(), reverse=True)
    merged = pd.concat([df for _, df in frames])
    # 日期重叠时取"能提供最新一天"的那个源，避免同一行两个口径
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    primary_name = chain[0][0]
    primary_empty = all(n != primary_name for n, _ in frames)
    return merged, [n for n, _ in frames], primary_empty


def update_daily(codes: list[str], store: ParquetStore, cfg: Config,
                 index_fetcher=None, fetcher=None, fallback_fetcher=None,
                 index_symbol: str = "sh000300", batch_fetcher=None) -> dict:
    """每日增量更新；batch_fetcher 传 None 时按源链自动决定是否用批量报价。"""
    if fetcher is None:
        from .fetchers import resolve_fetchers
        fetcher, _, _ = resolve_fetchers(cfg)
    if fallback_fetcher is None:
        from .fetchers import resolve_fallback_fetchers
        fallback_fetcher = resolve_fallback_fetchers(cfg)
    if batch_fetcher is None:
        batch_fetcher = _resolve_batch_fetcher(fetcher, fallback_fetcher)
    manifest = store.read_manifest()
    prev_index_end = manifest.get(index_symbol, {}).get("end")
    # 探测起点：上一个交易日（无 manifest 时由源自行给全量）
    probe_start = prev_index_end
    idx_df, index_sources, primary_index_empty = _fetch_index(
        cfg, index_symbol, index_fetcher, probe_start)
    if idx_df is None:
        # 主备源全部无返回：无法判定最新交易日，本次不更新任何行情。
        # 以前这里会静默沿用旧日期并返回 up_to_date="all"，于是"源挂了"
        # 连续多天被记成"已最新"（09-10、09-11 mootdx 公开服务器全挂）。
        # 个股口径按"上一个已知指数日"报（没有它就只能报总数）。
        stock = _stock_health(codes, store, manifest, pd.Timestamp(prev_index_end)) \
            if prev_index_end else {"stocks_total": len(codes)}
        return {"new_index_date": prev_index_end, "updated": [], "up_to_date": [],
                "failed": [], "no_data": [], "new_data": False, "stale": 0,
                "index_status": "all_sources_empty", "index_sources": index_sources,
                "primary_index_empty": primary_index_empty, **stock,
                "error": (f"指数数据源全部无返回（尝试：{'、'.join(index_sources) or '无可用源'}），"
                          f"数据仍停在 {prev_index_end}；请检查网络或在 config.yaml 更换 "
                          f"data_source / fallback_sources")}
    store.append(index_symbol, idx_df)
    last = idx_df.index.max()
    index_status = "unchanged" if prev_index_end and str(last.date()) == prev_index_end \
        else "updated"
    base = {"index_status": index_status, "index_sources": index_sources,
            "primary_index_empty": primary_index_empty}
    if prev_index_end and str(last.date()) == prev_index_end:
        # 指数无新交易日（且至少一个源存活），但上次更新可能中断：检查股票是否落后
        stale = [c for c in codes
                 if (end := _symbol_end(manifest, store, c)) is None or end < last]
        if not stale:
            return {**base, "new_index_date": str(last.date()), "updated": [],
                    "up_to_date": "all", "failed": [], "new_data": False, "stale": 0,
                    **_stock_health(codes, store, manifest, last)}
        codes_to_update = stale
    else:
        codes_to_update = codes

    # 失败冷却：当天已失败过的股票不再反复重试（停牌/接口异常），次日自动重试
    today = str(pd.Timestamp.today().normalize().date())
    failed_cache = _load_failed_cache(store)
    cooldown = {c for c, d in failed_cache.items() if d == today}
    codes_to_update = [c for c in codes_to_update if c not in cooldown]
    if prev_index_end and str(last.date()) == prev_index_end and not codes_to_update:
        # 落后的股票全在今日冷却里（多为真停牌/当日源未发布）→ 预期内，不算故障
        return {**base, "new_index_date": str(last.date()), "updated": [],
                "up_to_date": "all", "failed": [], "new_data": False,
                "stale": len(stale), "cooldown_skipped": True,
                **_stock_health(codes, store, manifest, last, expected=len(stale))}

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
            if df is None or df.empty:
                return "no_data"
            store.append(code, df)
            # 只有拿到"目标交易日"的 bar 才算更新成功。新浪个股当日 bar 要滞后
            # 数小时：16:05 跑时它只返回 D-1，旧代码"非空即 updated"，于是整批
            # 股票被记为已更新却没有前进一天，也永远不会走到备源（腾讯当日即有）
            # ——2026-09-16 实测 5140/5360 只停在 D-1、updated 却报 4989。
            # 拿到的早期 bar 仍然保留（是真实数据前进），只是状态交给备源继续确认。
            return "updated" if df.index.max() >= last else "no_data"

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

    updated: list[str] = []
    stale_count = len(codes_to_update)
    # —— 批量快路径：只缺"最新一根 bar"的股票用腾讯批量报价一次补齐 ——
    # 逐只源链每只要 1~3 次 HTTP（akshare 1.26s/只），而批量报价一次可带 120 只、
    # 实测 310~340 只/s；且腾讯收盘后当日 bar 立即就有，正是新浪缺的那根。
    idx_hist = store.load(index_symbol)
    prev_trading_day = None
    if idx_hist is not None and len(idx_hist):
        prior = idx_hist.index[idx_hist.index < last]
        if len(prior):
            prev_trading_day = prior.max()
    if batch_fetcher is not None and codes_to_update and prev_trading_day is not None:
        codes_to_update = _fill_latest_bar_via_quotes(
            codes_to_update, store, manifest, last, prev_trading_day,
            batch_fetcher, cfg.max_workers, updated)

    up_to_date, failed, no_data = [], [], []
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
    # —— 更新后复核：到底有多少股票真的走到了目标交易日 ——
    # updated 只是"文件被写过"（旧代码连这个都不保证）；这里直读 manifest 给出
    # 可对外报数的口径，供 cli/dashboard 判断"数据完整性"与是否该出决策。
    fresh = store.read_manifest()
    target = str(last.date())
    # 真停牌 / 当日无 bar（no_data）与今日冷却中的代码本来就到不了目标日，
    # 不算故障——否则每天的停牌股都会变成假警报（AGENTS.md 的"噪声当故障"教训）。
    stock = _stock_health(codes, store, fresh, last,
                          expected=len(no_data) + len(cooldown))
    # 指数前进了、但个股几乎没拿到当日 bar（主备源都未发布/全挂）：不能照旧
    # 出决策——09-16 就是这样在"指数 09-16 + 96% 个股 09-15"的塌缩面板上选的股。
    new_data = bool(updated) and stock["completeness"] >= 0.5
    return {**base, "new_index_date": target, "updated": sorted(updated),
            "up_to_date": sorted(up_to_date), "failed": sorted(failed),
            "no_data": sorted(no_data), "new_data": new_data,
            "stale": stale_count, **stock}
