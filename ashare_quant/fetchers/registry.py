"""数据源注册表：新增数据源只需实现 fetch_daily / fetch_index_daily 并注册。"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Any] = {}


def register_source(name: str, module) -> None:
    """注册一个数据源模块（需含 fetch_daily；指数可选 fetch_index_daily）。"""
    if name in _REGISTRY:
        raise ValueError(f"source already registered: {name}")
    _REGISTRY[name] = module


def get_source(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown data source '{name}', available: {list_sources()}")
    return _REGISTRY[name]


def list_sources() -> list[str]:
    return list(_REGISTRY)


def get_fetch_daily(name: str) -> Callable:
    return get_source(name).fetch_daily


def get_fetch_index_daily(name: str) -> Callable | None:
    return getattr(get_source(name), "fetch_index_daily", None)


def _source_names(cfg) -> list[str]:
    """主源 + 多级备源的有序源名（去重，主源在最前）。"""
    names: list[str] = [getattr(cfg, "data_source", None)]
    names += list(getattr(cfg, "fallback_sources", ()) or ())
    single = getattr(cfg, "fallback_source", None)
    if single:
        names.append(single)
    out: list[str] = []
    for n in names:
        if n and n not in out:
            out.append(n)
    return out


def resolve_index_fetchers(cfg) -> list[tuple[str, Callable]]:
    """指数抓取链 [(源名, fetch_index_daily)]，主源在前。

    指数是"最新交易日"的唯一依据，单源一旦拉不到数据就会把整批更新静默
    停在旧日期（09-10/09-11 mootdx 公开服务器全挂，连续两天报"已最新=all"），
    因此指数必须和个股一样有多级备源。

    **末尾固定追加"报价端点"兜底源 `index_snapshot`（2026-09-18 新增）**：
    能当日给到指数的**历史 K 线**端点只有腾讯，而它会返回 HTTP 501（WAF，实测
    09-18 16:07），新浪指数又天然滞后一个交易日 —— 缺了这一级，"今天"就永远推不动
    （09-17 与 09-18 连续两天的 16:05 任务都是这个原因空转）。报价端点
    （qt.gtimg.cn / hq.sinajs.cn）是另外的 host，同一时刻仍返回当日 OHLC。
    它排在最后：历史 K 线源正常时按日期合并会优先取前者，兜底源只在需要时生效。
    """
    out: list[tuple[str, Callable]] = []
    for name in _source_names(cfg):
        try:
            fn = get_fetch_index_daily(name)
        except KeyError:
            continue
        if fn is not None:
            out.append((name, fn))
    if not any(n == "index_snapshot" for n, _ in out):
        from . import index_snapshot
        out.append(("index_snapshot", index_snapshot.fetch_index_daily))
    return out


def resolve_fetchers(cfg) -> tuple[Callable, Callable | None, Callable | None]:
    """按配置解析 (fetch_daily, fetch_index_daily, fallback_fetch_daily)。"""
    primary = get_source(cfg.data_source)
    fallback = None
    if getattr(cfg, "fallback_source", None):
        try:
            fallback = get_source(cfg.fallback_source).fetch_daily
        except KeyError:
            fallback = None
    return (primary.fetch_daily,
            getattr(primary, "fetch_index_daily", None),
            fallback)


def resolve_fallback_fetchers(cfg) -> list[Callable]:
    """按配置解析多级备源链（如 akshare → tencent → mootdx），去重、跳过未知源与主源。"""
    primary = getattr(cfg, "data_source", None)
    out: list[Callable] = []
    for name in _source_names(cfg):
        if name == primary:
            continue
        try:
            f = get_source(name).fetch_daily
        except KeyError:
            continue
        if f not in out:
            out.append(f)
    return out
