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
    """按配置解析多级备源链（如 mootdx → tencent → akshare），去重、跳过未知源。"""
    out: list[Callable] = []
    names = list(getattr(cfg, "fallback_sources", ()) or ())
    single = getattr(cfg, "fallback_source", None)
    if single and single not in names:
        names.append(single)
    for name in names:
        try:
            f = get_source(name).fetch_daily
        except KeyError:
            continue
        if f not in out:
            out.append(f)
    return out
