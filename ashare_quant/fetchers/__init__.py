"""行情数据抓取器（AKShare 主、BaoStock 备）。"""
"""数据源包：内置 akshare / baostock，可注册第三方源。"""

from . import akshare_fetcher, baostock_fetcher
from .registry import (get_fetch_daily, get_fetch_index_daily, get_source,
                       list_sources, register_source, resolve_fetchers)

register_source("akshare", akshare_fetcher)
register_source("baostock", baostock_fetcher)

__all__ = [
    "akshare_fetcher",
    "baostock_fetcher",
    "get_fetch_daily",
    "get_fetch_index_daily",
    "get_source",
    "list_sources",
    "register_source",
    "resolve_fetchers",
]
