"""行情数据抓取器（AKShare 主、BaoStock 备）。"""
"""数据源包：内置 tencent / mootdx / akshare / baostock，可注册第三方源。"""

from . import akshare_fetcher, baostock_fetcher, mootdx_fetcher, tencent_fetcher
from .registry import (get_fetch_daily, get_fetch_index_daily, get_source,
                       list_sources, register_source, resolve_fetchers)

register_source("akshare", akshare_fetcher)
register_source("baostock", baostock_fetcher)
register_source("tencent", tencent_fetcher)
register_source("mootdx", mootdx_fetcher)

__all__ = [
    "akshare_fetcher",
    "baostock_fetcher",
    "mootdx_fetcher",
    "tencent_fetcher",
    "get_fetch_daily",
    "get_fetch_index_daily",
    "get_source",
    "list_sources",
    "register_source",
    "resolve_fetchers",
]
