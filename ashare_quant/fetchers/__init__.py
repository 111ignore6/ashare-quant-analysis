"""数据源包：内置 tencent / mootdx / akshare / baostock，可注册第三方源。"""

from . import (akshare_fetcher, baostock_fetcher, index_snapshot, mootdx_fetcher,
               tencent_fetcher)
from .registry import (get_fetch_daily, get_fetch_index_daily, get_source,
                       list_sources, register_source, resolve_fallback_fetchers,
                       resolve_fetchers, resolve_index_fetchers)

register_source("akshare", akshare_fetcher)
register_source("baostock", baostock_fetcher)
register_source("tencent", tencent_fetcher)
register_source("mootdx", mootdx_fetcher)
# 只提供 fetch_index_daily 的兜底源（指数当日 bar），不是个股源，见模块说明
register_source("index_snapshot", index_snapshot)

__all__ = [
    "akshare_fetcher",
    "baostock_fetcher",
    "index_snapshot",
    "mootdx_fetcher",
    "tencent_fetcher",
    "get_fetch_daily",
    "get_fetch_index_daily",
    "get_source",
    "list_sources",
    "register_source",
    "resolve_fallback_fetchers",
    "resolve_fetchers",
    "resolve_index_fetchers",
]
