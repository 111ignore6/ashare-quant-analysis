from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cache import ParquetStore
from .calendar import TradingCalendar
from .config import Config
from .pipeline import download_universe
from .universe import load_universe


def _calendar(cfg: Config, store: ParquetStore) -> TradingCalendar:
    if store.symbols():
        dates = sorted({d for s in store.symbols() for d in store.load(s).index})
        return TradingCalendar.from_dates(dates)
    from .fetchers import akshare_fetcher
    df = akshare_fetcher.fetch_index_daily("sh000300")
    return TradingCalendar.from_dates(df.index)


def cmd_fetch(args) -> None:
    cfg = Config.from_yaml(Path(args.config))
    if args.universe:
        cfg.universe_mode = args.universe
    if args.years:
        cfg.years = args.years
    codes = load_universe(cfg.universe_mode)
    store = ParquetStore(cfg.data_root)
    res = download_universe(codes, store, cfg, universe_name=cfg.universe_mode)
    print(f"universe={res['universe']} ok={len(res['ok'])} skipped={len(res['skipped'])} "
          f"failed={len(res['failed'])} no_data={len(res['no_data'])} rows={res['rows']}")
    if res["failed"]:
        print("failed:", ",".join(res["failed"][:20]))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ashare_quant", description="A股量化研究·模拟分析（研究阶段）")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="下载行情到本地缓存")
    f.add_argument("--config", default="config.yaml")
    f.add_argument("--universe", choices=["csi300", "all"])
    f.add_argument("--years", type=int)
    f.set_defaults(func=cmd_fetch)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
