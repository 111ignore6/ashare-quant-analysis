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
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    codes = load_universe(cfg.universe_mode)
    store = ParquetStore(cfg.data_root)
    res = download_universe(codes, store, cfg, universe_name=cfg.universe_mode)
    print(f"universe={res['universe']} ok={len(res['ok'])} skipped={len(res['skipped'])} "
          f"failed={len(res['failed'])} no_data={len(res['no_data'])} rows={res['rows']}")
    if res["failed"]:
        print("failed:", ",".join(res["failed"][:20]))


def cmd_research(args) -> None:
    import json

    from .pipeline import build_panels
    from .research import momentum, redundancy, regimes, stats
    from .research.factor_stats import factor_report
    from .research.factors import compute_factors, winsorize_zscore
    from .research.report import build_report

    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    panels = build_panels(store)
    close, volume = panels["close"], panels["volume"]
    index_close = panels["index_close"]
    if index_close.empty:
        from .fetchers import akshare_fetcher
        idx_df = akshare_fetcher.fetch_index_daily("sh000300")
        store.save("sh000300", idx_df)
        index_close = idx_df["close"]

    raw_factors = compute_factors(close, volume)
    z_factors = {name: winsorize_zscore(f) for name, f in raw_factors.items()}
    results = {
        "distribution": stats.distribution_stats(close),
        "volatility": stats.volatility_clustering(index_close.pct_change().dropna()),
        "momentum": momentum.horizon_scan(close),
        "factor_report": factor_report(close, volume),
        "pca": redundancy.pca_redundancy(z_factors),
        "regimes": regimes.state_forward_returns(index_close),
    }
    results["factor_summary"] = results["factor_report"]["ic_summary"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in results.items() if k in (
        "distribution", "volatility", "momentum", "factor_summary", "pca", "regimes")}
    results["raw_json"] = json.dumps(
        {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in serializable.items()},
        ensure_ascii=False, default=str)
    build_report(results, out)
    print(f"研究报告已生成: {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ashare_quant", description="A股量化研究·模拟分析（研究阶段）")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="下载行情到本地缓存")
    f.add_argument("--config", default="config.yaml")
    f.add_argument("--universe", choices=["csi300", "all"])
    f.add_argument("--years", type=int)
    f.add_argument("--data-root")
    f.set_defaults(func=cmd_fetch)
    r = sub.add_parser("research", help="运行历史数据研究并生成报告")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--out", default="docs/research/data-research.md")
    r.add_argument("--data-root")
    r.set_defaults(func=cmd_research)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
