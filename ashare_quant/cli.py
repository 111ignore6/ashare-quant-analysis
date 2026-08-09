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
        "data_through": str(close.index.max().date()),
    }
    results["factor_summary"] = results["factor_report"]["ic_summary"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in results.items() if k in (
        "distribution", "volatility", "momentum", "factor_summary", "pca", "regimes", "data_through")}
    results["raw_json"] = json.dumps(
        {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in serializable.items()},
        ensure_ascii=False, default=str)
    build_report(results, out)
    print(f"研究报告已生成: {out}")


def cmd_select(args) -> None:
    import json

    from .pipeline import build_panels
    from .research.report import screening_to_markdown
    from .screening import run_screening

    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    panels = build_panels(store)
    close, volume = panels["close"], panels["volume"]
    bench = panels["index_close"]
    if bench.empty:
        from .fetchers import akshare_fetcher
        idx_df = akshare_fetcher.fetch_index_daily("sh000300")
        store.save("sh000300", idx_df)
        bench = idx_df["close"]
    out = run_screening(close, volume, bench, top_n=cfg.top_n)
    target = Path(args.out)
    screening_to_markdown(out, target)
    (target.with_suffix(".json")).write_text(
        json.dumps(out.to_dict(orient="records"), ensure_ascii=False, default=str), encoding="utf-8")
    print(f"模型筛选报告已生成: {target}")
    print(out.to_string(index=False))


def cmd_simulate(args) -> None:
    import json

    from .backtest.metrics import metrics_from_returns
    from .models.candidates import LowVolModel, MomentumModel, MultiFactorModel, ReversalModel
    from .pipeline import build_panels
    from .research.report import simulation_to_markdown
    from .simulation import run_simulation

    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    panels = build_panels(store)
    close, volume = panels["close"], panels["volume"]
    open_ = pd.DataFrame({s: store.load(s)["open"] for s in close.columns}).sort_index()
    models = {"momentum": MomentumModel(60), "reversal": ReversalModel(60),
              "lowvol": LowVolModel(60), "multifactor": MultiFactorModel(
                  {"volume_ratio": 0.4, "ma_deviation": 0.2, "reversal60": 0.2, "lowvol": 0.2})}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "adjustments.jsonl"
    sim = run_simulation(models, close, open_, volume, top_n=cfg.top_n, log_path=log_path)
    simulation_to_markdown(sim["summary"], sim["log"].read(), out_dir / "simulation.md")
    (out_dir / "simulation.json").write_text(
        json.dumps({"summary": sim["summary"].to_dict(orient="records"),
                    "rotation_sharpe": float(metrics_from_returns(sim["rotation_returns"])["sharpe"])},
                   ensure_ascii=False, default=str), encoding="utf-8")
    sim["model_returns"].to_csv(out_dir / "model_returns.csv", encoding="utf-8-sig")
    print(sim["summary"].to_string(index=False))
    print(f"模拟盘报告已生成: {out_dir}")


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
    s = sub.add_parser("select", help="运行候选模型筛选")
    s.add_argument("--config", default="config.yaml")
    s.add_argument("--data-root")
    s.add_argument("--out", default="docs/research/model-selection.md")
    s.set_defaults(func=cmd_select)
    sm = sub.add_parser("simulate", help="运行模拟盘与反馈调整")
    sm.add_argument("--config", default="config.yaml")
    sm.add_argument("--data-root")
    sm.add_argument("--out-dir", default="docs/simulation")
    sm.set_defaults(func=cmd_simulate)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
