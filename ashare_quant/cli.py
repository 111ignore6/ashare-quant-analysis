from __future__ import annotations

import argparse
import sys
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
    from .fetchers import resolve_fetchers
    _, idx_fetch, _ = resolve_fetchers(cfg)
    df = idx_fetch("sh000300")
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
    print(f"开始下载 {len(codes)} 只（数据目录 {cfg.data_root}），约需 "
          f"{len(codes) // max(1, cfg.max_workers) * 1 // 60 + 1} 分钟，请耐心等待…",
          flush=True)
    res = download_universe(codes, store, cfg, universe_name=cfg.universe_mode)
    print(f"下载完成：成功 {len(res['ok'])}，已存在 {len(res['skipped'])}，"
          f"失败 {len(res['failed'])}，无数据 {len(res['no_data'])}，"
          f"共 {res['rows']} 行", flush=True)
    if res["failed"]:
        print("失败股票：", ",".join(res["failed"][:20]), flush=True)
        if len(res["failed"]) > 20:
            print(f"……共 {len(res['failed'])} 只失败，可重跑 fetch 命令重试", flush=True)


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
        from .fetchers import resolve_fetchers
        _, idx_fetch, _ = resolve_fetchers(cfg)
        idx_df = idx_fetch("sh000300")
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
        from .fetchers import resolve_fetchers
        _, idx_fetch, _ = resolve_fetchers(cfg)
        idx_df = idx_fetch("sh000300")
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
    index_close = panels["index_close"]
    open_ = panels["open"]
    models = {"momentum": MomentumModel(60), "reversal": ReversalModel(60),
              "lowvol": LowVolModel(60), "multifactor": MultiFactorModel(
                  {"volume_ratio": 0.4, "ma_deviation": 0.2, "reversal60": 0.2, "lowvol": 0.2})}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "adjustments.jsonl"
    sim = run_simulation(models, close, open_, volume, top_n=cfg.top_n, log_path=log_path,
                         stop_loss=cfg.stop_loss, take_profit=cfg.take_profit)
    simulation_to_markdown(sim["summary"], sim["log"].read(), out_dir / "simulation.md")
    (out_dir / "simulation.json").write_text(
        json.dumps({"summary": sim["summary"].to_dict(orient="records"),
                    "rotation_sharpe": float(metrics_from_returns(sim["rotation_returns"])["sharpe"])},
                   ensure_ascii=False, default=str), encoding="utf-8")
    _simulation_full_returns(close, index_close, sim).to_csv(
        out_dir / "model_returns.csv", encoding="utf-8-sig")
    print(sim["summary"].to_string(index=False))
    print(f"模拟盘报告已生成: {out_dir}")


def _build_html_report(cfg, store, out_dir, panels=None) -> None:
    from .models.candidates import LowVolModel, MomentumModel, MultiFactorModel, ReversalModel
    from .pipeline import build_panels
    from .research.factor_stats import factor_report
    from .report.html_report import build_html_report, drawdown_figure, equity_figure, factor_heatmap
    from .simulation import run_simulation

    if panels is None:
        panels = build_panels(store)
    close, volume = panels["close"], panels["volume"]
    index_close = panels["index_close"]
    open_ = panels["open"]
    models = {"momentum": MomentumModel(60), "reversal": ReversalModel(60),
              "lowvol": LowVolModel(60), "multifactor": MultiFactorModel(
                  {"volume_ratio": 0.4, "ma_deviation": 0.2, "reversal60": 0.2, "lowvol": 0.2})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = run_simulation(models, close, open_, volume, top_n=cfg.top_n,
                         log_path=out_dir / "adjustments.jsonl",
                         stop_loss=cfg.stop_loss, take_profit=cfg.take_profit)
    all_returns = _simulation_full_returns(close, index_close, sim)
    all_returns.to_csv(out_dir / "model_returns.csv", encoding="utf-8-sig")
    summary = factor_report(close, volume)["ic_summary"]
    build_html_report(equity_figure(all_returns), drawdown_figure(all_returns),
                      factor_heatmap(summary), sim["log"].read(),
                      data_through=str(close.index.max().date()), path=out_dir / "report.html")


def _simulation_full_returns(close: pd.DataFrame, index_close: pd.Series,
                             sim: dict) -> pd.DataFrame:
    """模拟盘全收益表：4 个手工模型 + 轮动 + 两条真实基准（等权全市场、沪深300）。"""
    from .backtest.simple import monthly_rebalance_dates

    out = sim["model_returns"].copy()
    out["rotation"] = sim["rotation_returns"]
    rdates = monthly_rebalance_dates(close.index)
    bench_eq = close.loc[rdates].pct_change(fill_method=None).mean(axis=1).dropna()
    bench_idx = index_close.reindex(rdates).pct_change(fill_method=None).dropna()
    out["基准·等权全市场"] = bench_eq.reindex(out.index)
    out["基准·沪深300"] = bench_idx.reindex(out.index)
    return out


def cmd_daily(args) -> None:
    import json
    import time

    from .daily import update_daily
    from .pipeline import build_panels
    from .universe import load_universe_cached

    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    local_symbols = [s for s in store.symbols() if s != "sh000300"]
    codes = load_universe_cached(
        cfg.universe_mode, cache_path=cfg.data_root / "universe.json",
        extra=local_symbols)

    def _write_stats(t0: float, t1: float | None, t2: float | None,
                     t3: float | None, extra: dict | None = None) -> None:
        stats = {
            "last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": cfg.data_source,
            "index_date": out.get("new_index_date"),
            "updated": len(out.get("updated", [])),
            "up_to_date": (len(out["up_to_date"]) if isinstance(out.get("up_to_date"), list)
                           else out.get("up_to_date", 0)),
            "failed": len(out.get("failed", [])),
            "no_data": len(out.get("no_data", [])),
            "phase1_sec": round(t1 - t0, 1) if t1 is not None else None,
            "phase2_sec": round(t2 - t1, 1) if t1 is not None and t2 is not None else None,
            "phase3_sec": round(t3 - t2, 1) if t2 is not None and t3 is not None else None,
            "total_sec": round((t3 or t1 or t0) - t0, 1),
            **(extra or {}),
        }
        try:
            (cfg.data_root / "update_stats.json").write_text(
                json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    t0 = time.time()
    print("阶段 1/3：增量更新行情数据（有进度条，首次/大涨后约 1-5 分钟）…", flush=True)
    out = update_daily(codes, store, cfg)
    t1 = time.time()
    n_up = len(out["up_to_date"]) if isinstance(out["up_to_date"], list) else "all"
    print(f"指数截止={out['new_index_date']} 更新={len(out['updated'])} "
          f"已最新={n_up} 失败={len(out['failed'])}", flush=True)
    if out.get("stale"):
        print(f"检测到 {out['stale']} 只股票数据落后（上次更新可能中断），"
              f"已补齐 {len(out['updated'])} 只", flush=True)
    if out["failed"]:
        print("更新失败：", ",".join(out["failed"][:20]), flush=True)
        if len(out["failed"]) > 20:
            print(f"……共 {len(out['failed'])} 只失败，可重跑 daily 重试", flush=True)
    if out.get("no_data"):
        print(f"无数据（可能停牌/未上市）：{len(out['no_data'])} 只，"
              f"如 {','.join(out['no_data'][:5])}…，已跳过今日", flush=True)
    if not out.get("new_data", True) and not args.force:
        print("数据已是最新交易日，跳过报告与决策重算（--force 可强制重算）", flush=True)
        _write_stats(t0, t1, None, None)
        return
    panels = build_panels(store)
    t2 = time.time()
    print("阶段 2/3：生成报告与模拟盘（约 20 秒）…", flush=True)
    _build_html_report(cfg, store, args.out_dir, panels=panels)
    t3 = time.time()
    if not args.no_decision:
        print("阶段 3/3：训练/加载模型并生成今日决策…", flush=True)
        payload, account = _save_decision(cfg, store, args.model_dir, args.sample_size,
                                          args.retrain, args.out_dir, panels=panels)
        from .report.daily_report import build_daily_report
        report_path = build_daily_report(
            cfg, out, payload, account, store.read_manifest(),
            Path(args.out_dir) / "daily_report.md")
        print(f"每日决策日报已生成: {report_path}", flush=True)
    t4 = time.time()
    print(f"当日报告已生成: {args.out_dir}/report.html")
    _write_stats(t0, t1, t2, t4,
                 extra={"report_sec": round(t3 - t2, 1),
                        "decision_sec": round(t4 - t3, 1) if not args.no_decision else None})


def cmd_report(args) -> None:
    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    _build_html_report(cfg, store, args.out_dir)
    print(f"报告已生成: {args.out_dir}/report.html")


def cmd_benchmark(args) -> None:
    from .ml.benchmark import run_benchmark, write_report
    from .pipeline import build_panels

    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    panels = build_panels(store)
    close, volume = panels["close"], panels["volume"]
    index_close = panels["index_close"]
    if index_close.empty:
        from .fetchers import resolve_fetchers
        _, idx_fetch, _ = resolve_fetchers(cfg)
        idx_df = idx_fetch("sh000300")
        store.save("sh000300", idx_df)
        index_close = idx_df["close"]
    table, series = run_benchmark(close, volume, index_close,
                                  top_n=cfg.top_n, with_dl=args.with_dl)
    out = Path(args.out)
    write_report(table, out, out.with_suffix(".json"))
    returns_path = out.with_name(out.stem + ".returns.csv")
    pd.DataFrame(series).to_csv(returns_path, encoding="utf-8-sig")
    print(f"算法收益序列已保存: {returns_path}")
    print(table.to_string(index=False))
    print(f"算法对比报告已生成: {out}")


def _save_decision(cfg, store, model_dir, sample_size: int, retrain: bool,
                   out_dir, panels=None) -> None:
    import json

    from .ml.decision import decide, load_models, train_and_save
    from .ml.features import build_dataset, load_feature_cache, save_feature_cache
    from .pipeline import build_panels

    if panels is None:
        panels = build_panels(store)
    close, volume = panels["close"], panels["volume"]
    index_close = panels["index_close"]
    if index_close.empty:
        from .fetchers import resolve_fetchers
        _, idx_fetch, _ = resolve_fetchers(cfg)
        idx_df = idx_fetch("sh000300")
        store.save("sh000300", idx_df)
        index_close = idx_df["close"]
    cache_path = Path(cfg.data_root) / "features.parquet"
    cached = load_feature_cache(cache_path, close.index.max())
    if cached is not None:
        X_all, y_all = cached
    else:
        X_all, y_all = build_dataset(close, volume, index_close, horizon=20, require_target=False)
        save_feature_cache(X_all, y_all, cache_path, close.index.max())
    ok = y_all.notna()
    X, y = X_all[ok], y_all[ok]
    model_dir = Path(model_dir)
    nested = model_dir / cfg.universe_mode
    if (nested / "meta.json").exists():
        # universe 专属目录优先（避免与旧根目录模型混淆）
        model_dir = nested
    meta_path = model_dir / "meta.json"
    need_retrain = retrain or not meta_path.exists()
    if not need_retrain:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        trained = pd.Timestamp(meta.get("trained_on"))
        if (pd.Timestamp.today().normalize() - trained).days > 30:
            need_retrain = True
    if need_retrain:
        train_and_save(X, y, model_dir, sample_size=sample_size,
                       as_of=close.index.max())
    loaded = load_models(model_dir)
    last_date = X_all.index.get_level_values("date").max()
    picks = decide(loaded, X_all, close, last_date, top_n=cfg.top_n)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": str(last_date.date()),
        "top_n": int(len(picks)),
        "initial_capital": float(cfg.initial_capital),
        "models": loaded["meta"]["models"],
        "thresholds": loaded["meta"]["thresholds"],
        "picks": picks.to_dict(orient="records"),
        "disclaimer": "模拟研究，不构成投资建议",
    }
    (out_dir / "decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    picks.to_csv(out_dir / "decision.csv", index=False, encoding="utf-8-sig")
    # 决策历史追踪：回填历史调仓日 + 记录当日 + 更新账户净值曲线
    from .portfolio import update_portfolio
    summary = update_portfolio(close, cfg, cfg.data_root / "portfolio",
                               decision=payload)
    print(f"账户曲线已更新：决策 {summary['decisions']} 次，"
          f"总收益率 {summary['total_return']:+.2%}", flush=True)
    print(f"决策日期: {last_date.date()}  持仓 {len(picks)} 只")
    print(picks.head(20).to_string(index=False))
    print(f"决策已保存: {out_dir}/decision.json")
    return payload, summary


def cmd_decision(args) -> None:
    from .pipeline import build_panels

    cfg = Config.from_yaml(Path(args.config))
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    store = ParquetStore(cfg.data_root)
    _save_decision(cfg, store, args.model_dir, args.sample_size,
                   args.retrain, args.out)


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", write_through=True)
        except (AttributeError, ValueError, OSError):
            pass
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
    d = sub.add_parser("daily", help="每日增量更新并生成报告")
    d.add_argument("--config", default="config.yaml")
    d.add_argument("--data-root")
    d.add_argument("--out-dir", default="docs/simulation")
    d.add_argument("--model-dir", default="models")
    d.add_argument("--sample-size", type=int, default=60000)
    d.add_argument("--retrain", action="store_true")
    d.add_argument("--force", action="store_true")
    d.add_argument("--no-decision", action="store_true")
    d.set_defaults(func=cmd_daily)
    rep = sub.add_parser("report", help="仅重新生成 HTML 报告")
    rep.add_argument("--config", default="config.yaml")
    rep.add_argument("--data-root")
    rep.add_argument("--out-dir", default="docs/simulation")
    rep.set_defaults(func=cmd_report)
    bm = sub.add_parser("benchmark", help="运行算法表现对比评测")
    bm.add_argument("--config", default="config.yaml")
    bm.add_argument("--data-root")
    bm.add_argument("--out", default="docs/research/algorithm-benchmark.md")
    bm.add_argument("--with-dl", action="store_true", help="包含 GRU 深度模型")
    bm.set_defaults(func=cmd_benchmark)
    dc = sub.add_parser("decision", help="训练模型并生成当日模拟投资决策")
    dc.add_argument("--config", default="config.yaml")
    dc.add_argument("--data-root")
    dc.add_argument("--model-dir", default="models")
    dc.add_argument("--out", default="docs/decision")
    dc.add_argument("--sample-size", type=int, default=60000)
    dc.add_argument("--retrain", action="store_true")
    dc.set_defaults(func=cmd_decision)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
