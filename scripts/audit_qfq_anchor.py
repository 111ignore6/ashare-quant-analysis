"""跨源追加的"前复权锚点一致性"审计与修复。

为什么需要：前复权以"最新价"为锚。用 B 源往 A 源建立的历史里追加几天时，
若这期间某只股票除权除息，B 给出的重叠日收盘价会整体缩放，与 A 的历史值
不一致 —— 拼接后会在第一天产生一根假涨跌幅（2026-09-11 实测 5360 只里有
43 只不一致，最大 2.66%，足以污染特征与决策）。

用法（在项目根目录）：
    python scripts/audit_qfq_anchor.py                 # 只审计，输出 logs/qfq_anchor_audit.json
    python scripts/audit_qfq_anchor.py --fix           # 审计 + 用当前主源全量重建不一致的个股
    python scripts/audit_qfq_anchor.py --days 5        # 比对最近 5 个重叠日

换数据源（如 mootdx ↔ akshare ↔ tencent）之后必须跑一次；--fix 后跑
`python -m ashare_quant.cli daily --force` 重建面板与决策。
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ashare_quant.cache import ParquetStore  # noqa: E402
from ashare_quant.config import Config  # noqa: E402
from ashare_quant.fetchers import get_source  # noqa: E402

TOL = 0.001  # 0.1%：超过即认为锚点不同（分红缩放）


def _overlap_days(store: ParquetStore, codes: list[str], n: int) -> list[pd.Timestamp]:
    """取缓存里最近的 n 个交易日作为比对日（不含当天，避免盘中口径）。"""
    idx = store.load("sh000300")
    if idx is None or len(idx) == 0:
        return []
    days = list(idx.index[-(n + 1):-1]) or list(idx.index[-n:])
    return days


def _check_one(code: str, store: ParquetStore, days: list[pd.Timestamp], fetch, start: str,
               end: str, adjust: str):
    old = store.load(code)
    if old is None or old.empty:
        return None
    try:
        new = fetch(code, start, end, adjust)
    except Exception as e:  # noqa: BLE001
        return (code, f"fetch-error {type(e).__name__}", None)
    if new is None or new.empty:
        return None
    bad = []
    for d in days:
        if d in old.index and d in new.index:
            a, b = float(old.loc[d, "close"]), float(new.loc[d, "close"])
            if a > 0 and abs(b - a) / a > TOL:
                bad.append((str(d.date()), a, b, round((b - a) / a, 6)))
    return (code, "mismatch", bad) if bad else (code, "ok", None)


def _rebuild(code: str, store: ParquetStore, fetch, start: str, end: str, adjust: str) -> bool:
    try:
        df = fetch(code, start, end, adjust)
    except Exception:  # noqa: BLE001
        return False
    if df is None or df.empty:
        return False
    store.save(code, df)   # 整段用同一口径覆写，而不是只补最后几天
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="前复权锚点一致性审计")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--days", type=int, default=3, help="比对最近几个重叠交易日")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fix", action="store_true", help="用当前主源全量重建不一致的个股")
    args = ap.parse_args()

    cfg = Config.from_yaml(Path(args.config))
    if not Path(cfg.data_root).is_absolute():
        cfg.data_root = ROOT / cfg.data_root   # 脚本里按项目根解析，不依赖当前目录
    store = ParquetStore(cfg.data_root)
    fetch = get_source(cfg.data_source).fetch_daily
    manifest = store.read_manifest()
    codes = sorted(c for c in manifest if c != "sh000300")
    today = pd.Timestamp.today().normalize()
    days = _overlap_days(store, codes, args.days)
    look_start = (days[0] if days else today - pd.Timedelta(days=args.days)).strftime("%Y%m%d")
    print(f"主源 {cfg.data_source}｜{len(codes)} 只｜比对 {len(days)} 个重叠日 "
          f"{'、'.join(str(d.date()) for d in days)}（容差 {TOL:.1%}）", flush=True)
    if not days:
        print("指数缓存为空，无法确定比对日。", flush=True)
        return 1

    mismatched: list[tuple[str, list]] = []
    errors: list[tuple[str, str]] = []
    ok = 0
    end = today.strftime("%Y%m%d")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(
                lambda c: _check_one(c, store, days, fetch, look_start, end, cfg.adjust),
                codes), 1):
            if r is None:
                continue
            code, kind, bad = r
            if kind.startswith("fetch-error"):
                errors.append((code, kind))
            elif kind == "mismatch":
                mismatched.append((code, bad))
            else:
                ok += 1
            if i % 500 == 0:
                print(f"  …{i}/{len(codes)} ok={ok} mismatch={len(mismatched)} "
                      f"err={len(errors)}", flush=True)

    print(f"\n结果：一致 {ok} 只｜锚点不一致 {len(mismatched)} 只｜抓取异常 {len(errors)} 只")
    for code, bad in mismatched[:40]:
        print("  ", code, bad)
    out = ROOT / "logs" / "qfq_anchor_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"source": cfg.data_source, "checked_days": [str(d.date()) for d in days],
         "ok": ok, "mismatched": {c: b for c, b in mismatched},
         "errors": {c: e for c, e in errors}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {out.relative_to(ROOT)}", flush=True)

    if args.fix and mismatched:
        full_start = (today - pd.DateOffset(years=cfg.years)).strftime("%Y%m%d")
        print(f"\n用 {cfg.data_source} 全量重建 {len(mismatched)} 只…", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            done = sum(ex.map(lambda c: _rebuild(c, store, fetch, full_start, end, cfg.adjust),
                              [c for c, _ in mismatched]))
        print(f"重建完成 {done}/{len(mismatched)}；请接着跑 daily --force 重建面板与决策", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
