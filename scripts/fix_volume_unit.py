"""修复本地日线 volume 的量纲污染（全市场，2026-09-18）。

背景与完整证据见 AGENTS.md「已知问题」4。**全市场**的本地 volume 有两处台阶：
① **09-10 换源**（mootdx → akshare）：mootdx 时代（≤09-09）写的是"手"、akshare 是"股"，
   全市场 4710 只、约 339 万行因此比"股"小 100 倍；
② **科创板另有 09-16 起的"股×100"**（`tencent_fetcher` 旧版对 688/689 也无条件 ×100）。
跨台阶会让
`features.py` 的 `vol_ratio = volume.rolling(5).mean()/volume.rolling(20).mean()`
算出错误值，且台阶后约 20 个交易日都受影响。

判据（两层，逐行判定，不用"日期分段"这种脆弱假设）：

    ref   = amount / close          # 股数真值（amount 单位元、close 元/股）
    ratio = volume / ref

  第一层（amount 可信时）：
    ratio > 10   → volume /= 100    # volume 虚高 100 倍（科创板 09-16 起）
    ratio < 0.1  → volume *= 100    # volume 单位是"手"（全市场 ≤09-09 的历史）
  第二层（ratio ≈ 1，两列可能一起错）：腾讯的 ``amount = volume*close`` 是**估算式**，
    会跟着 volume 同倍缩放，此时 ratio 恒为 1，**靠数据自身无法定方向**。
    这类行必须用**独立源真值**（akshare 成交额）判定：
      local_amt / truth_amt > 10 → 两列同除 100；< 0.1 → 两列同乘 100；否则不动。

⚠️ 两个实测踩过的坑，别改回去：
  （a）用 ``amount == volume*close`` 当**判据**不幂等 —— 两列同除 100 后等式依然
      成立，再跑一次会再除一次（数据毁成 1/10000）；
  （b）用"邻域中位成交额"当**方向判据**会误伤 —— 干净行紧挨着被放大的行时，
      它相对被污染的邻域反而显得"偏小"，会被错误地乘 100（实测 78 行假阳性）。
      邻域只能用来**发现可疑行**，方向必须问独立源。

用法::

    python scripts/fix_star_market_volume.py             # dry-run（默认，只报告）
    python scripts/fix_star_market_volume.py --apply     # 写入（先备份被改文件）
    python scripts/fix_star_market_volume.py --verify    # 全量对撞 akshare 验收

备份目录：``data/tencent_volume_bak_<YYYYMMDD>/``（仅备份被修改的文件）。
改完必须接着跑：``daily --force``（重建面板/特征/决策）+
``daily --retrain``（训练集里的 688 特征同样被污染过），最后 ``--verify`` 复测。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
# 以 scripts/ 为 sys.path[0] 运行时 import ashare_quant 会失败（曾被宽 except 吞成
# "真值取不到"，导致 130 行全部跳过而看不出原因）——这里显式补项目根。
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

DATA_ROOT = PROJ / "data" / "tencent"
STAR_PREFIXES = ("688", "689")  # 仅用于诊断标注
# 可疑行判据用的长期窗口（±60 交易日：远长于任何一次污染连续段，又短于价格量级漂移）
_LEVEL_WIN = 121
_TRUTH_ERRORS: list[str] = []


def market_symbols(root: Path) -> list[str]:
    """数据目录里的全部个股（6 位代码；指数等其他文件由 ``is_symbol_stem`` 之外的规则排除）。"""
    codes: set[str] = set()
    up = root / "universe.json"
    if up.exists():
        raw = json.loads(up.read_text(encoding="utf-8"))
        seq = raw if isinstance(raw, list) else (raw.get("symbols") or raw.get("codes") or [])
        codes |= {str(c).zfill(6) for c in seq}
    codes |= {p.stem for p in root.glob("*.parquet") if p.stem.isdigit() and len(p.stem) == 6}
    return sorted(c for c in codes if (root / f"{c}.parquet").exists())


def star_symbols(root: Path) -> list[str]:
    """科创板子集（诊断用）。"""
    return [c for c in market_symbols(root) if c[:3] in STAR_PREFIXES]


def akshare_truth(sym: str) -> pd.DataFrame | None:
    """独立源真值（akshare 新浪日线，volume 单位股、amount 单位元）；失败返回 None。

    失败原因不静默：首条错误会打印出来（真值取不到时脚本只会"跳过该行"，
    若不报错，方向不明的行会被无声跳过而看不出原因）。
    """
    try:
        from ashare_quant.fetchers import registry
        fetch = registry.get_fetch_daily("akshare")
        df = fetch(sym, "20200101", date.today().strftime("%Y%m%d"), "qfq")
        return df if df is not None and not df.empty else None
    except Exception as exc:  # noqa: BLE001 - 真值取不到就保守跳过（不猜方向）
        if len(_TRUTH_ERRORS) < 3:
            _TRUTH_ERRORS.append(f"{sym}: {type(exc).__name__} {exc}")
            print(f"  ! 独立源真值获取失败（该股方向不明的行将跳过）：{_TRUTH_ERRORS[-1]}")
        return None


def classify(df: pd.DataFrame):
    """→ (inflated, in_hand, suspect) 三个行掩码。

    inflated / in_hand：``amount`` 可作锚（ratio 明显偏离 1），仅 volume 需要 /100 或 *100；
    suspect：ratio ≈ 1、但两列**可能一起错**的行（腾讯 ``amount = volume*close`` 是估算式，
    会跟着 volume 同倍缩放，此时数据自身无法定方向）—— 必须交给 ``resolve()``
    用独立源真值定方向。
    """
    vol = df["volume"].astype(float)
    amt = df["amount"].astype(float)
    close = df["close"].astype(float)
    ref = amt / close.replace(0, np.nan)
    ratio = vol / ref.replace(0, np.nan)
    inflated = (ratio > 10).fillna(False)
    in_hand = (ratio < 0.1) & ratio.notna()
    mid = ratio.notna() & ~inflated & ~in_hand

    # 发现可疑行：估算式（amount 恰等于 volume*close），或与长期成交额水平差 100 倍
    roll = amt.rolling(_LEVEL_WIN, center=True, min_periods=20)
    level = (roll.sum() - amt) / (roll.count() - 1).replace(0, np.nan)
    suspect = mid & (
        np.isclose(amt, vol * close, rtol=1e-9, equal_nan=False)
        | (level.notna() & ((amt > 10 * level) | (amt < 0.1 * level))))
    return inflated, in_hand.fillna(False), suspect.fillna(False)


def resolve(df: pd.DataFrame, suspect, truth: pd.DataFrame):
    """用独立源真值给可疑行定方向 → (over, under, checked_fine)。"""
    amt = df["amount"].astype(float)
    ta = truth["amount"].astype(float).reindex(df.index)
    rel = amt / ta.replace(0, np.nan)
    over = (suspect & (rel > 10)).fillna(False)
    under = (suspect & (rel < 0.1)).fillna(False)
    return over, under, (suspect & ~over & ~under).fillna(False)


def normalize(df: pd.DataFrame, inflated, in_hand, over, under) -> pd.DataFrame:
    out = df.copy()
    vol = out["volume"].astype(float).copy()
    amt = out["amount"].astype(float).copy()
    vol[inflated] = vol[inflated] / 100.0     # 股×100 → 股
    vol[in_hand] = vol[in_hand] * 100.0       # 手 → 股
    vol[over] = vol[over] / 100.0             # 两列一起被放大
    amt[over] = amt[over] / 100.0
    vol[under] = vol[under] * 100.0           # 两列一起被缩小
    amt[under] = amt[under] * 100.0
    out["volume"] = vol
    out["amount"] = amt
    return out


def tencent_truth(sym: str) -> pd.DataFrame | None:
    """第二独立源真值（腾讯日线；`tencent_fetcher` 已按板块修正单位，688/689 输出股）。

    akshare 取不到时兜底（实测 689009 CDR 会让 akshare 抛 JSONDecodeError）。
    """
    try:
        from ashare_quant.fetchers import registry
        fetch = registry.get_fetch_daily("tencent")
        df = fetch(sym, "20200101", date.today().strftime("%Y%m%d"), "qfq")
        return df if df is not None and not df.empty else None
    except Exception as exc:  # noqa: BLE001
        if len(_TRUTH_ERRORS) < 6:
            _TRUTH_ERRORS.append(f"tencent {sym}: {type(exc).__name__} {exc}")
            print(f"  ! 腾讯真值获取失败：{_TRUTH_ERRORS[-1]}")
        return None


def verify(root: Path, syms: list[str], workers: int = 12) -> int:
    """全量验收：逐只与 akshare 对撞 volume/amount（重叠日全比），返回失败只数。"""
    def one(sym: str):
        local = pd.read_parquet(root / f"{sym}.parquet")
        truth = akshare_truth(sym)
        if truth is None or local.empty:
            return sym, None
        common = local.index.intersection(truth.index)
        if len(common) == 0:
            return sym, None
        lv = local.loc[common, "volume"].astype(float).to_numpy()
        tv = truth.loc[common, "volume"].astype(float).to_numpy()
        la = local.loc[common, "amount"].astype(float).to_numpy()
        ta = truth.loc[common, "amount"].astype(float).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rv = np.where(tv > 0, lv / tv, np.nan)
            ra = np.where(ta > 0, la / ta, np.nan)
        bad_v = int(np.nansum(np.abs(np.log10(np.where(rv > 0, rv, np.nan))) > 0.05))
        bad_a = int(np.nansum(np.abs(np.log10(np.where(ra > 0, ra, np.nan))) > 0.05))
        return sym, (len(common), bad_v, bad_a)

    bad: list[str] = []
    checked = 0
    rows = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, res in ex.map(one, syms):
            if res is None:
                continue
            n, bad_v, bad_a = res
            checked += 1
            rows += n
            if bad_v or bad_a:
                bad.append(f"{sym}(vol差{bad_v}/amt差{bad_a}, n={n})")
    print(f"[VERIFY] 对撞 akshare：{checked}/{len(syms)} 只有重叠数据，"
          f"共比对 {rows:,} 个交易日")
    if bad:
        print(f"  !! 仍不一致 {len(bad)} 只：{', '.join(bad[:10])}"
              f"{' …' if len(bad) > 10 else ''}")
    else:
        print("  OK：全部重叠日的 volume 与 amount 都在 0.05 dex（≈±12%）容差内")
    return len(bad)


def main() -> int:
    ap = argparse.ArgumentParser(description="修复科创板 volume 量纲")
    ap.add_argument("--apply", action="store_true", help="实际写入（默认 dry-run）")
    ap.add_argument("--verify", action="store_true", help="只做全量 akshare 验收")
    ap.add_argument("--workers", type=int, default=12,
                    help="验收并发（新浪会在高并发下限流，补验用 3~4）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只扫描前 N 只（快速抽查/幂等自检用；0=全部）")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = ap.parse_args()
    root: Path = args.data_root

    syms = market_symbols(root)
    if args.limit:
        syms = syms[:args.limit]
    if not syms:
        print(f"!! 未找到科创板数据（{root}）")
        return 1
    if args.verify:
        return 1 if verify(root, syms, workers=args.workers) else 0

    bak = root.parent / f"tencent_volume_bak_{date.today():%Y%m%d}"
    totals = {"files": 0, "inflated": 0, "hand": 0, "over": 0, "under": 0}
    unresolved = 0          # 两个独立源都取不到 → 保守跳过（绝不猜方向）
    checked_fine = 0        # 对撞真值后确认本来就正常
    truth_stats = {"akshare": 0, "tencent": 0}
    samples: list[str] = []

    for sym in syms:
        p = root / f"{sym}.parquet"
        df = pd.read_parquet(p)
        if not {"volume", "amount", "close"} <= set(df.columns) or df.empty:
            continue
        inflated, in_hand, suspect = classify(df)
        over = pd.Series(False, index=df.index)
        under = pd.Series(False, index=df.index)
        if suspect.any():
            # 方向不明的行（估算式/异常水平）：依次问两个独立源，绝不猜方向
            for label, getter in (("akshare", akshare_truth), ("tencent", tencent_truth)):
                truth = getter(sym)
                if truth is None or truth.empty:
                    continue
                over, under, fine = resolve(df, suspect, truth)
                truth_stats[label] += 1
                checked_fine += int(fine.sum())
                break
            else:
                unresolved += int(suspect.sum())
        n = int(inflated.sum() + in_hand.sum() + over.sum() + under.sum())
        if n == 0:
            continue
        totals["files"] += 1
        totals["inflated"] += int(inflated.sum())
        totals["hand"] += int(in_hand.sum())
        totals["over"] += int(over.sum())
        totals["under"] += int(under.sum())
        if len(samples) < 3:
            idx = df.index[inflated | in_hand | over | under][:2]
            samples.append(f"  {sym}: " + ", ".join(str(d.date()) for d in idx))
        if args.apply:
            bak.mkdir(parents=True, exist_ok=True)
            if not (bak / p.name).exists():
                shutil.copy2(p, bak / p.name)
            normalize(df, inflated, in_hand, over, under).to_parquet(p)

    mode = "APPLY" if args.apply else "DRY-RUN"
    star = sum(1 for s in syms if s[:3] in STAR_PREFIXES)
    print(f"[{mode}] 扫描 {len(syms)} 只（其中科创板 {star} 只），需要修复 {totals['files']} 只")
    print(f"  行数：volume虚高100倍 {totals['inflated']:,}；"
          f"volume单位是手 {totals['hand']:,}；"
          f"两列一起放大 {totals['over']:,}；两列一起缩小 {totals['under']:,}")
    print(f"  独立源定方向：akshare {truth_stats['akshare']} 只 / "
          f"tencent {truth_stats['tencent']} 只；"
          f"对撞后确认正常 {checked_fine} 行；两源都取不到而跳过 {unresolved} 行")
    for s in samples:
        print(s)
    if args.apply:
        print(f"  备份：{bak}")
    else:
        print("  未写入任何文件（加 --apply 才落盘）")

    # 快速复测：最近一日比值应 ≈ 1.00
    ratios = []
    for sym in syms[:400]:
        d = pd.read_parquet(root / f"{sym}.parquet")
        if d.empty:
            continue
        v, a, c = (float(d[k].iloc[-1]) for k in ("volume", "amount", "close"))
        if c and a:
            ratios.append(v / (a / c))
    if ratios:
        med = float(np.median(ratios))
        flag = "OK" if 0.5 < med < 2 else "!! 仍不正常"
        print(f"  复测（最近一日，{len(ratios)} 只）：中位数 {med:.3f} {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
