"""成交量完整性审计：本地日线 volume/amount 与独立源逐日对撞（2026-09-18 新增）。

**为什么需要它**：项目早就有 `scripts/audit_qfq_anchor.py` 审"价格/前复权锚点"
（每次换主源必跑），但**没有成交量的等价物** —— 而两次真实事故恰恰都出在成交量：
① 09-10 换源时全市场历史 volume 单位是"手"（比"股"小 100 倍）；
② 09-16 起腾讯兜底把科创板 volume 又放大 100 倍。
两者同属**系统性量纲**问题，由 `scripts/fix_volume_unit.py`（按 ratio 判方向）修；
本脚本补的是**逐个 (股票, 日期) 的完整性**：量纲对了，个别行仍可能是错值
（实测 603559 于 2025-01-03：本地 volume 46,400，akshare 与腾讯一致为 5,900）。

**哪个源能当真值，是实测出来的、不是假设的**（2026-09-18 对撞 600000/000001 各 1628 日）：

| 源 | volume | amount |
|---|---|---|
| akshare（新浪） | 真实（股）→ **真值来源** | 真实（元）→ **真值来源** |
| tencent | 可用于**独立确证**（与 akshare 1628/1628 日一致） | ⚠️ **估算式** `volume × 调整后收盘`，实测 1101/1094 日与真值不符（中位比 1.25）→ **绝不可当真值** |

判据与纪律（与 `fix_volume_unit.py` 同款）：
- 容差 ``TOL_DEX``（默认 0.05 dex ≈ ±12%），``volume`` 与 ``amount`` 两列**独立**判定；
- 方向绝不猜，按证据分四类：
  * ``confirmed``：本地 volume 与 akshare 不符，**且腾讯 volume 与 akshare 一致**
    → 该行确证为坏，用 **akshare 的值**修（volume 必改，amount 仅在同样超差时一并改）；
  * ``amount_only``：只有 amount 超差（两源 volume 一致）→ 拿腾讯的估算 amount 去"修"
    只会把好行改坏，**只报告不改**；
  * ``single_source``：腾讯缺该日（多为北交所）→ 无第二确证，**只报告不改**；
  * ``sources_disagree``：腾讯与 akshare 互相矛盾 → **只报告不改**；
- 写入前备份被改文件；幂等：修完再跑应报 ``confirmed = 0``。

用法::

    python scripts/audit_volume_integrity.py                  # 全市场扫描（只报告）
    python scripts/audit_volume_integrity.py --limit 300       # 抽查前 300 只
    python scripts/audit_volume_integrity.py --apply           # 修 confirmed 行（先备份）
    python scripts/audit_volume_integrity.py --verify          # 复测：confirmed 应为 0

产物：``docs/research/volume_integrity_report.json``（逐行明细，便于复现与对账）。
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
# 以 scripts/ 为 sys.path[0] 运行时必须显式补项目根（否则 import ashare_quant 失败，
# 且很容易被宽 except 吞成"真值取不到"—— fix_volume_unit 踩过这个坑）
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

DATA_ROOT = PROJ / "data" / "tencent"
REPORT_PATH = PROJ / "docs" / "research" / "volume_integrity_report.json"
TOL_DEX = 0.05          # 0.05 dex ≈ ±12%
_FETCH_ERRORS: list[str] = []


def market_symbols(root: Path) -> list[str]:
    codes: set[str] = set()
    up = root / "universe.json"
    if up.exists():
        raw = json.loads(up.read_text(encoding="utf-8"))
        seq = raw if isinstance(raw, list) else (raw.get("symbols") or raw.get("codes") or [])
        codes |= {str(c).zfill(6) for c in seq}
    codes |= {p.stem for p in root.glob("*.parquet") if p.stem.isdigit() and len(p.stem) == 6}
    return sorted(c for c in codes if (root / f"{c}.parquet").exists())


def fetch_truth(source: str, sym: str) -> pd.DataFrame | None:
    """独立源日线（volume 单位股、amount 单位元）。取不到返回 None（不猜）。"""
    try:
        from ashare_quant.fetchers import registry
        fetch = registry.get_fetch_daily(source)
        df = fetch(sym, "20200101", date.today().strftime("%Y%m%d"), "qfq")
        return df if df is not None and not df.empty else None
    except Exception as exc:  # noqa: BLE001 - 真值取不到就跳过，但要让原因可见
        if len(_FETCH_ERRORS) < 5:
            _FETCH_ERRORS.append(f"{source} {sym}: {type(exc).__name__} {exc}")
            print(f"  ! {source} 真值获取失败：{_FETCH_ERRORS[-1]}")
        return None


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """a/b，仅在两者都 >0 时有值。"""
    aa, bb = a.astype(float), b.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((aa > 0) & (bb > 0), aa / bb, np.nan)
    return pd.Series(r, index=a.index)


def _out_of_tol(ratio: pd.Series, tol_dex: float) -> pd.Series:
    """log10 距离超容差即为 True；NaN（某一侧为 0/缺失）不算异常。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.abs(np.log10(ratio.where(ratio > 0))) > tol_dex).fillna(False)


def compare(local: pd.DataFrame, truth: pd.DataFrame, tol_dex: float = TOL_DEX) -> pd.DataFrame:
    """→ index=日期，列含 vol_ratio/amt_ratio/bad_vol/bad_amt；只保留任一项超差的行。"""
    common = local.index.intersection(truth.index)
    if len(common) == 0:
        return pd.DataFrame()
    out = pd.DataFrame({
        "vol_ratio": _ratio(local.loc[common, "volume"], truth.loc[common, "volume"]),
        "amt_ratio": _ratio(local.loc[common, "amount"], truth.loc[common, "amount"]),
    })
    out["bad_vol"] = _out_of_tol(out["vol_ratio"], tol_dex)
    out["bad_amt"] = _out_of_tol(out["amt_ratio"], tol_dex)
    return out[out["bad_vol"] | out["bad_amt"]]


def _f(x) -> float | None:
    """取列值时保持"缺就是 None"，且把 NaN 归一成 None（便于 JSON 落盘）。"""
    if x is None:
        return None
    v = float(x)
    return None if not np.isfinite(v) else v


def adjudicate(sym: str, local: pd.DataFrame, ak: pd.DataFrame, tc: pd.DataFrame | None,
               diff: pd.DataFrame, tol_dex: float = TOL_DEX) -> list[dict]:
    """把一只股票的三方对撞结果归入四类。**纯函数、不联网**，判据可被测试直接钉住。

    ``diff`` 是 ``compare(local, ak)`` 的输出；``tc`` 是腾讯日线或 None。
    关键：腾讯只贡献 **volume** 作为独立确证 —— 它的 amount 是估算式，绝不当真值。
    """
    rows: list[dict] = []
    for ts, rec in diff.iterrows():
        bad_vol, bad_amt = bool(rec["bad_vol"]), bool(rec["bad_amt"])
        entry = {
            "symbol": sym, "date": str(ts.date()),
            "bad_vol": bad_vol, "bad_amt": bad_amt,
            "vol_ratio_ak": _f(None if pd.isna(rec["vol_ratio"]) else round(float(rec["vol_ratio"]), 4)),
            "amt_ratio_ak": _f(None if pd.isna(rec["amt_ratio"]) else round(float(rec["amt_ratio"]), 4)),
            "local_volume": _f(local.loc[ts, "volume"]),
            "local_amount": _f(local.loc[ts, "amount"]),
            "ak_volume": _f(ak.loc[ts, "volume"]) if ts in ak.index else None,
            "ak_amount": _f(ak.loc[ts, "amount"]) if ts in ak.index else None,
        }
        if not bad_vol:
            # 只有成交额不符（两源 volume 一致）→ 腾讯的 amount 是估算式，无第二真值
            entry["kind"] = "amount_only"
            rows.append(entry)
            continue
        if tc is None or ts not in tc.index:
            entry["kind"] = "single_source"
            rows.append(entry)
            continue
        tv = _f(tc.loc[ts, "volume"])
        entry["tencent_volume"] = tv
        # 腾讯 volume 与 akshare 一致吗？（这才是"第二源确证"）
        tx_vs_ak = (tv is not None and entry["ak_volume"] is not None
                    and tv > 0 and entry["ak_volume"] > 0
                    and abs(np.log10(entry["ak_volume"] / tv)) <= tol_dex)
        if tx_vs_ak:
            # 两个独立源在 volume 上彼此一致、且都与本地不符 → 本地确证为错值
            entry["kind"] = "confirmed"
            entry["fix"] = {"volume_true": entry["ak_volume"],
                            "amount_true": entry["ak_amount"],
                            "fix_amount": bad_amt}
        else:
            entry["kind"] = "sources_disagree"
        rows.append(entry)
    return rows


def scan(root: Path, syms: list[str], workers: int = 8, tol_dex: float = TOL_DEX) -> dict:
    """第一趟用 akshare 找疑点；第二趟只用腾讯的 **volume** 给疑点定案。"""
    def first_pass(sym: str):
        local = pd.read_parquet(root / f"{sym}.parquet")
        if local.empty or not {"volume", "amount"} <= set(local.columns):
            return sym, None, None, None
        ak = fetch_truth("akshare", sym)
        if ak is None:
            return sym, None, None, "no_truth"
        return sym, local, ak, compare(local, ak, tol_dex)

    suspects: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    no_truth: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, local, ak, diff in ex.map(first_pass, syms):
            if local is None:
                no_truth.append(sym)
            elif diff is not None and not diff.empty:
                suspects.append((sym, local, ak, diff))

    def _adjudicate_stock(item):
        sym, local, ak, diff = item
        return adjudicate(sym, local, ak, fetch_truth("tencent", sym), diff, tol_dex)

    buckets: dict[str, list[dict]] = {"confirmed": [], "amount_only": [],
                                      "single_source": [], "sources_disagree": []}
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as ex:
        for rows in ex.map(_adjudicate_stock, suspects):
            for entry in rows:
                buckets[entry["kind"]].append(entry)

    return {"scanned": len(syms), "no_truth": len(no_truth), "suspect_stocks": len(suspects),
            "tol_dex": tol_dex, **buckets}


def apply_fixes(root: Path, confirmed: list[dict], bak_dir: Path) -> int:
    """confirmed 行：volume 用 akshare 值；amount 仅在 bad_amt 时一并改用 akshare 值。"""
    changed = 0
    by_sym: dict[str, list[dict]] = {}
    for e in confirmed:
        by_sym.setdefault(e["symbol"], []).append(e)
    for sym, entries in by_sym.items():
        p = root / f"{sym}.parquet"
        df = pd.read_parquet(p)
        bak_dir.mkdir(parents=True, exist_ok=True)
        if not (bak_dir / p.name).exists():
            shutil.copy2(p, bak_dir / p.name)
        for e in entries:
            ts = pd.Timestamp(e["date"])
            if ts not in df.index:
                continue
            if e["fix"]["volume_true"] is not None:
                df.loc[ts, "volume"] = float(e["fix"]["volume_true"])
            if e["fix"]["fix_amount"] and e["fix"]["amount_true"] is not None:
                df.loc[ts, "amount"] = float(e["fix"]["amount_true"])
            changed += 1
        df.to_parquet(p)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="成交量完整性审计（本地 vs 独立源）")
    ap.add_argument("--apply", action="store_true", help="修复 confirmed 行（先备份）")
    ap.add_argument("--verify", action="store_true", help="复测：confirmed 应为 0")
    ap.add_argument("--limit", type=int, default=0, help="只扫描前 N 只（0=全部）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = ap.parse_args()
    root: Path = args.data_root

    syms = market_symbols(root)
    if args.limit:
        syms = syms[:args.limit]
    if not syms:
        print("!! 未找到个股数据")
        return 1

    print(f"[SCAN] {len(syms)} 只：先用 akshare 找疑点、再用腾讯 volume 定案"
          f"（容差 {TOL_DEX} dex ≈ ±12%）…", flush=True)
    rep = scan(root, syms, workers=args.workers)
    print(f"  有独立源可比的 {rep['scanned'] - rep['no_truth']} 只，"
          f"其中 {rep['suspect_stocks']} 只有不一致行；{rep['no_truth']} 只取不到 akshare 真值")
    print(f"  行级判定：confirmed(volume 被两源共同确证为错) {len(rep['confirmed'])}；"
          f"amount_only(仅成交额不符，无第二真值) {len(rep['amount_only'])}；"
          f"sources_disagree {len(rep['sources_disagree'])}；single_source {len(rep['single_source'])}")
    for e in rep["confirmed"][:10]:
        line = (f"    · {e['symbol']} {e['date']}: volume {e['local_volume']:,.0f} → "
                f"{e['fix']['volume_true']:,.0f}")
        if e["fix"]["fix_amount"]:
            line += f" | amount {e['local_amount']:,.0f} → {e['fix']['amount_true']:,.0f}"
        print(line)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细报告：{REPORT_PATH}")

    if args.apply:
        if not rep["confirmed"]:
            print("[APPLY] 没有 confirmed 行，未改动任何数据")
        else:
            bak = root.parent / f"tencent_integrity_bak_{date.today():%Y%m%d}"
            n = apply_fixes(root, rep["confirmed"], bak)
            print(f"[APPLY] 已修复 {n} 行；备份：{bak}")
    if args.verify and rep["confirmed"]:
        print(f"  !! verify 失败：仍存在 {len(rep['confirmed'])} 个 confirmed 行")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
