"""精确复现 2026-08-11 15:58 仪表盘「+0.59%」的完整证据链。

结论（复现值 +0.5915%，与用户看到的 +0.59% 一致）：
1. 当日 10:35 盘中抓取把 08-11「当日 bar」当成收盘写入，实际是 11:25 的
   盘中快照价（如 000779 旧 close=10.42 = 地天板前的盘中低点，真实收盘
   12.73 涨停；688807 旧 247.20 → 真实 258.52）。
2. 13:10/13:25 两次运行用 10:35 生成的特征缓存（data/tencent_parquet_bak
   /features.parquet，08-11 当日仅 1880 只有完整特征）产出 08-11 决策。
3. 15:58 仪表盘「盘中实时估值」按 决策日(08-11)收盘成本 × 腾讯实时快照
   估值：成本=旧面板 08-11 盘中价，现价=腾讯快照 now（15:00 收盘后冻结
   为真实收盘价）→ 对 000779 等日内反转股产生巨大失真收益。
4. 对照：账户页 08-10 决策 × 旧面板 08-11 盘中价 = -0.1637%（≈ 日志里的
   -0.16%），与实时估值 +0.59% 是两个不同口径，所以「总览/账户」和
   「实时」对不上。

git 对比（255fc5c@13:24 → HEAD）：decide/特征逻辑未变，唯一差异是 17:03
新增的 target 列剔除，与本复现无关（特征缓存已无 target 列）。关键证据
是备份目录中的旧 features.parquet（10:35 写入）——它才是当时决策用的 X。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BAK = ROOT / "data" / "tencent_parquet_bak"


def old_row(code: str, date: str) -> float | None:
    """读取备份 parquet 中指定日期的 close（旧盘中价/旧前复权数据）。"""
    f = BAK / f"{code}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) \
        else pd.to_datetime(df["date"])
    return float(df.loc[date, "close"]) if date in idx else None


def main() -> None:
    from ashare_quant.ml.decision import decide

    # 1) 13:10/13:25 决策用的正是这份 10:35 生成的特征缓存
    cache = pd.read_parquet(BAK / "features.parquet")
    X_old = cache.drop(columns=["target"])
    d0811 = X_old.index.get_level_values("date") == pd.Timestamp("2026-08-11")
    print(f"旧特征缓存: {len(X_old)} 行, 08-11 当日 {int(d0811.sum())} 只")

    models = {
        "models": {
            name: joblib.load(ROOT / "models" / "all" / f"{name}.joblib")
            for name in ("lgbm", "histgb", "rf", "svm", "knn", "linear")
        },
        "meta": json.loads(
            (ROOT / "models" / "all" / "meta.json").read_text(encoding="utf-8")),
    }
    picks = decide(models, X_old, pd.DataFrame(), "2026-08-11", top_n=50)
    syms = list(picks["symbol"])
    print(f"重建 13:25 决策: {len(syms)} 只")
    print("Top20:", ",".join(syms[:20]))

    # 与日志中 13:25 打印的前 20 只比对
    log20 = ["001309", "002208", "300005", "688807", "002993", "000988",
             "300553", "300042", "300136", "300302", "300257", "002272",
             "002965", "002733", "002782", "300322", "002943", "002865",
             "002518", "001259"]
    print("与日志前20一致:", syms[:20] == log20)

    # 2) 15:58 实时估值：成本=旧面板 08-11 盘中价，现价=真实 08-11 收盘
    panel_new = pd.read_parquet(
        ROOT / "data" / "tencent" / "panels" / "close.parquet")
    cost = pd.Series({c: old_row(c, "2026-08-11") for c in syms}, dtype=float)
    real = panel_new.loc["2026-08-11", syms].astype(float)
    w = pd.Series(1.0 / len(syms), index=syms)
    valid = cost.notna() & real.notna() & (cost > 0)
    total_return = float(
        (w[valid] * real[valid] / cost[valid]).sum() / w[valid].sum() - 1)
    print(f"\n15:58 实时估值复算（旧面板08-11盘中价成本 × 真实收盘）: "
          f"{total_return:+.4%}")

    # 3) 对照：账户页 08-10 决策 × 旧面板 08-11 盘中价（= 日志 -0.16%）
    hist = [json.loads(line) for line in
            (ROOT / "data" / "tencent" / "portfolio" / "account_history.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    dec810 = next(d for d in hist if d.get("date") == "2026-08-10")
    s10 = [p["symbol"] for p in dec810["picks"]]
    c10 = pd.Series({c: old_row(c, "2026-08-10") for c in s10}, dtype=float)
    o11 = pd.Series({c: old_row(c, "2026-08-11") for c in s10}, dtype=float)
    w10 = pd.Series(1.0 / len(s10), index=s10)
    m10 = c10.notna() & o11.notna() & (c10 > 0)
    day_ret = float(
        (w10[m10] * o11[m10] / c10[m10]).sum() / w10[m10].sum() - 1)
    print(f"对照 账户页 08-10→08-11 段（旧面板盘中价, 日志 -0.16%）: "
          f"{day_ret:+.4%}")

    # 4) 失真最大的个股（供复核）
    contrib = (w[valid] * real[valid] / cost[valid] - w[valid])
    top = contrib.abs().sort_values(ascending=False).head(5)
    print("\n失真贡献 Top5:")
    for c, _ in top.items():
        print(f"  {c}: {contrib[c]:+.4%}  旧盘中={cost[c]:.2f} 真实收盘={real[c]:.2f}")

    out = {
        "rebuilt_decision_date": "2026-08-11",
        "picks": syms,
        "top20_matches_log": syms[:20] == log20,
        "realtime_return_1558": total_return,
        "account_day_return_0810_0811": day_ret,
        "note": "成本=旧面板08-11盘中价(10:35抓取), 现价=真实收盘(腾讯15:58快照冻结值)",
    }
    (ROOT / "docs" / "repro_1558.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已保存 docs/repro_1558.json")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
