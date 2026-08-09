from __future__ import annotations

from pathlib import Path

import pandas as pd


def conclusions(results: dict) -> list[str]:
    conc: list[str] = []
    dist = results.get("distribution", {})
    kurt = dist.get("kurtosis") or 0.0
    conc.append(f"R1 收益分布{'呈现厚尾' if kurt > 3 else '接近正态'}（横截面中位峰度 {kurt:.2f}，样本 {dist.get('n_symbols', '?')} 只）")

    vol = results.get("volatility", {})
    p = vol.get("ljungbox_p", 1.0)
    conc.append(f"R2 波动率{'存在显著聚集效应' if p < 0.05 else '未发现显著聚集'}（Ljung-Box p={p:.4f}，ARCH p={vol.get('arch_p', 1.0):.4f}）")

    mom = results.get("momentum")
    if mom is not None and not mom.empty:
        best = mom.loc[mom["mean_ic"].abs().idxmax()]
        direction = "动量" if best["mean_ic"] > 0 else "反转"
        icir_txt = f"，ICIR={best['icir']:.2f}" if "icir" in mom.columns else ""
        conc.append(f"R3 {direction}效应在 {int(best['horizon'])} 日尺度最强（mean_IC={best['mean_ic']:.3f}{icir_txt}）")

    fac = results.get("factor_summary")
    if fac is not None and not fac.empty:
        strong = fac[fac["icir"].abs() > 0.3].index.tolist()
        conc.append(f"R4 稳定有效因子：{'、'.join(strong) if strong else '暂无明显有效因子（|ICIR|>0.3）'}")

    pca = results.get("pca", {})
    r1 = pca.get("first_ratio", 0.0)
    conc.append(f"R5 因子冗余度：第一主成分解释 {r1:.0%}（{'因子高度冗余，需降维' if r1 > 0.5 else '因子相对独立'}）")

    reg = results.get("regimes")
    if reg is not None and not reg.empty and len(reg) > 1:
        spread = reg["mean_fwd"].max() - reg["mean_fwd"].min()
        conc.append(f"R6 不同市场状态下未来收益差异明显（状态间均值差 {spread:.3f}）")
    return conc


def build_report(results: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# A股历史数据研究报告（研究阶段）", "",
             "> 模拟研究，仅用于数据分析与学习，不构成投资建议。", "",
             f"> 数据截止日期：{results.get('data_through', '未知')}", "",
             "## 研究结论", ""]
    lines += [f"- {c}" for c in conclusions(results)]
    lines += ["", "## 明细数据", "", "```json", results.get("raw_json", "见同目录 results.json"), "```"]
    path.write_text("\n".join(lines), encoding="utf-8")
