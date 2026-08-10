"""每日决策日报：数据状态 + 今日持仓 + 账户概览 + 数据质量，输出 Markdown。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _data_quality(manifest: dict, index_symbol: str = "sh000300") -> dict:
    idx_end = (manifest or {}).get(index_symbol, {}).get("end")
    stocks = {k: v for k, v in (manifest or {}).items() if k != index_symbol}
    if not stocks or not idx_end:
        return {"stocks": 0, "complete": 0, "stale": [], "missing": [], "rate": 0.0}
    stale = sorted(k for k, v in stocks.items() if v.get("end") and v["end"] < idx_end)
    return {
        "stocks": len(stocks),
        "complete": len(stocks) - len(stale),
        "stale": stale,
        "missing": sorted(k for k, v in stocks.items() if not v.get("end")),
        "rate": (len(stocks) - len(stale)) / len(stocks),
    }


def build_daily_report(cfg, update_out: dict, decision: dict | None,
                       account: dict | None, manifest: dict,
                       out_path: Path) -> Path:
    """生成 Markdown 日报并返回路径。"""
    dq = _data_quality(manifest)
    lines = [
        "# A股量化研究 · 每日决策日报",
        "",
        f"> 生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}　"
        "模拟研究，不构成投资建议。",
        "",
        "## 数据状态",
        "",
        f"- 指数截止：{update_out.get('new_index_date', '-')}",
        f"- 增量更新：{len(update_out.get('updated', []))} 只；"
        f"已最新：{len(update_out.get('up_to_date', [])) if isinstance(update_out.get('up_to_date'), list) else '全部'}；"
        f"失败：{len(update_out.get('failed', []))} 只",
    ]
    if update_out.get("no_data"):
        lines.append(f"- 无数据（可能停牌）：{len(update_out['no_data'])} 只，"
                     f"如 {'、'.join(update_out['no_data'][:5])}…")
    lines += [
        "",
        "## 数据质量",
        "",
        f"- 覆盖股票：{dq['stocks']} 只；与指数同步（数据完整率）：**{dq['rate']:.1%}**"
        f"（{dq['complete']}/{dq['stocks']}）",
    ]
    if dq["stale"]:
        lines.append(f"- 落后股票：{len(dq['stale'])} 只，"
                     f"如 {'、'.join(dq['stale'][:10])}…")
    if update_out.get("failed"):
        lines.append(f"- 更新失败：{'、'.join(update_out['failed'][:10])}"
                     f"{'…' if len(update_out['failed']) > 10 else ''}"
                     "（重跑 daily 自动重试）")
    lines += ["", "## 今日决策", ""]
    if decision is None:
        lines.append("- 今日未生成决策。")
    else:
        models = "、".join(decision.get("models", []))
        lines += [
            f"- 决策日期：{decision['date']}　持仓 {len(decision.get('picks', []))} 只　"
            f"模型：{models}",
            "",
            "| 代码 | 加权得分 | 预期收益(20日) |",
            "|---|---|---|",
        ]
        for p in decision.get("picks", [])[:15]:
            lines.append(f"| {p['symbol']} | {p.get('score', 0):.4f} | "
                         f"{p.get('score', 0):+.2%} |")
        if len(decision.get("picks", [])) > 15:
            lines.append(f"| … | 共 {len(decision['picks'])} 只 | |")
    lines += ["", "## 模拟账户", ""]
    if account is None:
        lines.append("- 账户尚未开始记录（首日生成后开始跟踪）。")
    else:
        initial = account.get("initial_capital", account.get("initial", 0))
        total = account.get("total_asset", initial)
        lines += [
            f"- 初始资金：{initial:,.0f} 元　总资产：{total:,.0f} 元",
            f"- 总收益率：**{account.get('total_return', 0.0):+.2%}**",
            f"- 决策记录：{account.get('decisions', '-')} 次　"
            f"数据截至：{account.get('as_of', '-')}",
        ]
    lines += ["", "---", "",
              "> 每日 16:05 由 Windows 计划任务自动生成；"
              "打开仪表盘（start.bat → 启动系统.bat）可查看净值曲线与实时行情。"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
