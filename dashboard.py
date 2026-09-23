"""A股量化研究·模拟分析控制台（Streamlit 单页应用）。

运行： python -m streamlit run dashboard.py
功能：状态总览 / 数据下载与更新（后台任务+实时输出）/ 模拟盘 / 今日决策 / 算法对比 / 日志。

界面约定（改样式前先读）：
  * 主题令牌集中在 ``.streamlit/config.toml``；组件样式集中在下方 ``_CSS``。
  * A 股配色习惯是「红涨绿跌」，与欧美默认相反 —— 方向色一律用 UP / DOWN，
    图表与 KPI 卡片都按这个约定，不要用 plotly/Streamlit 的默认涨跌色。
  * KPI 卡片是自绘 HTML（``kpi_row``），比 st.metric 更容易做等高等宽与方向着色；
    数值一律经 ``html.escape`` 转义，不要把原始行情字符串直接拼进 HTML。

口径约定（不要改）：
  * 实时估值基准 = 决策日累计净资产 ``portfolio.account_basis``，不是初始资金。
  * 绩效指标（年化/夏普/回撤/胜率等）不足 ``MIN_METRIC_DAYS`` 个交易日不展示。
  * ``st.cache_data`` 一律设 TTL(<=60s)，且禁止在一个 cache_data 函数里调用另一个。
"""

from __future__ import annotations

import html
import json
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from ashare_quant.account import account_snapshot
from ashare_quant.config import update_config_yaml
from ashare_quant.portfolio import (account_basis, build_trade_ledger,
                                    costs_from_config, load_history,
                                    monthly_returns_table, recompute_account)
from ashare_quant.realtime import index_snapshot, snapshot

PROJECT = Path(__file__).parent
DISCLAIMER = "模拟研究，仅用于数据分析与学习，不构成投资建议。"


def account_params():
    """账户口径参数（交易成本 + 换仓节奏）。

    必须与 portfolio.update_portfolio 写盘时用的一致，否则仪表盘预览（按当前资金
    重算）会和已落盘的 account_equity.csv 对不上：写盘那边扣了成本、按 config 的
    节奏换仓，这里若用默认值就是毛收益 + 每日换仓，两个数会差一大截。
    （刻意不用 st.cache_data 包裹：config.yaml 很小，且 AGENTS.md 记过
      "cached 函数里嵌套调用另一个 cached 函数会跨刷新抛 KeyError" 的坑。）
    """
    from ashare_quant.config import Config
    cfg = Config.from_yaml(PROJECT / "config.yaml")
    return costs_from_config(cfg), cfg.rebalance

# 账户绩效指标（年化/夏普/回撤/胜率等）至少需要这么多交易日才有统计意义
MIN_METRIC_DAYS = 20
# 等权全市场基准要求当日有效成分股占比不低于该阈值（见 equal_weight_bench）
BENCH_MIN_COVERAGE = 0.5

# ---------------------------------------------------------------- 设计令牌
#
# 为什么是「两套令牌 + 运行时选一套」，而不是「从 Streamlit 的 CSS 变量派生」：
# 实测（playwright 遍历全站样式表 + 各容器 computedStyle）Streamlit
# **不向外暴露任何 CSS 变量** —— document.documentElement 上 `--*` 数量为 0，
# 全站带 `--` 声明的规则只有本文件自己注入的那十几条。因此没有
# `var(--background-color)` 之类可以派生，只能自己维护明/暗两套色值：
#   * `.streamlit/config.toml` 的 [theme.light] / [theme.dark] 管 Streamlit 自带组件；
#   * 本文件的 THEMES 管自绘组件（KPI 卡片/药丸/空状态）与 plotly 图表。
# 两边色值必须成对维护，改一边记得改另一边。
#
# 主题类型用 st.context.theme.type 读取（本机 streamlit 1.61.1 实测存在）。
# 官方注明该值在「会话首次加载」与「用户刚在设置菜单切主题」这两种时刻可能滞后一
# 次 rerun（streamlit#11920）——即那一瞬间自绘组件可能用错配色，下一次 rerun 自愈。

FONT_STACK = ('Inter, "Segoe UI", "Microsoft YaHei", "PingFang SC", '
              '"Hiragino Sans GB", "Noto Sans CJK SC", sans-serif')
# 与 .streamlit/config.toml 的 chartCategoricalColors 保持同步
CHART_COLORS = ["#1d4ed8", "#0f9960", "#d92b2b", "#b45309", "#7c3aed",
                "#0891b2", "#64748b", "#be185d", "#4d7c0f", "#a16207"]
PLOTLY_CONFIG = {"displaylogo": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

THEMES = {
    "light": dict(
        up="#d92b2b", down="#0f9960", accent="#1d4ed8",
        ink="#0f172a", muted="#64748b", line="#e3e8ef", grid="#eef2f7",
        card="#ffffff", page="#f5f7fa", soft="#f8fafc",
        tab_fg="#4a5568", code_fg="#334155",
        accent_soft="rgba(29,78,216,.07)", accent_soft2="rgba(29,78,216,.06)",
        shadow="0 1px 2px rgba(15,23,42,.04), 0 1px 3px rgba(15,23,42,.06)",
        ok_bg="#f0fbf5", ok_line="#bbe3cd", ok_fg="#0b7a4b",
        warn_bg="#fffaf0", warn_line="#f3d9a6", warn_fg="#96590d",
        bad_bg="#fdf3f3", bad_line="#f4c4c4", bad_fg="#b02525",
        info_bg="#f2f7ff", info_line="#c7d9f7", info_fg="#1c4d9e",
        empty_bg="#fbfcfe", empty_line="#d3dae6",
        heat_mid="#f8fafc", plotly_template="plotly_white",
        bench1="#94a3b8", bench2="#cbd5e1", bar_warn="#f0a020",
        ma1="#1d4ed8", ma2="#b45309", ma3="#7c3aed",
    ),
    "dark": dict(
        up="#f85149", down="#3fb950", accent="#4c8dff",
        ink="#e6edf3", muted="#8b949e", line="#262c36", grid="#1f2630",
        card="#161b22", page="#0d1117", soft="#1b222c",
        tab_fg="#9aa4b2", code_fg="#c9d1d9",
        accent_soft="rgba(76,141,255,.14)", accent_soft2="rgba(76,141,255,.10)",
        shadow="0 1px 2px rgba(0,0,0,.35), 0 1px 3px rgba(0,0,0,.45)",
        ok_bg="rgba(63,185,80,.13)", ok_line="rgba(63,185,80,.38)", ok_fg="#56d364",
        warn_bg="rgba(210,153,34,.15)", warn_line="rgba(210,153,34,.42)", warn_fg="#e3b341",
        bad_bg="rgba(248,81,73,.15)", bad_line="rgba(248,81,73,.42)", bad_fg="#ff7b72",
        info_bg="rgba(76,141,255,.13)", info_line="rgba(76,141,255,.37)", info_fg="#79b8ff",
        empty_bg="#12181f", empty_line="#2b333d",
        heat_mid="#161b22", plotly_template="plotly_dark",
        bench1="#6e7681", bench2="#484f58", bar_warn="#d29922",
        ma1="#4c8dff", ma2="#e3b341", ma3="#a78bfa",
    ),
}


def active_theme_type() -> str:
    """当前主题类型：'light' / 'dark'（取不到时按浅色处理）。"""
    try:
        theme_type = st.context.theme.type
    except Exception:  # noqa: BLE001
        theme_type = None
    return "dark" if theme_type == "dark" else "light"


# 模块级令牌：每次脚本运行解析一次，主题切换会触发 rerun 从而重新解析。
# 下面的函数在调用时读取这些全局量，所以不需要把 T 传进去。
THEME_TYPE = active_theme_type()
T = THEMES[THEME_TYPE]

UP = T["up"]          # 涨 / 正收益（A股习惯：红涨）
DOWN = T["down"]      # 跌 / 负收益（绿跌）
ACCENT = T["accent"]
INK = T["ink"]
MUTED = T["muted"]
LINE = T["line"]
GRID = T["grid"]
# 月度收益热力图：A 股习惯「红涨绿跌」，故与 plotly 的 RdYlGn 相反；
# 中点跟随主题底色，否则浅色中点会在暗色页面上糊成一块白斑。
HEAT_SCALE = [[0.0, T["down"]], [0.5, T["heat_mid"]], [1.0, T["up"]]]


def _token_block(tokens: dict, selector: str) -> str:
    """把一套令牌写成某个选择器下的 CSS 变量声明。"""
    kebab = {"tab_fg": "tab-fg", "code_fg": "code-fg",
             "accent_soft": "accent-soft", "accent_soft2": "accent-soft2"}
    decls = []
    for key, value in tokens.items():
        if key in ("plotly_template", "heat_mid"):
            continue
        decls.append(f"--{kebab.get(key, key).replace('_', '-')}: {value};")
    return f"{selector} {{ --radius: 12px; {' '.join(decls)} }}"


def _root_css() -> str:
    """令牌层，三段：

    1. 裸 ``:root``：服务端按 ``st.context.theme.type`` 给的初判（首次加载即正确）；
    2. ``:root[data-dsh-theme="light"|"dark"]``：由客户端探针（``THEME_PROBE_JS``）
       按**实际渲染出来的底色**打上的属性。

    为什么要有第 2、3 段：在设置菜单里点切换主题时，当次 rerun 传回来的
    ``color_scheme`` 还是旧值（streamlit#11920），服务端初判会滞后一拍 ——
    表现为「点 Dark 后深底配白卡片、标题文字压暗底」，必须刷新一次才恢复。
    属性选择器特异性(0,2,0)高于裸 ``:root``(0,1,0)，探针一跑就覆盖初判，
    所以切换当次就正确，不需要刷新、也不需要等下一次 rerun。
    三段色值都来自同一个 THEMES，没有手工重复。
    """
    return "".join([
        _token_block(THEMES[THEME_TYPE], ":root"),
        _token_block(THEMES["light"], ':root[data-dsh-theme="light"]'),
        _token_block(THEMES["dark"], ':root[data-dsh-theme="dark"]'),
    ])


# 客户端主题探针。为什么要它：Streamlit 不暴露任何 CSS 变量，documentElement 上
# 既没有 data-theme 也没有 class，服务端 st.context.theme.type 在"刚点完切换"的
# 那一次 rerun 里又是旧值 —— 纯 CSS 与纯服务端都判断不出当前到底是什么主题。
# 只能由客户端读**实际计算出来的底色**来判断，再把结果写到 documentElement 上，
# 让上面的 [data-dsh-theme=...] 规则接管。
#
# 注入方式实测（本机 Chromium 140 + streamlit 1.61.1，scratch 应用 A/B 过）：
#   * st.markdown 里的 <script> / <img onerror> / <svg onload> **全部被过滤、不执行**；
#   * st.components.v1.html 的 iframe 与主页面同源，脚本能拿到 window.parent.document —— 可用。
#   该 iframe 高度为 0，不占版面。
THEME_PROBE_JS = r"""
<script>
(function () {
  var doc = window.parent.document, root = doc.documentElement;
  function readLum() {
    // 从容器往上找【第一个有实际不透明底色】的元素再算亮度。
    // 坑（2026-09-16 实测）：[data-testid="stAppViewContainer"] 的 background 是
    // rgba(0,0,0,0)（全透明），直接读它会把 (0,0,0) 当成"暗色"，于是**恒判 dark** ——
    // 表现为浅色系统下首屏 body=#f5f7fa、配色却是暗的（页面浅、卡片暗）。
    // 实测真正承载主题底色的是 doc.body（浅 rgb(245,247,250) / 暗 rgb(13,17,23)）。
    var n = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
    while (n && n.nodeType === 1) {
      var m = getComputedStyle(n).backgroundColor.match(/[\d.]+/g);
      if (m) {
        var a = m.length >= 4 ? parseFloat(m[3]) : 1;   // rgb(...) 无 alpha 视为不透明
        if (a > 0) {
          return (0.2126 * +m[0] + 0.7152 * +m[1] + 0.0722 * +m[2]) / 255;
        }
      }
      n = n.parentElement;   // 全透明就继续往上找
    }
    return null;
  }
  function detect() {
    var lum = readLum();
    // 读不到（全透明链走到底）就不要下结论，保持服务端给的值 —— 宁可不动手，
    // 也不要用一个猜出来的值覆盖正确的首屏配色。
    if (lum === null) return null;
    return lum < 0.5 ? "dark" : "light";
  }
  function apply() {
    var t = detect();
    if (t && root.getAttribute("data-dsh-theme") !== t) {
      root.setAttribute("data-dsh-theme", t);
    }
  }
  apply();
  try { new MutationObserver(apply).observe(root, {attributes: true}); } catch (e) {}
  try { new MutationObserver(apply).observe(doc.body, {attributes: true}); } catch (e) {}
  setInterval(apply, 400);  // 轮询兜底：一次 getComputedStyle，开销可忽略
})();
</script>
"""

# ---------------------------------------------------------------- UI 组件


def html_block(markup: str) -> None:
    """渲染一段自绘 HTML（调用方负责转义用户数据）。"""
    st.markdown(markup, unsafe_allow_html=True)


def inject_css() -> None:
    html_block(f"<style>{_root_css()}{_CSS}</style>")


def inject_theme_probe() -> None:
    """注入客户端主题探针（零高度 iframe，不占版面、不影响布局）。

    ⚠️ **已弃用 API 的降级路径（2026-09-18 补）**：streamlit 1.61.1 起对
    `st.components.v1.html` 报弃用（提示改用 `st.iframe`），而 `st.iframe` 只接受 URL、
    `st.html` 又不执行 `<script>` —— 本探针**没有等价替代**（其必要性见上面的实测注释）。
    所以这里"能用就用、不能用就安静跳过"：将来某个版本真把它删掉时，
    页面退化成"服务端 `st.context.theme.type` 初判 + 手动切换时的一次性滞后"，
    而不是整个仪表盘抛异常打不开。**已弃用 API 不能带走整个页面。**
    """
    try:
        components.html(THEME_PROBE_JS, height=0)
    except Exception:  # noqa: BLE001 - 见 docstring：弃用/移除都不能让页面崩
        pass


# 组件样式。只命中 Streamlit 公开的 data-testid，避免依赖易变的 emotion 类名。
# 颜色一律走 var(--...)（见上面的令牌层），不要在下面写死的十六进制色值 ——
# 写死就只在浅色下正确，深色下会变成"白底黑字漂在暗页上"。
_CSS = """
/* ---- 页面骨架：收窄留白，卡片化 ---- */
.stMainBlockContainer { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
[data-testid="stHeader"] { background: transparent; }

/* ---- 顶部标题区 ---- */
.app-hero { display:flex; flex-wrap:wrap; align-items:baseline; gap:.7rem;
  padding-bottom:.55rem; border-bottom:1px solid var(--line); margin-bottom:.9rem; }
.app-hero h1 { font-size:1.62rem; font-weight:750; letter-spacing:-.01em;
  color:var(--ink); margin:0; }
.app-hero .app-sub { color:var(--muted); font-size:.85rem; }
.pill-row { display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin:.1rem 0 .85rem; }
.pill { display:inline-flex; align-items:center; gap:.32rem; font-size:.76rem;
  font-weight:600; line-height:1; padding:.36rem .6rem; border-radius:999px;
  border:1px solid var(--line); background:var(--card); color:var(--muted);
  white-space:nowrap; }
.pill b { color:var(--ink); font-weight:700; }
.pill-ok     { border-color:var(--ok-line);   background:var(--ok-bg);   color:var(--ok-fg); }
.pill-warn   { border-color:var(--warn-line); background:var(--warn-bg); color:var(--warn-fg); }
.pill-bad    { border-color:var(--bad-line);  background:var(--bad-bg);  color:var(--bad-fg); }
.pill-info   { border-color:var(--info-line); background:var(--info-bg); color:var(--info-fg); }
.pill-accent { border-color:var(--info-line); background:var(--info-bg); color:var(--accent); }

/* ---- 区块标题 ---- */
h3 { letter-spacing:-.005em; }
[data-testid="stHeading"] h3 { font-size:1.12rem; font-weight:700; color:var(--ink);
  padding-left:.6rem; border-left:3px solid var(--accent); line-height:1.25; }
[data-testid="stCaptionContainer"] p { color:var(--muted); }

/* ---- KPI 卡片 ---- */
.kpi-grid { display:grid; grid-template-columns:repeat(var(--n,4),minmax(0,1fr));
  gap:.7rem; margin:.15rem 0 .9rem; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
  box-shadow:var(--shadow); padding:.8rem .95rem; min-width:0; }
.kpi-label { font-size:.78rem; font-weight:650; color:var(--muted); margin-bottom:.32rem;
  display:flex; align-items:center; gap:.25rem; }
.kpi-label .q { display:inline-flex; align-items:center; justify-content:center;
  width:.95rem; height:.95rem; border-radius:50%; border:1px solid var(--line);
  font-size:.62rem; color:var(--muted); cursor:help; }
.kpi-value { font-size:1.55rem; font-weight:750; color:var(--ink); line-height:1.16;
  font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }
.kpi-value.up   { color:var(--up); }
.kpi-value.down { color:var(--down); }
.kpi-unit { font-size:.8rem; font-weight:600; color:var(--muted); margin-left:.18rem; }
.kpi-delta { font-size:.76rem; color:var(--muted); margin-top:.3rem;
  font-variant-numeric:tabular-nums; }
.kpi-delta.up   { color:var(--up); }
.kpi-delta.down { color:var(--down); }
@media (max-width: 1200px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }

/* ---- 空状态 ---- */
.empty { border:1px dashed var(--empty-line); border-radius:var(--radius);
  background:var(--empty-bg); padding:1.05rem 1.15rem; margin:.2rem 0 .9rem; }
.empty-t { font-weight:700; color:var(--ink); font-size:.95rem; margin-bottom:.28rem; }
.empty-b { color:var(--muted); font-size:.86rem; line-height:1.6; }

/* ---- 标签页：药丸式，选中态更明确 ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.2rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] button[role="tab"] { padding:.5rem .92rem; font-weight:650;
  font-size:.9rem; color:var(--tab-fg); border-radius:9px 9px 0 0; }
[data-testid="stTabs"] button[role="tab"]:hover { background:var(--accent-soft2); color:var(--accent); }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:var(--accent);
  background:var(--accent-soft); }
[data-baseweb="tab-highlight"] { background-color:var(--accent); height:2px; }
[data-baseweb="tab-border"] { background-color:var(--line); }

/* ---- 提示条：左侧色条 + 圆角 ---- */
[data-testid="stAlertContainer"], [data-testid="stAlert"] { border-radius:10px;
  border-left-width:3px; }
[data-testid="stAlertContainer"] p { font-size:.87rem; }

/* ---- 表格 / 图表容器 ---- */
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px;
  overflow:hidden; }
[data-testid="stPlotlyChart"] { border:1px solid var(--line); border-radius:var(--radius);
  background:var(--card); padding:.35rem .4rem .1rem; box-shadow:var(--shadow); }

/* ---- 按钮 ---- */
[data-testid="stBaseButton-primary"] { font-weight:700; box-shadow:0 1px 2px rgba(29,78,216,.25); }
[data-testid="stBaseButton-secondary"] { font-weight:600; background:var(--card); }

/* ---- 侧边栏 ---- */
[data-testid="stSidebarUserContent"] { padding-top:1.1rem; }
.side-sec { font-size:.72rem; font-weight:750; letter-spacing:.08em; color:var(--muted);
  text-transform:uppercase; margin:.35rem 0 .45rem; }
.side-card { background:var(--soft); border:1px solid var(--line); border-radius:10px;
  padding:.6rem .7rem; font-size:.78rem; color:var(--muted); line-height:1.75; }
.side-card b { color:var(--ink); font-weight:700; }
.side-card code { font-size:.72rem; color:var(--code-fg); }
"""

# ---------------------------------------------------------------- UI 组件


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def pill_row(items: list[tuple]) -> None:
    """一行状态药丸；items 为 (标签, 值, tone)，值为空串时只显示标签。

    标签与值分别转义后再拼 <b>，所以调用方传纯文本即可（不要自己拼 HTML）。
    tone: ok / warn / bad / info / accent（空串为中性灰）。
    """
    parts = []
    for label, value, tone in items:
        cls = f"pill pill-{tone}" if tone else "pill"
        body = f"{esc(label)} <b>{esc(value)}</b>" if value else esc(label)
        parts.append(f'<span class="{cls}">{body}</span>')
    html_block(f'<div class="pill-row">{"".join(parts)}</div>')


def hero(title: str, subtitle: str, pills: list[tuple] | None = None) -> None:
    """应用级标题区（标题 + 一句话定位 + 全局状态药丸）。"""
    html_block(
        f'<div class="app-hero"><h1>{esc(title)}</h1>'
        f'<span class="app-sub">{esc(subtitle)}</span></div>'
    )
    if pills:
        pill_row(pills)


def page_header(title: str, subtitle: str) -> None:
    """每个标签页开头的页面标题（标题 + 一句话用途说明）。"""
    html_block(
        f'<div style="margin:.2rem 0 .7rem">'
        f'<div style="font-size:1.28rem;font-weight:750;color:var(--ink);'
        f'letter-spacing:-.01em">{esc(title)}</div>'
        f'<div style="color:var(--muted);font-size:.86rem;margin-top:.22rem">'
        f'{esc(subtitle)}</div></div>'
    )


def empty_state(title: str, body: str) -> None:
    """统一的空状态卡片（替代裸 st.info，信息密度更高）。"""
    html_block(f'<div class="empty"><div class="empty-t">{esc(title)}</div>'
               f'<div class="empty-b">{esc(body)}</div></div>')


def tone_of(value) -> str:
    """由数值正负得到 KPI 方向色类名（A股：正=红，负=绿）。"""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(num) or num == 0:
        return ""
    return "up" if num > 0 else "down"


def kpi(label: str, value: str, *, unit: str = "", delta: str = "",
        tone: str = "", delta_tone: str = "", help_text: str = "") -> str:
    """单张 KPI 卡片的 HTML；value/delta 由调用方格式化后再传入。

    tone/delta_tone 取 "up" / "down" / ""（"" 表示中性不着色）。
    """
    q = (f'<span class="q" title="{esc(help_text)}">?</span>' if help_text else "")
    unit_html = f'<span class="kpi-unit">{esc(unit)}</span>' if unit else ""
    delta_html = (f'<div class="kpi-delta {delta_tone}">{esc(delta)}</div>'
                  if delta else "")
    return (f'<div class="kpi"><div class="kpi-label">{esc(label)}{q}</div>'
            f'<div class="kpi-value {tone}">{esc(value)}{unit_html}</div>'
            f'{delta_html}</div>')


def kpi_row(items: list[dict]) -> None:
    """一行等宽 KPI 卡片；每项为 kpi() 的关键字参数。"""
    if not items:
        return
    cards = "".join(kpi(**item) for item in items)
    html_block(f'<div class="kpi-grid" style="--n:{len(items)}">{cards}</div>')


def style_fig(fig: go.Figure, *, height: int | None = None,
              hovermode: str = "x unified", showlegend: bool | None = None,
              legend_pos: str = "bottom", margin_top: int = 46) -> go.Figure:
    """统一图表观感：透明底、细网格、横排图例、悬停统一。

    注意四点（前三条都踩过）：
      1. 轴标题一律不放中文（plotly 会把它竖排，中文会糊成一团），单位写进 title；
      2. 图例默认放**下方**（legend_pos="bottom"）。放上方时 plotly 把标题在
         margin 里垂直居中、图例又紧贴绘图区顶端，标题一长就会和图例叠字；
         图例条数不定时（如 20 条策略线会折行）根本算不准要留多少 px。
         放下方一劳永逸，不依赖条数与标题长度。
      3. 图例放上方时（legend_pos="top"）必须给足 margin_top，否则必叠。
      4. **图表的「底色/字色/网格色/坐标轴线色」一律不在这里写死**，交给
         plotly_chart(theme="streamlit") 由前端按**当前真实主题**套模板 ——
         原因同自绘组件：服务端的 st.context.theme.type 在刚点完主题切换的
         那一次 rerun 里是旧值，若在这里写死颜色，切换当次图表就会出现
         "深字压暗底"。只保留与主题无关的：尺寸/边距/图例位置/悬停模式。
    """
    if legend_pos == "bottom":
        legend = dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0,
                      title=None, bgcolor="rgba(0,0,0,0)")
        margin = dict(l=6, r=10, t=margin_top, b=72)
    else:
        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                      title=None, bgcolor="rgba(0,0,0,0)")
        margin = dict(l=6, r=10, t=max(margin_top, 72), b=6)
    fig.update_layout(
        colorway=CHART_COLORS,
        font=dict(family=FONT_STACK, size=12),
        margin=margin,
        hovermode=hovermode,
        legend=legend,
        title=dict(font=dict(size=14), x=0, xanchor="left"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False,
                     ticks="outside", tickfont=dict(size=11))
    fig.update_yaxes(zeroline=False, tickfont=dict(size=11))
    if height is not None:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


def chart(fig: go.Figure, *, key: str | None = None) -> None:
    """统一渲染 plotly 图表：套 Streamlit 主题模板 + 去掉多余工具条按钮。

    theme="streamlit" 让**前端**按当前真实主题给图表上色（底/字/网格/坐标轴），
    因此刚点完主题切换的那一次也能立刻正确 —— 不受服务端 theme 值滞后影响。
    """
    st.plotly_chart(fig, width="stretch", key=key, config=PLOTLY_CONFIG,
                    theme="streamlit")


def local_close_table(data_dir: Path, close_panel, symbols: list[str],
                      decision_date: str):
    """实时快照不可用时的兜底表：本地最新收盘 vs 决策日收盘（明确非实时）。

    实时行情接口在非交易时段/限流时会返回空，此时页面不该只剩一块黄色警告——
    用本地日线给出「最新可得」的持仓快照，并标清数据日期。
    """
    if close_panel is None or len(close_panel) == 0:
        return None
    last_date = close_panel.index.max()
    last = close_panel.iloc[-1].reindex(symbols)
    out = pd.DataFrame({"代码": [str(s) for s in symbols],
                        "最新收盘": last.to_numpy(),
                        "数据日期": str(last_date.date())})
    d0 = pd.Timestamp(decision_date)
    if d0 in close_panel.index:
        base = close_panel.loc[d0].reindex(symbols)
        chg = last.to_numpy() / base.to_numpy() - 1
        out["决策日收盘"] = base.to_numpy()
        out["自决策日涨跌"] = chg
        out["自决策日涨跌"] = out["自决策日涨跌"].map(
            lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
        out["决策日收盘"] = out["决策日收盘"].map(
            lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    out["最新收盘"] = out["最新收盘"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    st.caption("本地日线口径（非实时）：最新收盘为本地缓存中最新一个交易日的收盘价。")
    return out


def _weight_transition(entry: dict, strategy: str) -> str:
    """把一条调整记录里的 before/after 权重渲染成「前 → 后 (变化)」。"""
    before = (entry.get("before") or {}).get(strategy)
    after = (entry.get("after") or {}).get(strategy)
    if before is None and after is None:
        return "—"
    try:
        b_txt = f"{float(before):.1%}" if before is not None else "—"
        a_txt = f"{float(after):.1%}" if after is not None else "—"
    except (TypeError, ValueError):
        return f"{before} → {after}"
    if before is not None and after is not None:
        try:
            delta = float(after) - float(before)
            return f"{b_txt} → {a_txt} ({delta:+.1%})"
        except (TypeError, ValueError):
            pass
    return f"{b_txt} → {a_txt}"


# ---------------------------------------------------------------- 名称映射

# 模型 / 算法 / 基准的中文名称映射
MODEL_NAMES = {
    "momentum": "动量(60日)",
    "reversal": "反转(60日)",
    "lowvol": "低波(60日)",
    "multifactor": "多因子",
    "rotation": "模型轮动",
    "momentum20": "动量(20日)",
    "reversal120": "反转(120日)",
    "lowvol20": "低波(20日)",
    "linear": "线性回归",
    "rf": "随机森林",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rank_lgb": "LightGBM排序(LambdaRank)",
    "rank_xgb": "XGBoost排序(pairwise)",
    "rank_ensemble": "排序融合(截面排名均值)",
    "agreement_ensemble": "一致性排序融合",
    "ic_rank_ensemble": "IC加权排序融合",
    "temporal_decay_lgb": "LGBM时间衰减",
    "risk_aware_lgb": "LGBM风险调整",
    "huber_lgb": "LightGBM(Huber)",
    "mlp_deep": "神经网络MLP(深层)",
    "pls": "偏最小二乘PLS",
    "enet": "弹性网ElasticNet",
    "histgb": "梯度提升(HistGB)",
    "svm": "支持向量机",
    "knn": "K近邻",
    "mlp": "神经网络(MLP)",
    "ensemble": "集成(LGBM+HistGB+RF)",
    "conformal": "LGBM+保形门控",
    "regime": "状态路由",
    "ic_adaptive": "IC自适应加权",
    "gru": "GRU深度模型",
    "benchmark_等权全市场": "基准·等权全市场",
    "benchmark_沪深300": "基准·沪深300",
    "ensemble(lgbm+histgb+rf)": "集成(LGBM+HistGB+RF)",
    "lgbm+conformal_gate": "LGBM+保形门控",
    "regime_routing": "状态路由",
    "ic_adaptive_weights": "IC自适应加权",
    "基准·等权全市场": "基准·等权全市场",
    "基准·沪深300": "基准·沪深300",
}

METRIC_NAMES = {
    "model": "模型",
    "annual_return": "年化收益",
    "annual_vol": "年化波动",
    "sharpe": "夏普",
    "max_drawdown": "最大回撤",
    "win_rate": "胜率",
    "profit_loss_ratio": "盈亏比",
    "calmar": "Calmar",
    "mean_ic": "平均IC",
    "n_periods": "期数",
}

PICK_NAMES = {"symbol": "代码", "score": "预期收益(20日)", "weight": "权重",
              "rank_score": "排名得分"}


def display_name(name: str) -> str:
    """列名 -> 中文展示名；已是中文或无法映射时原样返回。"""
    return MODEL_NAMES.get(str(name), str(name))


def format_metric(df: pd.DataFrame) -> pd.DataFrame:
    """指标表汉化并格式化（收益/回撤/胜率/波动转百分比，比率保留小数）。"""
    out = df.rename(columns=METRIC_NAMES)
    for col in ("年化收益", "最大回撤", "胜率", "年化波动"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "-")
    for col in ("夏普", "平均IC"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    for col in ("盈亏比", "Calmar"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    if "期数" in out.columns:
        out["期数"] = out["期数"].map(lambda v: f"{int(v)}" if pd.notna(v) else "-")
    if "模型" in out.columns:
        out["模型"] = out["模型"].map(display_name)
    return out


# ---------------------------------------------------------------- 数据读取
# 约定：st.cache_data 一律设 TTL，且不得在 cache_data 函数内调用另一个
# cache_data 函数（跨刷新会抛 KeyError，见 AGENTS.md 的 load_decision 教训）。


@st.cache_data(ttl=60)
def load_csv(path: Path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data(ttl=60)
def load_json(path: Path):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def load_history_cached(path: Path) -> list[dict]:
    """账户历史（jsonl，约 0.5MB）缓存读取；load_history 本身是普通函数。"""
    return load_history(path)


def load_decision(sim_dir: Path) -> dict | None:
    """读取最新决策：兼容多个输出目录（计划任务历史写 docs/simulation）。

    普通函数（非 st.cache_data）：内部直接读文件，避免嵌套缓存调用——
    Streamlit 的 cache_data 嵌套在跨刷新读取时可能抛 KeyError。
    """
    candidates = [
        sim_dir / "decision.json",
        PROJECT / "docs" / "simulation" / "decision.json",
        PROJECT / "docs" / "decision" / "decision.json",
    ]
    best = None
    for p in candidates:
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if d and (best is None or str(d.get("date", "")) > str(best.get("date", ""))):
            best = d
    return best


@st.cache_data(ttl=60)
def load_panel_close(data_dir: Path):
    """读取面板缓存中的收盘价矩阵（date × symbol）。"""
    p = data_dir / "panels" / "close.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(ttl=5)
def _cached_snapshot(symbols: tuple) -> pd.DataFrame:
    """同一页面运行内共享同一份快照（5 秒 TTL），避免多处显示不一致。"""
    return snapshot(list(symbols))


@st.cache_data(ttl=60)
def load_symbol(data_dir: Path, code: str):
    """读取单只股票本地日线（K线详情用，秒级）。"""
    p = data_dir / f"{code}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(ttl=60)
def auto_update_status() -> str:
    """读取 Windows 计划任务状态（只读）。"""
    import subprocess
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-ScheduledTask -TaskName 'AshareQuantDaily').State"],
            capture_output=True, text=True, timeout=10)
        state = r.stdout.strip()
        return "开启" if state == "Ready" else ("关闭" if state == "Disabled" else state or "未注册")
    except Exception:  # noqa: BLE001
        return "未知"


# ---------------------------------------------------------------- 数值格式


def format_pct_nan(v) -> str:
    """NaN 显示为 —，否则显示带符号百分比（收益类指标用）。"""
    return "—" if pd.isna(v) else f"{v:+.2%}"


def format_pct(v) -> str:
    """NaN 显示为 —，否则显示无符号百分比（波动/胜率等无方向指标用）。"""
    return "—" if pd.isna(v) else f"{v:.2%}"


def format_num(v, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):,.{digits}f}"


def money(v, digits: int = 0) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):,.{digits}f}"


def signed_money(v, digits: int = 0) -> str:
    """带符号金额；恰好为 0 时显示 0 而不是 +0 / -0（后者看着像 bug）。"""
    if v is None or pd.isna(v):
        return "—"
    num = float(v)
    if num == 0:
        return f"{0:,.{digits}f}"
    return f"{num:+,.{digits}f}"


def pct_cell(v, digits: int = 2) -> str:
    """表格里的百分比单元格（带正负号，缺失显示 —）。"""
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):+.{digits}%}"


# ---------------------------------------------------------------- 业务图表


def equal_weight_bench(close_panel, start, end=None) -> tuple[pd.Series, int]:
    """等权全市场基准序列（口径修正：剔除成分股覆盖不足的交易日）。

    原实现直接按日取面板的非空均值。当某天只更新了少数股票时（2026-09-16
    实测仅 209/5360 只，当日均值 12.01 元 vs 前一交易日 28.27 元），等权基准
    会凭空出现 -57% 的假暴跌，账户页的对比基准与「同期收益对比」全被带偏。
    这里要求当日有效成分股占比 >= BENCH_MIN_COVERAGE 才纳入基准。

    返回 (基准序列（未归一）, 被剔除的交易日数)。
    """
    if close_panel is None or len(close_panel) == 0:
        return pd.Series(dtype=float), 0
    window = close_panel.loc[start:end] if end is not None else close_panel.loc[start:]
    if window.empty:
        return pd.Series(dtype=float), 0
    n_total = max(1, int(close_panel.shape[1]))
    coverage = window.notna().sum(axis=1) / n_total
    kept = window[coverage >= BENCH_MIN_COVERAGE]
    dropped = int(len(window) - len(kept))
    if kept.empty:
        return pd.Series(dtype=float), dropped
    return kept.mean(axis=1), dropped


def account_figure(equity: pd.Series, data_dir: Path,
                   close_panel, capital: float) -> go.Figure:
    """账户净值曲线 + 真实基准（沪深300 / 等权全市场）。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines",
                             name="账户净值", line=dict(color=ACCENT, width=2.4),
                             hovertemplate="账户净值 %{y:,.0f} 元<extra></extra>"))
    start = equity.index[0]
    idx_path = data_dir / "sh000300.parquet"
    if idx_path.exists():
        idx_close = pd.read_parquet(idx_path)["close"]
        idx_sel = idx_close.loc[start:]
        if len(idx_sel) >= 2:
            bench = capital * idx_sel / idx_sel.iloc[0]
            fig.add_trace(go.Scatter(x=bench.index, y=bench, name="基准·沪深300",
                                     line=dict(dash="dash", color=T["bench1"], width=1.6),
                                     hovertemplate="沪深300 %{y:,.0f}<extra></extra>"))
    bench_eq, _ = equal_weight_bench(close_panel, start)
    if len(bench_eq) >= 2:
        bench2 = capital * bench_eq / bench_eq.iloc[0]
        fig.add_trace(go.Scatter(x=bench2.index, y=bench2, name="基准·等权全市场",
                                 line=dict(dash="dot", color=T["bench2"], width=1.6),
                                 hovertemplate="等权全市场 %{y:,.0f}<extra></extra>"))
    fig.update_layout(title="账户净值曲线（元，虚线为真实基准）", hovermode="x unified")
    fig.update_yaxes(tickformat=",.0f")
    return style_fig(fig, height=380)


def equity_figure(returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    equity = (1 + returns.fillna(0)).cumprod()
    for col in equity.columns:
        name = display_name(col)
        is_bench = str(col).startswith("基准") or str(col).startswith("benchmark_")
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity[col], mode="lines", name=name,
            line=dict(dash="dash" if is_bench else "solid",
                      width=1.6 if is_bench else 2.0)))
    fig.update_layout(title="策略净值曲线（模拟，含真实基准）", hovermode="x unified")
    return style_fig(fig, height=380)


def kline_figure(tail: pd.DataFrame, name: str, title: str,
                 height: int = 520) -> go.Figure:
    """K线 + 均线 + 成交量（A股红涨绿跌）。"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.74, 0.26])
    fig.add_trace(go.Candlestick(
        x=tail.index, open=tail["open"], high=tail["high"],
        low=tail["low"], close=tail["close"], name=name,
        increasing_line_color=UP, increasing_fillcolor=UP,
        decreasing_line_color=DOWN, decreasing_fillcolor=DOWN,
        line=dict(width=1)), row=1, col=1)
    for n, color in ((5, T["ma1"]), (20, T["ma2"]), (60, T["ma3"])):
        fig.add_trace(go.Scatter(x=tail.index, y=tail["close"].rolling(n).mean(),
                                 name=f"MA{n}", line=dict(width=1.2, color=color)),
                      row=1, col=1)
    if "volume" in tail.columns:
        up_mask = tail["close"] >= tail["open"]
        fig.add_trace(go.Bar(x=tail.index, y=tail["volume"], name="成交量",
                             marker_color=[UP if u else DOWN for u in up_mask],
                             opacity=.55), row=2, col=1)
    fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False,
                      hovermode="x unified")
    # 用默认 margin_top（58）：K线也是「标题 + 横排图例」两层，留白给小了会叠字
    style_fig(fig, height=height)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ---------------------------------------------------------------- 实时估值


def render_realtime_valuation(decision: dict, data_dir: Path, key: str,
                              capital: float | None = None) -> None:
    """盘中实时估值（快照价）：开关开启才拉行情，避免每次刷新变慢。"""
    if decision is None or not decision.get("picks"):
        return
    if not st.toggle("盘中实时估值（快照价，秒级）", value=False, key=key):
        return
    try:
        initial = float(capital) if capital is not None \
            else float(decision.get("initial_capital", 100000.0))
        pick_syms = [p["symbol"] for p in decision["picks"]]
        snap = _cached_snapshot(tuple(pick_syms))
        if snap.empty:
            st.warning("未获取到实时行情（可能非交易时段或接口限流）。")
            return
        close_panel = load_panel_close(data_dir)
        # 以决策日累计净资产为基准：否则每个决策日实时总资产会重置回初始值，
        # 与账户页累计净值（98,599 而非 100,000）对不上
        history_path = data_dir / "portfolio" / "account_history.jsonl"
        hist_all = load_history_cached(history_path) if history_path.exists() else []
        hist_live = [e for e in hist_all if e.get("mode") == "live"] or hist_all
        basis = account_basis(hist_live, close_panel, decision["date"], initial,
                              *account_params()) \
            if close_panel is not None else initial
        d0 = pd.Timestamp(decision["date"])
        prices = snap.set_index("代码")["现价"].astype(float)
        fallback = close_panel.iloc[-1].reindex(pick_syms)
        if close_panel is not None and d0 in close_panel.index:
            fallback = fallback.fillna(close_panel.loc[d0, pick_syms])
        prices = prices.reindex(fallback.index).fillna(fallback)
        dec_view = {**decision, "initial_capital": basis}
        acc = account_snapshot(dec_view, close_panel, prices=prices)
        if acc is None:
            d0 = pd.Timestamp(decision["date"])
            last = close_panel.index.max() if close_panel is not None else None
            hint = ""
            if last is not None and d0 > last:
                hint = (f"（决策日期 {d0.date()} 晚于本地数据 {last.date()}——"
                        "多半是盘中点了「每日更新」把决策日期提前了，"
                        "收盘后 16:05 自动更新即恢复）")
            st.warning(f"无法按实时价估值（缺少决策日基准）。{hint}")
            return
        total_ret = acc["total_asset"] / initial - 1
        kpi_row([
            dict(label="实时总资产", value=money(acc["total_asset"]), unit="元",
                 delta=f"初始资金 {money(initial)} 元"),
            dict(label="实时总收益（累计）", value=f"{total_ret:+.2%}",
                 tone=tone_of(total_ret), delta="总资产 / 初始资金 − 1"),
            dict(label="本期浮动盈亏", value=signed_money(acc["total_pnl"]), unit="元",
                 tone=tone_of(acc["total_pnl"]),
                 delta="现价相对决策日成本"),
            dict(label="现金余额", value=money(acc["cash"]), unit="元",
                 delta="按决策日权重等权建仓后的剩余现金"),
        ])
        st.caption(f"基准：决策日 {decision['date']} 累计净资产 {basis:,.0f} 元"
                   "（自首个决策日跟踪）× 快照现价逐只估值；实时总收益=总资产/初始资金-1，"
                   "与「账户」页累计口径一致。缺失行情用本地最新收盘价兜底，"
                   "收盘后现价=收盘价、本期浮动盈亏为 0。")
    except Exception as e:  # noqa: BLE001
        st.error(f"实时估值获取失败：{e}")


# ---------------------------------------------------------------- 后台任务
# 在页面内直接跑数据下载/更新，实时回显输出


def _task_queue(key: str) -> queue.Queue:
    """任务队列存 session_state（跨页面刷新持久，后台线程与渲染共享）。"""
    st.session_state.setdefault("task_queues", {})
    return st.session_state["task_queues"].setdefault(key, queue.Queue())


def _task_worker(key: str, cmd: list[str], cwd: Path, q: queue.Queue) -> None:
    log_dir = cwd / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_dir / f"{key}.log", "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            log_f.write(line + "\n")
            log_f.flush()
            q.put(("line", line.rstrip()))
        proc.wait()
        q.put(("done", proc.returncode))
    except Exception as e:  # noqa: BLE001
        q.put(("line", f"启动失败：{e}"))
        q.put(("done", -1))
    finally:
        log_f.close()


def launch_task(key: str, cmd: list[str], cwd: Path) -> None:
    """启动一个后台任务（幂等：同一 key 运行中不重复启动）。"""
    state = st.session_state.setdefault("task_state", {}).setdefault(
        key, {"lines": [], "running": False, "code": None})
    if state["running"]:
        return
    q = _task_queue(key)
    while not q.empty():  # 清空上次遗留
        try:
            q.get_nowait()
        except queue.Empty:
            break
    state.update({"lines": [], "running": True, "code": None})
    threading.Thread(target=_task_worker, args=(key, cmd, cwd, q), daemon=True).start()


def render_task(key: str, title: str) -> bool:
    """渲染任务进度；返回是否仍在运行。"""
    state = st.session_state.setdefault("task_state", {}).setdefault(
        key, {"lines": [], "running": False, "code": None})
    q = st.session_state.get("task_queues", {}).get(key)
    if q is not None:
        while True:
            try:
                kind, payload = q.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                state["lines"].append(payload)
            elif kind == "done":
                state["running"] = False
                state["code"] = payload
    if state["running"]:
        with st.status(f"{title} 进行中…", expanded=True):
            tail = state["lines"][-30:]
            st.code("\n".join(tail) if tail else "等待输出…（长任务请耐心等待）")
        # 任务运行中每 5 秒自动刷新页面，实时显示进度
        st_autorefresh(interval=5000, key=f"task_refresh_{key}")
        return True
    if state["code"] == 0:
        st.success(f"{title} 完成")
    elif state["code"] is not None:
        st.error(f"{title} 失败（退出码 {state['code']}）")
    if state["lines"]:
        with st.expander("查看完整输出"):
            st.code("\n".join(state["lines"]))
    return False


# ---------------------------------------------------------------- 表格


def picks_column_config(view: pd.DataFrame) -> dict:
    """决策/持仓表列配置（数值保持原值，只改展示与排序体验）。"""
    cfg = {}
    if "代码" in view.columns:
        cfg["代码"] = st.column_config.TextColumn(
            "代码", width="small", help="A 股代码（6 位）")
    if "排名得分" in view.columns:
        cfg["排名得分"] = st.column_config.NumberColumn(
            "排名得分", format="%.4f",
            help="各模型预测转为当日横截面百分位排名后的均值，越高越靠前（本期排序依据）")
    if "预期收益(20日)" in view.columns:
        cfg["预期收益(20日)"] = st.column_config.NumberColumn(
            "预期收益(20日)", format="percent",
            help="各模型预测的中位数（展示用），不是排序依据")
    if "权重" in view.columns:
        cfg["权重"] = st.column_config.NumberColumn(
            "权重", format="percent", help="本期等权建仓权重")
    return cfg


def picks_table(decision: dict, *, head: int | None = None) -> None:
    """今日模拟持仓表：只展示可读列（model_scores 明细在「模型预测明细」里展开）。"""
    picks = pd.DataFrame(decision["picks"]).rename(columns=PICK_NAMES)
    cols = [c for c in ("代码", "排名得分", "预期收益(20日)", "权重")
            if c in picks.columns]
    view = picks[cols]
    if head is not None:
        view = view.head(head)
    view = view.copy()
    if "代码" in view.columns:
        view["代码"] = view["代码"].astype(str)
    st.dataframe(view, width="stretch", hide_index=True,
                 column_config=picks_column_config(view))


# ---------------------------------------------------------------- 页面装配

st.set_page_config(page_title="A股量化研究·模拟分析控制台", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
inject_theme_probe()

with st.sidebar:
    html_block('<div class="side-sec">数据范围</div>')
    data_root = st.selectbox("数据集", ["data/tencent", "data/all", "data/3y"], index=0)
    mode = "all" if data_root in ("data/tencent", "data/all") else "3y"
    sim_dir = PROJECT / ("docs/simulation-all" if mode == "all" else "docs/simulation")
    data_dir = PROJECT / data_root
    model_dir = PROJECT / "models" / mode
    st.caption(f"数据目录：{data_root}")
    st.divider()
    cfg_path = PROJECT / "config.yaml"
    cfg_d = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) \
        if cfg_path.exists() else {}
    html_block('<div class="side-sec">模拟参数</div>')
    capital = st.number_input(
        "初始资金（元）", min_value=10000.0, max_value=100000000.0,
        value=float(cfg_d.get("initial_capital", 100000.0)),
        step=10000.0, format="%.0f")
    top_n = st.slider("每期选股数量", 10, 200,
                      int(cfg_d.get("top_n", 50)), step=5)
    use_sl = st.toggle("启用止损", value=cfg_d.get("stop_loss") is not None)
    sl = st.number_input("止损线（相对成本）", min_value=-0.50, max_value=0.0,
                         value=float(cfg_d.get("stop_loss", -0.15)),
                         step=0.01, format="%.2f", disabled=not use_sl)
    use_tp = st.toggle("启用止盈", value=cfg_d.get("take_profit") is not None)
    tp = st.number_input("止盈线（相对成本）", min_value=0.0, max_value=1.0,
                         value=float(cfg_d.get("take_profit", 0.30)),
                         step=0.01, format="%.2f", disabled=not use_tp)
    src_names = ["tencent", "akshare", "mootdx", "baostock"]
    try:
        src_idx = src_names.index(str(cfg_d.get("data_source", "tencent")))
    except ValueError:
        src_idx = 0
    data_source = st.selectbox("数据源", src_names, index=src_idx)
    if st.button("保存参数到配置", width="stretch"):
        update_config_yaml(
            PROJECT / "config.yaml",
            initial_capital=capital, top_n=top_n,
            stop_loss=sl if use_sl else None,
            take_profit=tp if use_tp else None,
            data_source=data_source)
        st.session_state["cfg_saved"] = True
        st.rerun()
    if st.session_state.pop("cfg_saved", False):
        st.success("已保存到 config.yaml，下次「每日更新」生效（账户页已按新资金预览）。")
    st.caption("改资金只改变账户口径，不改变选股决策。")
    st.divider()
    html_block(
        '<div class="side-sec">使用方式</div>'
        '<div class="side-card">在「总览」页点按钮即可下载/更新数据；'
        '任务在后台运行、输出实时显示，刷新页面不中断。<br>'
        '盘中实时估值默认关闭，开启后按快照价估值。</div>')
    st.divider()
    st.caption(DISCLAIMER)

# ---------- 共享数据上下文（所有标签页共用一次读取） ----------
manifest = load_json(data_dir / "manifest.json") if data_dir.exists() else None
model_meta = load_json(model_dir / "meta.json")
decision = load_decision(sim_dir)
stocks = {k: v for k, v in (manifest or {}).items() if k != "sh000300"}
idx_end = (manifest or {}).get("sh000300", {}).get("end")
stale = [k for k, v in stocks.items()
         if v.get("end") and idx_end and v["end"] < idx_end]
close_panel = load_panel_close(data_dir)
history_path = data_dir / "portfolio" / "account_history.jsonl"
hist_all = load_history_cached(history_path) if history_path.exists() else []
hist_live = [e for e in hist_all if e.get("mode") == "live"] or hist_all
# 持仓明细/本期口径同样以决策日累计净资产为基准（避免重置回 10 万）
basis = account_basis(hist_live, close_panel, decision["date"], float(capital),
                      *account_params()) \
    if decision is not None and close_panel is not None else float(capital)
dec_view = {**decision, "initial_capital": basis} if decision is not None else None
account = account_snapshot(dec_view, close_panel) \
    if dec_view and close_panel is not None else None
equity_cum, _ = recompute_account(hist_live, close_panel, float(capital),
                                  *account_params()) \
    if hist_live and close_panel is not None else (pd.Series(dtype=float), {})
acc_summary = load_json(data_dir / "portfolio" / "account_summary.json")
sched_state = auto_update_status()

# ---------- 顶部标题 + 全局状态 ----------
hero("A股量化研究 · 模拟分析控制台",
     "全市场量化模拟研究 · 数据每日收盘后更新",
     pills=[
         ("数据截止", idx_end or "无数据", "accent" if idx_end else "bad"),
         ("覆盖股票", f"{len(stocks):,} 只" if stocks else "—",
          "info" if stocks else "bad"),
         ("落后股票", f"{len(stale):,} 只", "ok" if not stale else "warn"),
         ("决策日期", str(decision["date"]) if decision else "—",
          "info" if decision else "warn"),
         ("每日自动更新", sched_state, "ok" if sched_state == "开启" else "warn"),
         (f"⚠️ {DISCLAIMER}", "", "warn"),
     ])
if stale:
    st.warning(f"有 {len(stale)} 只股票数据落后于指数截止日（可能上次更新中断或停牌），"
               "点击「总览」页的「每日更新」自动补齐。")

tab_overview, tab_sim, tab_decision, tab_account, tab_realtime, tab_algo, tab_log, tab_data = st.tabs(
    ["总览", "模拟盘", "今日决策", "账户", "实时行情", "算法对比", "调整日志", "数据状态"])

# ================================================================ 总览
with tab_overview:
    page_header("系统状态与快速操作",
                "数据、模型、账户与今日持仓的一页式总览；数据更新也在这里启动。")

    kpi_row([
        dict(label="覆盖股票", value=f"{len(stocks):,}", unit="只",
             delta=f"清单条目 {len(manifest or {}):,} 条"),
        dict(label="数据截止", value=str(idx_end or "无数据"),
             delta=("沪深300 指数最新交易日" if idx_end else "尚未下载数据")),
        dict(label="落后股票", value=f"{len(stale):,}", unit="只",
             tone=("down" if not stale else "up"),
             delta=("全部与指数同步" if not stale else "落后于指数截止日，需补齐"),
             help_text="本地日线截止日 < 指数截止日的股票数；用于发现更新中断。"),
        dict(label="今日持仓", value=(f"{len(decision['picks'])}" if decision else "—"),
             unit=("只" if decision else ""),
             delta=(f"决策日 {decision['date']}" if decision else "暂无决策")),
    ])

    if not manifest:
        st.warning("尚未下载数据。首次使用请点击下方「下载/更新全市场数据」——"
                   "约 20-35 分钟，可断点续传（中断后重跑自动续传）。")
    elif not model_meta:
        st.info("模型尚未训练，点击下方「每日更新」会自动训练（约 1 分钟）。")
    else:
        st.success("数据与模型就绪。每日收盘后点击「每日更新」："
                   "增量拉数据 → 生成报告 → 输出今日模拟持仓。"
                   + ("（有股票待补齐，见上方提示）" if stale else ""))

    st.subheader("快速操作")
    cmd_base = ["python", "-X", "utf8", "-u", "-m", "ashare_quant.cli"]
    b1, b2, b3 = st.columns(3)
    if b1.button("每日更新（推荐）", type="primary", width="stretch"):
        launch_task("daily", cmd_base + ["daily", "--config", str(PROJECT / "config.yaml"),
                                         "--data-root", str(data_dir), "--out-dir", str(sim_dir),
                                         "--model-dir", str(model_dir)], PROJECT)
    if b2.button("强制重算报告+决策", width="stretch"):
        launch_task("force", cmd_base + ["daily", "--config", str(PROJECT / "config.yaml"),
                                         "--data-root", str(data_dir), "--out-dir", str(sim_dir),
                                         "--model-dir", str(model_dir), "--force"], PROJECT)
    if b3.button("下载/更新全市场数据（首次 20-35 分钟）", width="stretch"):
        launch_task("fetch", cmd_base + ["fetch", "--config", str(PROJECT / "config.yaml"),
                                         "--universe", "all", "--data-root", str(data_dir),
                                         "--years", "3"], PROJECT)
    render_task("daily", "每日更新")
    render_task("force", "强制重算")
    render_task("fetch", "全市场数据下载")
    st.caption(f"每日自动更新：{sched_state}（可在 start.bat 菜单 8 切换，默认开启）。"
               "盘中点「每日更新」不会推进决策日（未收盘 bar 会被丢弃），收盘后 16:05 正式更新。")

    st.divider()
    st.subheader("模拟账户")
    if account or not equity_cum.empty:
        if not equity_cum.empty:
            cum_asset = float(equity_cum.iloc[-1])
            cum_ret = cum_asset / float(capital) - 1
            cum_pnl = cum_asset - float(capital)
            cum_start = str(equity_cum.index[0].date())
            kpi_row([
                dict(label="初始资金", value=money(capital), unit="元",
                     delta="侧边栏可调，仅影响账户口径"),
                dict(label="累计总资产", value=money(cum_asset), unit="元",
                     delta=f"截至 {equity_cum.index[-1].date()}"),
                dict(label="累计收益率", value=f"{cum_ret:+.2%}", tone=tone_of(cum_ret),
                     delta=f"自 {cum_start} 首个决策日跟踪"),
                dict(label="累计盈亏", value=signed_money(cum_pnl), unit="元",
                     tone=tone_of(cum_pnl), delta="累计总资产 − 初始资金"),
            ])
            st.caption(f"累计口径：自 {cum_start} 首个决策日起按收盘价逐日盯市值，"
                       f"与「账户」页一致；截至 {equity_cum.index[-1].date()}。")
        else:
            kpi_row([
                dict(label="初始资金", value=money(account["initial"]), unit="元"),
                dict(label="本期总资产", value=money(account["total_asset"]), unit="元"),
                dict(label="本期收益率", value=f"{account['total_return']:+.2%}",
                     tone=tone_of(account["total_return"])),
                dict(label="本期浮动盈亏", value=signed_money(account["total_pnl"]), unit="元",
                     tone=tone_of(account["total_pnl"])),
            ])
        if account:
            st.caption(f"本期口径：决策日 {decision['date']} 收盘建仓"
                       f"（基准 {basis:,.0f} 元 = 当日累计净资产），自决策日收益 "
                       f"{account['total_return']:+.2%}（刚决策当日为 0，次日开始体现）。")
        render_realtime_valuation(decision, data_dir, key="overview_rt",
                                  capital=float(capital))
    else:
        empty_state("暂无账户数据",
                    "运行「每日更新」生成首个正式决策后，账户净值开始逐日记录。")

    st.divider()
    st.subheader("今日模拟持仓（前 10）")
    if decision:
        picks_table(decision, head=10)
        st.caption(f"决策日期 {decision['date']}，模型："
                   f"{'、'.join(display_name(m) for m in decision['models'])}。"
                   "选股按各模型预测的当日截面排名均值排序（rank 融合），"
                   "「预期收益」为各模型预测的中位数（展示用）；"
                   "每只股票的模型明细见「今日决策」页。")
    else:
        empty_state("暂无决策结果",
                    "在「每日更新」运行完成后生成今日模拟持仓；"
                    "首次运行会自动训练模型（约 1 分钟）。")

# ================================================================ 模拟盘
with tab_sim:
    page_header("模拟盘对比（月度调仓 Top-50，含交易成本）",
                "早期 5 个基础策略的月度调仓回测与真实市场基准对比。")
    risk_cfg = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8")) \
        if (PROJECT / "config.yaml").exists() else None
    returns = load_csv(sim_dir / "model_returns.csv")
    if returns is None:
        empty_state("未找到模拟盘结果",
                    "在「总览」运行「每日更新」，或先跑一次模拟盘命令后生成。")
    else:
        cum = (1 + returns.fillna(0)).prod() - 1
        # 只显示策略（基准线已在净值曲线中，避免把大盘涨幅当策略收益）
        strategy_cum = cum[[c for c in cum.index if not str(c).startswith("基准")]]
        kpi_row([dict(label=display_name(name), value=f"{v:+.2%}", tone=tone_of(v),
                      delta="月度调仓累计收益（含成本）")
                 for name, v in strategy_cum.items()])
        st.caption("虚线为真实市场基准：等权全市场与沪深300 指数买入持有。")
        if risk_cfg and (risk_cfg.get("stop_loss") is not None
                         or risk_cfg.get("take_profit") is not None):
            st.caption(f"仓位风控已启用：止损 {risk_cfg.get('stop_loss'):+.0%}、"
                       f"止盈 {risk_cfg.get('take_profit'):+.0%}（config.yaml 可调，None 关闭）。")
        chart(equity_figure(returns), key="sim_equity")
        with st.expander("口径说明（与「算法对比」页的区别）"):
            st.caption("此处为早期 5 个基础策略（动量/反转/低波/多因子/轮动）的月度调仓回测，"
                       "收益按调仓月标记（如 07-01 段已含最新交易日数据）；"
                       "当前每日决策使用的是 6 个 ML 模型，其样本外表现请见「算法对比」页。")
        sim_json = load_json(sim_dir / "simulation.json")
        if sim_json and "summary" in sim_json:
            st.subheader("累计总收益（近三年模拟）")
            st.dataframe(format_metric(pd.DataFrame(sim_json["summary"])),
                         width="stretch", hide_index=True)

# ================================================================ 今日决策
with tab_decision:
    page_header("今日模拟投资决策", "本期持仓、相对上一决策日的变化、个股K线。")
    if decision is None:
        empty_state("未找到决策结果",
                    "在「总览」运行「每日更新」后生成；首次运行会自动训练模型。")
    else:
        model_names = "、".join(display_name(m) for m in decision["models"])
        pill_row([("决策日期", str(decision["date"]), "accent"),
                  ("持仓", f"{len(decision['picks'])} 只", "info"),
                  ("模型", model_names, "")])
        picks_table(decision)
        st.caption("选股按各模型预测的当日截面排名均值排序（rank 融合）；"
                   "「预期收益」为各模型预测的中位数（展示用），模拟研究仅供学习。")
        if account:
            st.subheader("账户持仓明细")
            pos = account["rows"].copy()
            for col in ("投入金额", "股数", "市值"):
                if col in pos.columns:
                    pos[col] = pos[col].map(lambda v: f"{v:,.0f}")
            for col in ("成本价", "现价"):
                if col in pos.columns:
                    pos[col] = pos[col].map(lambda v: f"{v:.2f}")
            for col in ("权重", "盈亏率"):
                if col in pos.columns:
                    pos[col] = pos[col].map(lambda v: f"{v:.2%}")
            if "浮动盈亏" in pos.columns:
                pos["浮动盈亏"] = pos["浮动盈亏"].map(signed_money)
            st.dataframe(pos, width="stretch", hide_index=True)
        with st.expander("模型预测明细（为什么选这些股票）"):
            model_names_txt = "、".join(
                display_name(m) for m in decision.get("models", []))
            st.caption(f"每只股票在 {model_names_txt} 下的未来 20 日预测；"
                       "排序用各模型预测的当日截面排名均值（rank 融合），"
                       "「加权得分」为预测中位数（展示用）。")
            detail = pd.DataFrame(decision["picks"]).copy()
            if "model_scores" in detail.columns and detail["model_scores"].notna().any():
                scores = pd.json_normalize(detail["model_scores"].dropna().tolist())
                detail = pd.concat([detail[["symbol", "score"]], scores], axis=1)
                detail = detail.rename(columns={"symbol": "代码", "score": "加权得分"})
                detail = detail.rename(columns={c: display_name(c) for c in scores.columns})
                st.dataframe(detail, width="stretch", hide_index=True)
            else:
                st.info("当前决策文件缺少模型明细，重新运行 daily 后自动生成。")

        st.subheader("相对上一决策日变化")
        if len(hist_live) >= 2:
            cur_dec, prev_dec = hist_live[-1], hist_live[-2]
            cur_date, prev_date = cur_dec.get("date", "?"), prev_dec.get("date", "?")
            cur_map = {p["symbol"]: p for p in cur_dec.get("picks", [])}
            prev_map = {p["symbol"]: p for p in prev_dec.get("picks", [])}
            cur_syms, prev_syms = set(cur_map), set(prev_map)
            diff_rows = []
            for s in sorted(cur_syms - prev_syms):
                p = cur_map[s]
                diff_rows.append({"代码": s, "变化": "🆕 新增", "当前权重": p.get("weight"),
                                  "预期收益(20日)": p.get("score"), "_score": p.get("score")})
            for s in sorted(prev_syms - cur_syms):
                p = prev_map[s]
                diff_rows.append({"代码": s, "变化": "➖ 卖出", "当前权重": None,
                                  "预期收益(20日)": p.get("score"), "_score": p.get("score")})
            for s in sorted(cur_syms & prev_syms):
                p = cur_map[s]
                diff_rows.append({"代码": s, "变化": "✅ 持有", "当前权重": p.get("weight"),
                                  "预期收益(20日)": p.get("score"), "_score": p.get("score")})
            order = {"🆕 新增": 0, "➖ 卖出": 1, "✅ 持有": 2}
            diff_rows.sort(key=lambda r: (order[r["变化"]], -((r["_score"] or 0))))
            for row in diff_rows:
                row.pop("_score", None)
            kpi_row([
                dict(label="🆕 新增", value=f"{len(cur_syms - prev_syms)}", unit="只",
                     tone="up", delta="本期新买入（按预期收益降序）"),
                dict(label="➖ 卖出", value=f"{len(prev_syms - cur_syms)}", unit="只",
                     tone="down", delta="上期持有、本期剔除"),
                dict(label="✅ 持有", value=f"{len(cur_syms & prev_syms)}", unit="只",
                     delta="两期都在"),
            ])
            df_diff = pd.DataFrame(diff_rows, columns=["代码", "变化", "当前权重", "预期收益(20日)"])
            if "当前权重" in df_diff.columns:
                df_diff["当前权重"] = df_diff["当前权重"].map(
                    lambda v: f"{v:.1%}" if pd.notna(v) else "-")
            if "预期收益(20日)" in df_diff.columns:
                df_diff["预期收益(20日)"] = df_diff["预期收益(20日)"].map(
                    lambda v: f"{v:.2%}" if pd.notna(v) else "-")
            st.caption(f"对比 {prev_date} → {cur_date}：新增=本期新买入（按预期收益降序），"
                       "卖出=上期持有本期剔除，持有=两期都在。")
            st.dataframe(df_diff, width="stretch", hide_index=True)
        else:
            st.caption("暂无上一决策日对比（账户刚开始记录）。")

        st.subheader("个股K线（本地数据，最近 120 个交易日）")
        sel = st.selectbox("选择个股", [p["symbol"] for p in decision["picks"]])
        sym_df = load_symbol(data_dir, sel)
        if sym_df is None or sym_df.empty:
            st.info("本地无该股K线数据（可能为新上市或数据缺失）。")
        else:
            tail = sym_df.tail(120)
            chart(kline_figure(tail, sel, f"{sel} K线（最近 {len(tail)} 个交易日）", 560),
                  key="decision_kline")

# ================================================================ 账户
with tab_account:
    page_header("模拟账户净值（自首个正式决策日跟踪）",
                "收盘价口径的逐日净值、绩效指标、同期基准对比与交易台账。")
    st.caption("账户自首个正式决策日（样本外）开始逐日盯市值；"
               "决策每日收盘后生成，收益随每日更新持续累积。"
               "曲线为收盘价口径（每日更新后刷新）；盘中实时估值见下方开关。"
               "历史策略表现请参考「模拟盘」页。")
    render_realtime_valuation(decision, data_dir, key="account_rt",
                              capital=float(capital))
    equity_csv = load_csv(data_dir / "portfolio" / "account_equity.csv")
    if equity_csv is None or equity_csv.empty:
        if acc_summary:
            st.info("首个正式决策已生成，账户净值曲线将于下一交易日（每日更新后）开始记录。")
            kpi_row([
                dict(label="总资产", value=money(acc_summary.get("total_asset", 0.0)), unit="元"),
                dict(label="总收益率", value=f"{acc_summary.get('total_return', 0.0):+.2%}",
                     tone=tone_of(acc_summary.get("total_return", 0.0))),
            ])
            st.info(f"账户运行不足 {MIN_METRIC_DAYS} 个交易日，"
                    "年化/夏普/回撤等绩效指标暂无统计意义，曲线成形后自动展示。")
            st.caption(f"已记录 {acc_summary.get('decisions', 0)} 次决策，"
                       f"数据截至 {acc_summary.get('as_of', '-')}。")
        else:
            empty_state("暂无账户曲线",
                        "运行「每日更新」生成首个正式决策后开始记录。")
    else:
        view_capital = float(capital)
        hist_view = hist_live
        equity, metrics = recompute_account(hist_view, close_panel, view_capital,
                                            *account_params()) \
            if close_panel is not None and hist_view else (pd.Series(dtype=float), {})
        if equity.empty:
            # 回退到已保存的 CSV 与 summary（如缺少面板/历史）
            equity = equity_csv[equity_csv.columns[0]].astype(float)
            metrics = acc_summary or {}
            view_capital = float(acc_summary.get("initial_capital", view_capital)) \
                if acc_summary else view_capital
        chart(account_figure(equity, data_dir, close_panel, view_capital), key="acct_equity")
        if len(equity) < 5:
            st.info(f"账户刚开始记录（当前仅 {max(1, len(equity) - 1)} 个交易日收益），"
                    "曲线会随每日更新逐步成形；"
                    "想看完整历史策略表现，请切换到「模拟盘」页。")
        total_asset = float(equity.iloc[-1])
        total_return = total_asset / view_capital - 1 if view_capital > 0 else 0.0
        enough = len(equity) >= MIN_METRIC_DAYS
        plr = metrics.get("profit_loss_ratio", float("nan"))
        calmar = metrics.get("calmar", float("nan"))
        kpi_row([
            dict(label="总资产", value=money(total_asset), unit="元",
                 delta=f"截至 {equity.index[-1].date()}"),
            dict(label="总收益率", value=f"{total_return:+.2%}", tone=tone_of(total_return),
                 delta=f"相对 {money(view_capital)} 元本金"),
            dict(label="年化收益", value=format_pct_nan(metrics.get("annual_return")) if enough else "—",
                 tone=(tone_of(metrics.get("annual_return")) if enough else ""),
                 help_text="按日收益年化；不足 20 个交易日不展示"),
            dict(label="夏普", value=f"{metrics.get('sharpe', 0.0):.2f}" if enough else "—",
                 help_text="年化收益 / 年化波动；不足 20 个交易日不展示"),
            dict(label="最大回撤", value=format_pct_nan(metrics.get("max_drawdown")) if enough else "—",
                 tone=("down" if enough and pd.notna(metrics.get("max_drawdown")) else ""),
                 help_text="净值历史峰值到最低点的最大跌幅"),
        ])
        kpi_row([
            dict(label="胜率", value=format_pct(metrics.get("win_rate")) if enough else "—",
                 help_text="上涨交易日占比；不足 20 个交易日不展示"),
            dict(label="盈亏比", value=f"{plr:.2f}" if enough and not pd.isna(plr) else "—",
                 help_text="平均盈利日 / 平均亏损日"),
            dict(label="年化波动", value=format_pct(metrics.get("annual_vol")) if enough else "—",
                 help_text="日收益标准差年化"),
            dict(label="Calmar", value=f"{calmar:.2f}" if enough and not pd.isna(calmar) else "—",
                 help_text="年化收益 / 最大回撤"),
        ])
        if not enough:
            st.info(f"账户运行仅 {len(equity)} 个净值点（不足 {MIN_METRIC_DAYS} 天），"
                    "年化/夏普/回撤/胜率等指标暂无统计意义，曲线成形后自动展示。")
        decisions_n = (acc_summary or {}).get("decisions", len(hist_view))
        as_of = (acc_summary or {}).get("as_of", str(equity.index[-1].date()))
        st.caption(f"已记录 {decisions_n} 次决策，数据截至 {as_of}；"
                   f"账户按侧边栏资金 {view_capital:,.0f} 元预览，"
                   "保存参数后每日更新沿用新口径。")

        # 账户 vs 基准同期收益对比
        if len(equity) >= 2 and close_panel is not None:
            start, end = equity.index[0], equity.index[-1]
            bench_ret = {}
            bench_end = None
            idx_path = data_dir / "sh000300.parquet"
            if idx_path.exists():
                idx_close = pd.read_parquet(idx_path)["close"]
                idx_sel = idx_close.loc[start:end]
                if len(idx_sel) >= 2:
                    bench_ret["基准·沪深300"] = idx_sel.iloc[-1] / idx_sel.iloc[0] - 1
            bench_eq, dropped = equal_weight_bench(close_panel, start, end)
            if len(bench_eq) >= 2:
                bench_ret["基准·等权全市场"] = bench_eq.iloc[-1] / bench_eq.iloc[0] - 1
                bench_end = bench_eq.index[-1]
            st.subheader("同期收益对比（账户 vs 真实市场）")
            kpi_row([dict(label="模拟账户", value=f"{total_return:+.2%}",
                          tone=tone_of(total_return), delta=f"{start.date()} → {end.date()}")]
                    + [dict(label=name, value=f"{v:+.2%}", tone=tone_of(v),
                            delta=f"{start.date()} → {(bench_end or end).date()}")
                       for name, v in bench_ret.items()])
            if dropped:
                st.caption(f"注：等权全市场基准已剔除 {dropped} 个成分股覆盖不足的交易日"
                           f"（当日有效股票数少于全市场 {BENCH_MIN_COVERAGE:.0%}），"
                           "避免少数组件把基准拉偏。")

        # 月度收益热力图
        returns_v = equity.pct_change(fill_method=None).dropna()
        monthly = monthly_returns_table(returns_v)
        if not monthly.empty:
            st.subheader("月度收益热力图")
            mvals = monthly.values
            text = np.vectorize(lambda v: f"{v:.1%}" if pd.notna(v) else "")(mvals)
            hfig = go.Figure(go.Heatmap(
                z=mvals * 100, x=[f"{m}月" for m in monthly.columns],
                y=[str(y) for y in monthly.index], colorscale=HEAT_SCALE, zmid=0,
                text=text, texttemplate="%{text}",
                hovertemplate="%{y}年 %{x}: %{z:.2f}%<extra></extra>"))
            hfig.update_layout(title="月度收益（%，红涨绿跌）",
                               height=max(220, 45 * len(monthly.index)))
            chart(style_fig(hfig, hovermode="closest"), key="acct_heat")

        # 交易台账
        if close_panel is not None and hist_view:
            ledger = build_trade_ledger(hist_view, close_panel, view_capital)
            if not ledger.empty:
                st.subheader("交易台账（每次决策视为等权全换仓）")
                shown = ledger.copy()
                for coln in ("数量", "价格", "金额", "实现盈亏"):
                    if coln in shown.columns:
                        shown[coln] = shown[coln].map(lambda v: f"{v:,.2f}")
                st.dataframe(shown, width="stretch", hide_index=True)
                st.download_button("下载交易台账 CSV", ledger.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="trade_ledger.csv", mime="text/csv")

        csv_data = equity.to_csv().encode("utf-8-sig")
        st.download_button("下载账户净值 CSV", data=csv_data,
                           file_name="account_equity.csv", mime="text/csv")

# ================================================================ 实时行情
with tab_realtime:
    page_header("实时行情", "大盘速览、持仓快照与动态K线；免费源为秒级快照，非交易所 tick。")
    st.subheader("大盘速览")
    try:
        idx_df = index_snapshot()
        if idx_df.empty:
            st.warning("未获取到指数行情（可能非交易时段或接口限流）。")
        else:
            kpi_row([dict(label=str(row["名称"]), value=f"{row['现价']:,.2f}",
                          tone=tone_of(row["涨跌幅"]),
                          delta=f"{row['涨跌幅']:+.2%}")
                     for _, row in idx_df.iterrows()])
    except Exception as e:  # noqa: BLE001
        st.warning(f"指数行情获取失败：{e}")

    st.divider()
    st.subheader("实时行情（准实时快照，秒级延迟）")
    st.caption("免费行情源为快照级（延迟数秒），非交易所级 tick 数据；"
               "仅供盘中观察与持仓跟踪，不改变月度调仓决策逻辑。")
    auto = st.toggle("自动刷新（每 10 秒）", value=False)
    if auto:
        st_autorefresh(interval=10_000, key="realtime_refresh")
    if decision is None or not decision.get("picks"):
        empty_state("暂无持仓",
                    "先在「总览」运行「每日更新」生成决策后，这里会显示持仓的实时快照。")
    else:
        symbols = [p["symbol"] for p in decision["picks"]]
        snap = pd.DataFrame()
        snap_err: Exception | None = None
        try:
            snap = _cached_snapshot(tuple(symbols))
        except Exception as e:  # noqa: BLE001 - 行情源不可用 → 降级本地收盘兜底
            snap_err = e
        d0 = pd.Timestamp(decision["date"])
        if snap.empty:
            if snap_err is not None:
                # 显式给出失败原因：源已失效与"休市无行情"是两回事，
                # 旧实现把两者都显示成"可能非交易时段或接口限流"。
                st.error(f"实时行情获取失败：{snap_err}")
            st.warning("未获取到实时行情，可切换数据源重试。以下为本地最新收盘数据。")
            local = local_close_table(data_dir, close_panel, symbols, decision["date"])
            if local is not None:
                st.dataframe(local, width="stretch", hide_index=True)
        else:
            try:
                kline_lookup = snap.set_index("代码")[
                    ["现价", "今开", "最高", "最低", "成交量(手)"]].to_dict("index") \
                    if {"现价", "今开", "最高", "最低", "成交量(手)"} <= set(snap.columns) else {}
                acc_realtime = None
                if close_panel is not None and d0 in close_panel.index:
                    # 实时估值：快照现价优先，缺失（停牌等）依次用本地最新收盘价、
                    # 决策日收盘价兜底（保持估值连续）
                    prices = snap.set_index("代码")["现价"].astype(float)
                    pick_syms = [p["symbol"] for p in decision["picks"]]
                    fallback = close_panel.iloc[-1].reindex(pick_syms)
                    fallback = fallback.fillna(close_panel.loc[d0, pick_syms])
                    prices = prices.reindex(fallback.index).fillna(fallback)
                    b = account_basis(hist_live, close_panel, decision["date"], float(capital),
                                      *account_params())
                    try:
                        acc_realtime = account_snapshot(
                            {**decision, "initial_capital": b}, close_panel, prices=prices)
                    except Exception:  # noqa: BLE001
                        acc_realtime = None
                    if acc_realtime is not None:
                        st.markdown("**实时账户估值（按快照现价）**")
                        rt_ret = acc_realtime["total_asset"] / float(capital) - 1
                        kpi_row([
                            dict(label="实时总资产", value=money(acc_realtime["total_asset"]),
                                 unit="元"),
                            dict(label="实时总收益（累计）", value=f"{rt_ret:+.2%}",
                                 tone=tone_of(rt_ret)),
                            dict(label="本期浮动盈亏",
                                 value=signed_money(acc_realtime["total_pnl"]), unit="元",
                                 tone=tone_of(acc_realtime["total_pnl"])),
                            dict(label="现金余额", value=money(acc_realtime["cash"]), unit="元"),
                        ])
                        st.caption(f"基准：决策日 {decision['date']} 累计净资产 "
                                   f"{b:,.0f} 元（自首个决策日跟踪）× 快照现价；"
                                   "实时总收益=总资产/初始资金-1（累计口径，与账户页一致）。"
                                   "「自决策日涨跌」为现价相对决策日收盘的变化，"
                                   "「本期浮动盈亏」为现价相对决策日成本。")
                        # 每只股票实时收益列（数值版，先于下方格式化）
                        rows_r = acc_realtime["rows"].rename(columns={
                            "现价": "实时价", "市值": "持仓市值",
                            "浮动盈亏": "实时盈亏(元)", "盈亏率": "实时盈亏率",
                        })
                        snap = snap.merge(
                            rows_r[["代码", "成本价", "实时价", "持仓市值",
                                    "实时盈亏(元)", "实时盈亏率"]],
                            on="代码", how="left")
                if close_panel is not None and d0 in close_panel.index:
                    base = close_panel.loc[pd.Timestamp(decision["date"]), snap["代码"]]
                    snap["决策日收盘"] = base.to_numpy()
                    snap["自决策日涨跌"] = snap["现价"] / snap["决策日收盘"] - 1
                    snap["自决策日涨跌"] = snap["自决策日涨跌"].map(
                        lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
                    snap["决策日收盘"] = snap["决策日收盘"].map(
                        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
                if acc_realtime is not None:
                    for col in ("成本价", "实时价"):
                        if col in snap.columns:
                            snap[col] = snap[col].map(
                                lambda v: f"{v:.2f}" if pd.notna(v) else "-")
                    if "持仓市值" in snap.columns:
                        snap["持仓市值"] = snap["持仓市值"].map(
                            lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
                    if "实时盈亏(元)" in snap.columns:
                        snap["实时盈亏(元)"] = snap["实时盈亏(元)"].map(
                            lambda v: signed_money(v) if pd.notna(v) else "-")
                    if "实时盈亏率" in snap.columns:
                        snap["实时盈亏率"] = snap["实时盈亏率"].map(
                            lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
                if "涨跌幅" in snap.columns:
                    snap["涨跌幅"] = snap["涨跌幅"].map(
                        lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
                for col in ("现价", "今开", "最高", "最低", "昨收"):
                    if col in snap.columns:
                        snap[col] = snap[col].map(
                            lambda v: f"{v:.2f}" if pd.notna(v) else "-")
                st.dataframe(snap, width="stretch", hide_index=True)
                st.caption(f"决策日期 {decision['date']}，共 {len(snap)} 只；"
                           "「自决策日涨跌」为现价相对决策日收盘的变化，"
                           "「实时盈亏率」为现价相对持仓成本。")
                # 动态K线：本地日线 + 实时快照拼接当日 bar，随 10s 自动刷新动态更新
                st.subheader("动态K线（自动刷新时实时更新）")
                ksym = st.selectbox("选择个股",
                                    [p["symbol"] for p in decision["picks"]],
                                    key="realtime_kline_symbol")
                kdf = load_symbol(data_dir, ksym)
                if kdf is None or kdf.empty:
                    st.info("本地无该股K线数据。")
                else:
                    tail = kdf.tail(60).copy()
                    q = kline_lookup.get(ksym)
                    today = pd.Timestamp.today().normalize()
                    if q is not None and tail.index.max().date() < today.date():
                        now = q.get("现价")
                        if now not in (None, "-") and float(now) > 0:
                            now = float(now)
                            o = float(q["今开"]) if q.get("今开") not in (None, "-") else now
                            hi = float(q["最高"]) if q.get("最高") not in (None, "-") else now
                            lo = float(q["最低"]) if q.get("最低") not in (None, "-") else now
                            vol = float(q.get("成交量(手)") or 0) * 100
                            tail.loc[today] = {
                                "open": o, "high": max(hi, now), "low": min(lo, now),
                                "close": now, "volume": vol,
                            }
                    tail = tail.sort_index()
                    chart(kline_figure(tail, ksym,
                                       f"{ksym} 动态K线（最近 {len(tail)} 个交易日）", 520),
                          key="realtime_kline")
                    st.caption("最后一根为实时快照拼接的当日 bar（现价/高低随刷新更新）；"
                               "开启上方「自动刷新」后每 10 秒动态变化。")
            except Exception as e:  # noqa: BLE001
                st.error(f"实时行情获取失败：{e}")

# ================================================================ 算法对比
with tab_algo:
    page_header("算法表现对比（样本外夏普）",
                "全市场 walk-forward 的样本外指标与净值曲线。")
    bench_stem = "algorithm-benchmark-all" if mode == "all" else "algorithm-benchmark"
    bench = load_json(PROJECT / "docs/research" / f"{bench_stem}.json")
    if bench is None:
        empty_state("未找到算法对比结果",
                    "运行 benchmark 命令后生成（全市场约 6 分钟）。")
    else:
        df = pd.DataFrame(bench).sort_values("sharpe", ascending=False)
        fig = go.Figure(go.Bar(
            x=[display_name(m) for m in df["model"]], y=df["sharpe"],
            marker_color=[UP if v >= 2 else ACCENT for v in df["sharpe"]],
            hovertemplate="%{x}<br>夏普 %{y:.3f}<extra></extra>"))
        fig.update_layout(title="各算法样本外夏普（红 = 夏普 ≥ 2）", xaxis_tickangle=-30,
                          showlegend=False)
        chart(style_fig(fig, height=380, hovermode="closest"), key="algo_sharpe")
        st.dataframe(format_metric(df), width="stretch", hide_index=True)
        returns_a = load_csv(PROJECT / "docs/research" / f"{bench_stem}.returns.csv")
        if returns_a is not None:
            st.subheader("各算法样本外净值曲线（含真实基准）")
            st.caption("虚线为真实市场基准；曲线为月度调仓 Top-50 等权的样本外净值。")
            # 全部算法（20+ 条）画在一张图里会糊成一团，默认只画期末净值前 8 + 全部基准，
            # 其余可按需勾选（数据一条都没少，只是默认不堆在一张图上）
            bench_cols = [c for c in returns_a.columns
                          if str(c).startswith(("基准", "benchmark_"))]
            finals = (1 + returns_a.fillna(0)).cumprod().iloc[-1]
            top_cols = [c for c in finals.sort_values(ascending=False).index
                        if c not in bench_cols][:8]
            default_cols = [c for c in top_cols + bench_cols if c in returns_a.columns]
            picked = st.multiselect(
                "显示的策略（默认＝期末净值前 8 + 全部基准）",
                options=list(returns_a.columns), default=default_cols,
                format_func=display_name)
            if picked:
                chart(equity_figure(returns_a[picked]), key="algo_equity")
            else:
                st.info("至少勾选一个策略。")
        else:
            st.info("暂无收益曲线数据，重跑 benchmark 命令后自动生成。")

# ================================================================ 调整日志
with tab_log:
    page_header("反馈调整日志", "模型权重轮动的历史调整记录（宽表：调整前 → 调整后）。")
    log_path = sim_dir / "adjustments.jsonl"
    if not log_path.exists():
        empty_state("暂无调整日志",
                    "运行模拟盘/每日更新产生权重轮动后，这里会记录每次调整。")
    else:
        entries = [json.loads(line) for line in
                   log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not entries:
            empty_state("调整日志为空", "文件存在但没有任何记录。")
        else:
            triggers = sorted({str(e.get("trigger", "-")) for e in entries})
            kpi_row([
                dict(label="记录条数", value=f"{len(entries):,}", unit="条",
                     delta=f"{entries[0].get('date', '-')} → {entries[-1].get('date', '-')}"),
                dict(label="最近调整", value=str(entries[-1].get("date", "-")),
                     delta=f"触发：{entries[-1].get('trigger', '-')}"),
                dict(label="触发类型", value=f"{len(triggers)}", unit="类",
                     delta="、".join(display_name(t) for t in triggers)),
            ])
            # 宽表：把 before/after 两个 dict 展开成「调整前 → 调整后」
            strategies = sorted({k for e in entries
                                 for k in list((e.get("before") or {}).keys())
                                 + list((e.get("after") or {}).keys())})
            wide = pd.DataFrame([{
                "日期": e.get("date", "-"),
                "触发": display_name(e.get("trigger", "-")),
                "动作": e.get("action", "-"),
                **{display_name(s): _weight_transition(e, s) for s in strategies},
                "效果": e.get("effect", "-"),
            } for e in entries])
            st.caption(f"共 {len(wide):,} 条；权重列显示「调整前 → 调整后」，"
                       "空值表示该次调整未涉及该策略。")
            st.dataframe(wide.tail(200), width="stretch", hide_index=True)
            with st.expander(f"查看全部 {len(wide):,} 条 / 原始 JSON / 导出 CSV"):
                st.dataframe(wide, width="stretch", hide_index=True)
                st.download_button("下载调整日志 CSV",
                                   wide.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="adjustments_wide.csv", mime="text/csv")
                st.dataframe(pd.DataFrame(entries), width="stretch", hide_index=True)

# ================================================================ 数据状态
with tab_data:
    page_header("数据状态", "最近一次每日更新的结果、数据完整率与缓存规模。")
    if data_dir.exists():
        stats_path = data_dir / "update_stats.json"
        if stats_path.exists():
            stats = load_json(stats_path)
            st.subheader("最近一次每日更新")
            behind = stats.get("days_behind")
            kpi_row([
                dict(label="运行时间", value=str(stats.get("last_run") or "-")),
                dict(label="数据源", value=str(stats.get("source") or "-"),
                     delta="、".join(stats.get("index_sources") or []) or "—"),
                dict(label="更新股票", value=f"{stats.get('updated', 0):,}", unit="只"),
                dict(label="总耗时", value=str(stats.get("total_sec") or "-"), unit="秒",
                     # 用 `or "-"` 而不是 .get(k, "-")：键存在但值为 null 时
                     # .get 会返回 None，拼出来是 "Nones"（实测踩过）
                     delta=(f"拉取 {stats.get('phase1_sec') or '-'}s / "
                            f"面板 {stats.get('phase2_sec') or '-'}s / "
                            f"报告+决策 {stats.get('phase3_sec') or '-'}s")),
                dict(label="指数截止", value=str(stats.get("index_date") or "-")),
            ])
            if stats.get("index_status") == "all_sources_empty" or stats.get("error"):
                # 行情源全挂时更新会"看起来成功"（09-10/09-11 冻结两天的根因），
                # 这里必须显式报警，不能只靠日志。
                st.error(f"最近一次每日更新没有取到行情：{stats.get('error') or '指数源全部无返回'}"
                         f"（数据源 {stats.get('source')}，指数截止 {stats.get('index_date')}）")
            elif behind:
                st.warning(f"数据落后 {behind} 个交易日"
                           f"（指数截止 {stats.get('index_date')}），"
                           "请确认 16:05 计划任务与行情源可用。")
            elif stats.get("primary_index_empty"):
                st.warning(f"主源 {stats.get('source')} 指数无返回，本次指数来自备源 "
                           f"{'、'.join(stats.get('index_sources') or [])}；"
                           "持续如此请调整 config.yaml 的 data_source。")
            if stats.get("decision_sec") is not None:
                st.caption(f"其中报告 {stats.get('report_sec')}s、决策 {stats.get('decision_sec')}s。")
            st.divider()
        try:
            from ashare_quant.fetchers import list_sources
            st.caption("可用数据源：" + "、".join(list_sources()))
        except Exception:  # noqa: BLE001
            pass
        parquet = list(data_dir.glob("*.parquet"))
        manifest2 = load_json(data_dir / "manifest.json")
        st.subheader("缓存与完整性")
        if manifest2:
            stocks2 = {k: v for k, v in manifest2.items() if k != "sh000300"}
            idx_end2 = manifest2.get("sh000300", {}).get("end")
            stale2 = [k for k, v in stocks2.items()
                      if v.get("end") and idx_end2 and v["end"] < idx_end2]
            sync_n = len(stocks2) - len(stale2)
            rate = sync_n / len(stocks2) if stocks2 else 0.0
            kpi_row([
                dict(label="缓存文件", value=f"{len(parquet):,}", unit="个"),
                dict(label="清单条目", value=f"{len(manifest2):,}", unit="条"),
                dict(label="覆盖股票", value=f"{len(stocks2):,}", unit="只"),
                dict(label="与指数同步", value=f"{sync_n:,}", unit="只",
                     tone=("down" if rate >= 0.9 else "up"),
                     delta=f"{sync_n:,}/{len(stocks2):,}"),
            ])
            st.progress(min(1.0, max(0.0, rate)),
                        text=f"数据完整率 {rate:.2%}（与指数同步 {sync_n}/{len(stocks2)}）")
            ends = {}
            for k, v in manifest2.items():
                if k == "sh000300" or v.get("end") is None:
                    continue
                ends[v["end"]] = ends.get(v["end"], 0) + 1
            if ends:
                dist = pd.Series(ends).sort_index()
                dfig = go.Figure(go.Bar(
                    x=dist.index.astype(str), y=dist.values,
                    marker_color=[ACCENT if str(d) == str(idx_end2) else T["bar_warn"]
                                  for d in dist.index],
                    text=[f"{n:,}" for n in dist.values], textposition="outside",
                    hovertemplate="%{x}<br>%{y:,} 只<extra></extra>"))
                dfig.update_layout(title="股票数据截止日分布（橙 = 与指数不同步）",
                                   showlegend=False)
                chart(style_fig(dfig, height=300, hovermode="closest"), key="data_dist")
                st.caption("股票数据截止日分布：" + "，".join(
                    f"{d}:{n}只" for d, n in sorted(ends.items())))
        index_path = data_dir / "sh000300.parquet"
        if index_path.exists():
            idx = pd.read_parquet(index_path)
            st.caption(f"沪深300 指数数据截止：{idx.index.max().date()}　行数：{len(idx)}")
        failed_path = data_dir / "update_failed.json"
        if failed_path.exists():
            failed = load_json(failed_path)
            if failed:
                st.warning(f"今日跳过（停牌/异常，次日自动重试）：{len(failed)} 只 "
                           f"— {'、'.join(list(failed)[:10])}")
    else:
        empty_state("数据目录不存在",
                    "在「总览」页点击「下载/更新全市场数据」开始首次下载。")

st.divider()
st.caption(f"⚠️ {DISCLAIMER}")
