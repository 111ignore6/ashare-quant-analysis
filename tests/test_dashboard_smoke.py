"""仪表盘冒烟测试：mock 行情后整页运行，确保无异常（回归保护接线 bug）。

覆盖点：
  1. 整页可运行、8 个标签页都渲染出内容、无异常；
  2. 连续刷新稳定（st.cache_data 嵌套曾导致 KeyError）；
  3. 行情源返回空（非交易时段/限流）时优雅降级；
  4. 等权全市场基准的成分股覆盖度守卫（稀疏交易日会把基准拉偏）；
  5. .streamlit/config.toml 的两处用户可见回归（headless / toolbarMode）与主题可切换。
"""

import ast
import tomllib
from pathlib import Path

import pandas as pd
import pytest


PROJ = Path(__file__).resolve().parents[1]

# 只能放在全局 [theme]、放进 [theme.light]/[theme.dark] 不生效的键。
# 清单实测自本机 streamlit/config.py 里 _create_theme_options 的 categories
# （不是照文档记忆写的）：52 个 theme 选项里这 10 个不含 light/dark 分类。
GLOBAL_ONLY_THEME_KEYS = {
    "base", "baseFontSize", "baseFontWeight", "fontFaces",
    "metricValueFontSize", "metricValueFontWeight", "showSidebarBorder",
    "chartCategoricalColors", "chartSequentialColors", "chartDivergingColors",
}


def _fake_quote_text(url: str) -> str:
    """按请求的代码生成腾讯格式假行情原文（指数 + 个股）。

    接缝是底层 HTTP（``ashare_quant.realtime._get``）而不是某个行情库：
    行情实现已不再依赖 easyquotation（它把接口写死成 http，2026-09-18 起两家
    都返回 400），继续 patch 旧接缝会让冒烟测试真的去联网。
    """
    codes = url.split("=", 1)[1].split(",")
    lines = []
    for c in codes:
        if c in ("sh000001", "sz399001", "sz399006", "sh000300"):
            name, now, prev = c, 3000.0, 2980.0
        else:
            name, now, prev = "测试股", 10.0, 9.9
        f = [""] * 88
        f[0] = f'v_{c}="1'
        f[1] = name
        f[2] = c[-6:]
        f[3] = str(now)
        f[4] = str(prev)
        f[5] = "9.8"
        f[6] = "100"
        f[33] = "10.1"
        f[34] = "9.7"
        lines.append("~".join(f))
    return ";".join(lines) + ";"


def _patch_quotes(monkeypatch):
    monkeypatch.setattr("ashare_quant.realtime._get", _fake_quote_text)


def _dashboard_pieces(*names):
    """按 AST 从 dashboard.py 里摘出指定顶层常量/函数并执行。

    dashboard.py 是 Streamlit 脚本：直接 import 会执行整页（包括会联网的
    指数快照），因此这里只把被测代码摘出来跑 —— 测的仍是线上那一份源码。
    """
    tree = ast.parse((PROJ / "dashboard.py").read_text(encoding="utf-8"))
    wanted = set(names)
    picked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            picked.append(node)
        elif isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) in wanted for t in node.targets):
            picked.append(node)
    found = {getattr(n, "name", None) for n in picked}
    found |= {getattr(t, "id", None) for n in picked if isinstance(n, ast.Assign)
              for t in n.targets}
    missing = wanted - found
    assert not missing, f"dashboard.py 中找不到：{sorted(missing)}"
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=picked, type_ignores=[]), "dashboard.py", "exec"),
         namespace)
    return namespace


def test_dashboard_smoke(monkeypatch):
    _patch_quotes(monkeypatch)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJ / "dashboard.py"), default_timeout=180)
    at.run(timeout=180)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 8

    # 免责声明必须一直可见（顶部标题区 + 页脚）
    texts = [m.value or "" for m in at.markdown] + [c.value or "" for c in at.caption]
    assert any("不构成投资建议" in t for t in texts)

    # KPI 卡片是自绘 HTML，渲染失败会静默丢卡片 —— 这里显式断言它存在
    assert any("kpi-value" in t for t in texts)


def test_dashboard_smoke_cross_refresh(monkeypatch):
    """跨刷新缓存路径回归：st.cache_data 嵌套曾导致 KeyError，连续运行必须稳定。"""
    _patch_quotes(monkeypatch)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJ / "dashboard.py"), default_timeout=180)
    for _ in range(3):
        at.run(timeout=180)
        assert not at.exception, [e.value for e in at.exception]


def test_dashboard_all_tabs_render(monkeypatch):
    """8 个标签页全部渲染出内容（含各自页面标题），且全程无异常。"""
    _patch_quotes(monkeypatch)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJ / "dashboard.py"), default_timeout=180)
    at.run(timeout=180)
    assert not at.exception, [e.value for e in at.exception]

    labels = ["总览", "模拟盘", "今日决策", "账户", "实时行情", "算法对比", "调整日志", "数据状态"]
    headings = ["系统状态与快速操作", "模拟盘对比", "今日模拟投资决策",
                "模拟账户净值", "实时行情", "算法表现对比", "反馈调整日志", "数据状态"]
    assert len(at.tabs) == len(labels)
    for i, (tab, heading) in enumerate(zip(at.tabs, headings)):
        assert len(tab.markdown) >= 1, f"第 {i + 1} 个标签页（{labels[i]}）没有任何内容"
        assert any(heading in (m.value or "") for m in tab.markdown), \
            f"第 {i + 1} 个标签页（{labels[i]}）缺少页面标题「{heading}」"


def test_dashboard_realtime_empty_quotes(monkeypatch):
    """行情源返回空（非交易时段/限流）时必须优雅降级并给出本地收盘兜底。"""
    _patch_quotes(monkeypatch)
    monkeypatch.setattr("ashare_quant.realtime.snapshot", lambda codes: pd.DataFrame())
    monkeypatch.setattr("ashare_quant.realtime.index_snapshot", lambda: pd.DataFrame())
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()  # 避免命中上一个用例缓存到的快照
    at = AppTest.from_file(str(PROJ / "dashboard.py"), default_timeout=180)
    at.run(timeout=180)
    assert not at.exception, [e.value for e in at.exception]

    rt = at.tabs[4]
    texts = ([m.value or "" for m in rt.markdown]
             + [x.value or "" for x in rt.warning]
             + [c.value or "" for c in rt.caption])
    assert any("未获取到实时行情" in t for t in texts), "空行情时应给出降级提示"
    if (PROJ / "data" / "tencent" / "panels" / "close.parquet").exists():
        assert any("本地日线口径" in t for t in texts), "有本地面板时应给出本地收盘兜底表"


def test_equal_weight_bench_drops_sparse_dates():
    """等权全市场基准：成分股覆盖不足的交易日必须被剔除。

    2026-09-16 实测面板只有 209/5360 只有当日收盘（当日均价 12.01 元 vs
    前一交易日 28.27 元），原实现会算出 -57% 的假暴跌。
    """
    fn = _dashboard_pieces("BENCH_MIN_COVERAGE", "equal_weight_bench")
    bench = fn["equal_weight_bench"]
    idx = pd.date_range("2026-09-10", periods=4, freq="D")
    full = pd.DataFrame(
        {"a": [10.0, 11.0, 12.0, 13.0],
         "b": [20.0, 21.0, 22.0, 23.0],
         "c": [30.0, 31.0, 32.0, 33.0],
         "d": [40.0, 41.0, 42.0, 43.0]}, index=idx)
    # 最后一天只剩 1 只股票更新（占比 25% < 阈值 50%）
    sparse = full.copy()
    sparse.loc[idx[-1], ["b", "c", "d"]] = float("nan")

    series, dropped = bench(sparse, idx[0])
    assert dropped == 1
    assert len(series) == 3
    assert series.index.max() == idx[-2]
    # 基准应停在整片成分股都在的 09-12，而不是被单只股票带偏
    assert series.iloc[-1] == pytest.approx(full.loc[idx[-2]].mean())

    # 覆盖充足时不剔除任何一天
    series_full, dropped_full = bench(full, idx[0])
    assert dropped_full == 0
    assert len(series_full) == len(full)


def _streamlit_config() -> dict:
    with open(PROJ / ".streamlit" / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def test_streamlit_config_regressions():
    """守住两处「用户可见」的配置回归，并确认主题真的可切换。

    ① `[server] headless = true`：headless 的语义是「不要自动打开浏览器」，
       而这是全局配置 —— 一设就会把 scripts/start.ps1 与 start_dashboard.ps1
       的自动弹浏览器一起掐掉（那两个脚本自己不传 --server.headless），
       表现为「点了仪表盘没反应、得手动输网址」。截图/CI 请在命令行显式传参。
    ② `[client] toolbarMode = "minimal"`：官方语义是「没有剩余选项就隐藏整个
       菜单」，实测 stMainMenu / stToolbar 节点直接消失 —— 而主题切换器就在那个
       ⋯ 菜单里，于是用户根本切不到主题。用 "viewer"：照样藏掉 Deploy 等入口。
    """
    cfg = _streamlit_config()
    assert "headless" not in cfg.get("server", {}), (
        "[server] headless 会让启动脚本不再自动打开浏览器，用户必须手输网址；"
        "需要 headless 请在命令行传 --server.headless true")
    assert cfg.get("client", {}).get("toolbarMode") != "minimal", (
        'toolbarMode = "minimal" 会连 ⋯ 设置菜单一起隐藏（主题切换器在里面）；用 "viewer"')


def test_streamlit_theme_is_switchable():
    """浅色/深色两套主题都要在，且键放在合法的表里、暗色仍守 A 股红涨绿跌。"""
    theme = _streamlit_config().get("theme", {})
    assert "light" in theme and "dark" in theme, \
        "需要同时定义 [theme.light] 与 [theme.dark]，用户才能切换主题"
    for mode in ("light", "dark"):
        table = theme[mode]
        misplaced = GLOBAL_ONLY_THEME_KEYS & set(table)
        assert not misplaced, (
            f"[theme.{mode}] 里的 {sorted(misplaced)} 是全局项，放子表不生效，"
            "应移到 [theme]")
        # 暗色同样要「红=涨、绿=跌」，只是把两个色都提亮以适配深底
        assert "redColor" in table and "greenColor" in table, \
            f"[theme.{mode}] 需显式给出 redColor/greenColor（A股红涨绿跌口径）"


def _probe_source() -> str:
    """摘出 THEME_PROBE_JS 的脚本文本（不 import dashboard.py —— 它是 Streamlit 脚本，
    import 会直接执行整页渲染）。"""
    src = (PROJ / "dashboard.py").read_text(encoding="utf-8")
    return src.split("THEME_PROBE_JS = r", 1)[1].split('"""', 2)[1]


def test_theme_probe_does_not_read_transparent_container():
    """探针必须从【有不透明底色】的元素读明暗，不能直接读容器。

    2026-09-16 实测踩到的坑：`[data-testid="stAppViewContainer"]` 的 background 是
    ``rgba(0,0,0,0)``（全透明）。直接读它 → 拿到 (0,0,0) → 亮度 0 → **恒判 dark**。
    症状不是"首屏锁错"而是恒定输出 dark：浅色页面配暗色卡片（深字压暗底、标题看不见），
    点 Dark 时反倒恰好蒙对，看起来像"切换生效了"。
    实测真正承载主题底色的是 ``document.body``（浅 rgb(245,247,250) / 暗 rgb(13,17,23)）。

    边界（别把这条当万能）：本测试只钉住**源码里的结构不变量**，不执行 JS。
    行为验证在 ``.dsh-ui-shots/verify_theme_v4.py``：playwright 真实点击
    ⋯ → Dark → Light → Dark，逐步断言期望主题（pytest 里没有浏览器，跑不了那一段）。
    """
    probe = _probe_source()
    assert "m[3]" in probe and "parseFloat" in probe, (
        "探针必须读取 alpha 通道：全透明背景（rgba(0,0,0,0)）不能被当成黑色，"
        "否则会恒判 dark")
    assert "parentElement" in probe, (
        "探针必须沿父链向上找第一个不透明底色 —— stAppViewContainer 是全透明的")
    assert "return null" in probe, (
        "一路全透明读不到底色时必须返回 null（保持服务端给的值），"
        "不能用猜出来的值覆盖正确的首屏配色")
    src = (PROJ / "dashboard.py").read_text(encoding="utf-8")
    for sel in (':root[data-dsh-theme="light"]', ':root[data-dsh-theme="dark"]'):
        assert sel in src, f"令牌层缺少 {sel} 规则，探针写了属性也没人用"


def test_theme_probe_degrades_when_components_html_is_gone(monkeypatch):
    """`st.components.v1.html` 是**已弃用** API（1.61.1 起提示换 st.iframe，
    而 st.iframe 只收 URL、st.html 不执行 <script>）—— 将来被移除时不能让整页崩。

    这里把 `components.html` 换成一个"已移除"的桩（抛 AttributeError），
    断言 `inject_theme_probe()` 安静跳过；旧代码（直接调用）会抛出去。
    """
    import importlib
    import sys

    if "dashboard" not in sys.modules:
        # 本文件前面的 AppTest 用例通常在同一个进程里已把页面跑过一遍；
        # 单独跑本文件时手动加载一次（等于再跑一遍页面，几秒，只发生在这一条用例里）。
        sys.path.insert(0, str(PROJ))
        importlib.import_module("dashboard")
    dash = sys.modules["dashboard"]

    class _Removed:
        def html(self, *a, **k):
            raise AttributeError("module 'streamlit.components.v1' has no attribute 'html'")

    monkeypatch.setattr(dash, "components", _Removed())
    dash.inject_theme_probe()             # 不得抛出

    # 反向：能调用时必须真的注入（别把探针整个删掉还满分）
    calls = []

    class _Alive:
        def html(self, *a, **k):
            calls.append((a, k))

    monkeypatch.setattr(dash, "components", _Alive())
    dash.inject_theme_probe()
    assert len(calls) == 1 and calls[0][1].get("height") == 0
