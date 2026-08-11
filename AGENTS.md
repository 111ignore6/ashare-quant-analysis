# A股量化研究·模拟分析系统 — 项目知识库

> 本文件是项目级长期记忆：任何对话（包括新开会话）操作此仓库时，
> 先读这里，避免重复踩坑。模拟研究，不构成投资建议。

## 项目是什么

A 股全市场量化模拟研究系统：多源行情缓存（Parquet）→ 特征/模型（6 个 ML 模型）
→ 每日模拟决策 → 账户净值跟踪 → Streamlit 仪表盘。纯模拟，不构成投资建议。

## 数据源（重要结论，实测过）

当前主源 **mootdx（通达信 TCP）**，备源链 **mootdx → tencent → akshare**
（`config.yaml` 的 `data_source` / `fallback_sources`，注册表在
`ashare_quant/fetchers/registry.py`）。

- **mootdx 支持北交所**（v0.8.7+）：必须用新代码段 **920xxx**，走通达信
  市场号 **2（MARKET_BJ）**。mootdx 库自己的 `get_stock_market()` 会把 920
  误判为沪市导致返回空——`mootdx_fetcher._bj_bars()` 已绕过（`get_security_bars(9, 2, ...)`）。
  旧号段 43/83/87 已迁移作废，会返回空/僵尸数据，不要用。
- **腾讯直连**：快（22只/s）但高频会被 WAF 风控（返回 501 反爬页）。遇到 501
  必须抛异常（`tencent_fetcher._kline` 已处理），**不能静默当"无数据"**，否则
  会把整批股票误判成停牌写进冷却表。封禁通常几小时后自动解除。
- **akshare（新浪源）**：慢（~1.7s/只），且**北交所 920 没有当日数据**（只有前一日）。
  只能作最后兜底。
- **东财系（efinance / akshare 东财 / adata）与百度股市通**：当前网络环境不可达
  （403 / ConnectionError），不要浪费时间去接。
- **指数**：mootdx `index()` 接口（000300 等）；不依赖腾讯。

## 关键机制（改代码前先理解）

- `daily.py::update_daily`：指数先更新 → 落后股票增量 → 失败冷却表
  `update_failed.json`（当日失败不重试，次日自动恢复）。
  - 盘中（`market_session()` 返回 am/lunch/pm）**不等待备源、no_data 不写冷却**
    （备源当日数据未生成，等也白等）；收盘后才走备源链 + 写冷却。
  - 批量层退避只针对 `failed`（风控/断连），**不针对 no_data**（停牌/未生成是正常）。
  - 非终端环境（重定向/仪表盘后台）tqdm 会卡死批量循环，代码用
    `sys.stderr.isatty()` 判断，非 tty 禁用进度条改每 200 只打印。
- `mootdx_fetcher`：**必须线程本地连接**（全局单例并发只有 4只/s 且大量失败；
  线程本地 + xdxr 缓存 ≈ 43只/s）；除权信息 `xdxr` 用模块级缓存。
- `portfolio.py::equity_curve`：决策段覆盖 **(d0, d1]**（含换仓日 d1 当天收益）；
  写成 (d0, d1) 会让连续决策时账户曲线空白。
- 仪表盘 `dashboard.py`：**禁止在 `st.cache_data` 函数里嵌套调用另一个
  `st.cache_data` 函数**（跨刷新会抛 KeyError，见 load_decision 的教训）；
  同一页面多处实时估值共用 `_cached_snapshot`（5s TTL）+ 统一按侧边栏资金预览。
- 决策文件多目录兼容：仪表盘读日期最新的一份（`load_decision`），
  计划任务历史可能写 `docs/simulation`。

## 常用命令

```powershell
python -m ashare_quant.cli daily     # 增量更新 + 报告 + 决策（16:05 计划任务）
python -m ashare_quant.cli fetch     # 全量下载（幂等，已有则跳过）
python -m ashare_quant.cli simulate  # 模拟盘回测
python -m ashare_quant.cli decision  # 训练/加载模型 + 今日决策
python -m streamlit run dashboard.py # 仪表盘（端口 8501）
python -m pytest tests/ -q           # 全量测试（当前 114+ 项，必须全绿再提交）
python -m ruff check .               # 静态检查
```

## 工程约定

- Windows PowerShell 5.1 读中文 .ps1 需要 **UTF-8 BOM**；apply_patch 生成的文件
  无 BOM，写完 .ps1 用 `[System.IO.File]::WriteAllText($p, $c, (New-Object System.Text.UTF8Encoding $true))` 补。
- apply_patch 编辑文件；git commit 消息避免双引号（PowerShell 拆参数）。
- 删除/覆盖类操作先备份（如 `update_failed.json` → `.bak`）。
- 测试：改逻辑必须加/改测试并跑全量；仪表盘改动至少跑 `tests/test_dashboard_smoke.py`
  （含跨刷新回归）。
- 数据目录 `data/tencent`：5332 只 × 3 年日线 + `panels/` 面板缓存 +
  `portfolio/` 账户历史；`update_stats.json` 记录最近一次每日更新耗时。

## 当前状态（2026-08-11）

- 全市场 5332 只日线已到 2026-08-11；账户正式决策自 08-10 起（2 次）。
- 计划任务 AshareQuantDaily（周一至五 16:05）默认开启；若需更新任务
  out-dir 用管理员跑 `scripts/schedule_daily.ps1 -Force`。
