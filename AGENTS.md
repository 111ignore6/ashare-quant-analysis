# A股量化研究·模拟分析系统 — 项目知识库

> 本文件是项目级长期记忆：任何对话（包括新开会话）操作此仓库时，
> 先读这里，避免重复踩坑。模拟研究，不构成投资建议。
>
> ## 关于本文件（公开版说明）
>
> 这是一个**由 AI 维护的项目**，本文件是 AI 维护者在长期迭代中沉淀下来的工程日志。
> 它的写法刻意保留了**原始证据强度**，而不是写成"产品文档"式的成功叙事：
>
> - **每条结论都标注它是「实测 / 引用 / 推断 / 未证实」中的哪一种**；
>   被后续证据推翻的旧结论**不删除**，而是标注作废并保留原文——因为"当初为什么那么想"
>   本身就是防止重犯的材料；
> - **失败、误判、自相矛盾、探针自己说谎**都如实记录（例如「自述指标必须与实测对撞，
>   不能自证」这条约定，就是被一次真实漏报逼出来的）；
> - **未证实的东西明确写成"未证实"**，不给它配行动建议。
>
> 因此文中会出现「一次性探针脚本（未随仓库发布）」「scratch 副本」「隔离会话」这类字样：
> 它们指当时用于独立复核的临时手段，脚本本身不在仓库里，但**结论与判据是可复现的**
> （可复现的部分都写成了可直接粘贴的命令）。
>
> 面向公开读者的**诚实性结论摘要**见 [`docs/HONESTY.md`](docs/HONESTY.md)——
> 如果你只想快速知道"这个项目到底证明了什么、没证明什么"，读那一份就够了。

## 项目是什么

A 股全市场量化模拟研究系统：多源行情缓存（Parquet）→ 特征/模型（注册表内置 16 个模型，
每日决策池取其中 6 个）→ 每日模拟决策 → 账户净值跟踪 → Streamlit 仪表盘。纯模拟，不构成投资建议。

## 数据源（重要结论，实测过）

当前主源 **akshare（新浪）**，备源链 **akshare → tencent → mootdx**
（`config.yaml` 的 `data_source` / `fallback_sources`，注册表在
`ashare_quant/fetchers/registry.py`；指数与个股共用同一条链）。

> 2026-09-10 起主源从 mootdx 换成 akshare：本机探测的 7 台通达信公开服务器
> （TCP 7709）5 台直接超时、能连上的也返回空，而 mootdx 库版本未变
> （0.11.7，8-10 装的），属于被服务端限流/封禁。mootdx 恢复后把 `data_source`
> 改回 `mootdx` 即可（实测它仍是全市场最快：43 只/s vs 新浪 4~7 只/s）。

- **mootdx 支持北交所**（v0.8.7+）：必须用新代码段 **920xxx**，走通达信
  市场号 **2（MARKET_BJ）**。mootdx 库自己的 `get_stock_market()` 会把 920
  误判为沪市导致返回空——`mootdx_fetcher._bj_bars()` 已绕过（`get_security_bars(9, 2, ...)`）。
  旧号段 43/83/87 已迁移作废，会返回空/僵尸数据，不要用。
  ⚠️ **mootdx 的 volume 是"手"且 fetcher 不做归一**（`mootdx_fetcher` 只把 `vol` 改名成 `volume`，
  没有 ×100）—— 这正是 09-10 之前的历史全是"手"、与 akshare 时代差 100 倍的原因。
  **切回 mootdx 前必须先跑 `scripts/fix_volume_unit.py`（dry-run）确认量纲**，或先给 fetcher 补归一；
  科创板在通达信里的单位**未实测过**，别照抄腾讯的结论。
- **mootdx xdxr 混有全 NaN 的公告日行**：`float(row.get("fenhong") or 0)` 在
  NaN 时得到 NaN（NaN 是 truthy），`after <= 0` 拦不住 NaN，会把
  `factor[:pos] *= NaN` 把该事件之前**整个前复权历史抹成 NaN**（2026-08-11
  17:13 重拉后 600000 等大量股票 2025-10-27 前全 NaN、特征缓存缩水 3 倍）。
  已修复：字段按 `pd.notna`→0 解析 + `np.isfinite(after)` 守卫 + 回归测试
  （`test_qfq_adjust_ignores_nan_xdxr_rows`）。改复权逻辑后必须抽查全市场
  历史 NaN 率，别只看最新几行。
- **腾讯直连**：快（22只/s）但高频会被 WAF 风控（返回 501 反爬页）。遇到 501
  必须抛异常（`tencent_fetcher._kline` 已处理），**不能静默当"无数据"**，否则
  会把整批股票误判成停牌写进冷却表。封禁通常几小时后自动解除。
- **akshare（新浪源）**：慢（12 并发实测 4~7 只/s，全市场约 15~25 分钟）；
  北交所 920 号段**盘中只有前一日**，收盘后当日 bar 齐全（09-11 17:00 实测
  920100/920982 都能拿到当日）。与 mootdx 自算前复权在重叠日**逐日 0.00% 偏差**
  （09-11 抽查 30 只 × 3 个重叠日），所以可直接往 mootdx 建的历史里追加。
- **跨源追加必须先审计复权锚点**：前复权以"最新价"为锚，用 B 源往 A 源建的
  历史里追加几天时，若期间某只除权除息，B 的重叠日价格会整体缩放，拼出来就是
  一根假涨跌幅。09-11 实测：5360 只里 **43 只不一致（最大 2.66%）**，而当天
  全市场正好大跌 1.5%~2.2%，假跌幅会被混进真跌里。工具与流程：
  `python scripts/audit_qfq_anchor.py`（比对最近 N 个重叠日，容差 0.1%）→
  `--fix`（用当前主源**整段重建**不一致的个股，不是只补最后几天）→
  `daily --force` 重建面板与决策。**每次换主源后必须跑一遍。**
- **东财系（efinance / akshare 东财 / adata）与百度股市通**：当前网络环境不可达
  （403 / ConnectionError），不要浪费时间去接。`ak.stock_zh_a_hist`（东财）同样不可达，
  `ak.stock_zh_index_daily` / `stock_zh_a_daily`（新浪）可用。
- **指数**：**必须走多源链**（`fetchers.resolve_index_fetchers`，与个股一样
  主源 → 备源，各源都拉一次并取最靠后的一天）。发布时点不同：新浪指数
  **滞后一个交易日**、腾讯当日即有、通达信当日有——只用第一个非空结果会整体
  滞后一天。各源返回后统一按 `probe_start` 裁剪（akshare/通达信不实现 start
  过滤，会给 20 年全量，不裁剪会把指数从 3 年撑成 5991 行、改变日历口径）。
- **链尾的"报价端点"兜底源 `index_snapshot`（2026-09-18 新增，别再删掉它）**：
  能**当日**给到指数的是腾讯的 **K 线端点** `web.ifzq.gtimg.cn`，而它会返回
  **HTTP 501（WAF 反爬页）** —— 实测 09-18 **15:41 还是 200、16:07 变 501**；
  新浪指数天然滞后一个交易日、通达信自 09-10 起不可用。三者叠加的后果是
  **"今天"永远推不动**：09-17 16:05 与 09-18 16:05 两个计划任务连续空转
  （前者三源全空，后者只有 akshare 且停在 D-1），数据停在 09-17。
  但**报价端点是另外的 host**，实测同一时刻（K 线 501 时）仍 **200 且带当日 OHLC**：
  `https://qt.gtimg.cn/q=sh000300`、`https://hq.sinajs.cn/list=sh000300`（腾讯失败自动换新浪）。
  单位实测：报价端点的指数成交量字段是**手**（两家都 `189924166`），本地指数库是**股**
  （09-17 本地 `1.700256e10` == `ak.stock_zh_index_daily` 的 `17002564200`，逐位相同）→ `×100`；
  `amount` 两家都给**元**（`537699359227`）。**盘中安全**：当日那根会被
  `calendar.drop_intraday_today` 丢弃，不会把未收盘价当收盘价（`tests/test_index_snapshot.py`
  11 项 + `test_index_source_chain.py::test_index_chain_always_ends_with_quote_snapshot`；
  把链尾 append 删掉实测**红 2 项**）。

## 关键机制（改代码前先理解）

- `daily.py::update_daily`：指数先更新 → 落后股票增量 → 失败冷却表
  `update_failed.json`（当日失败不重试，次日自动恢复）。
  - **"指数取不到" ≠ "今天没有新交易日"（09-10/09-11 冻结两天的根因）**：
    "最新交易日"完全由指数决定，旧代码在指数空返回时静默沿用上一个截止日并
    返回 `up_to_date="all"`、`failed=0`，于是 16:05 计划任务连续两天"成功"、
    5360 只全市场冻结在 09-09、仪表盘看不到缺口。现在：
    `daily._fetch_index` 沿整条源链取最新一天，**全链都空**才返回
    `index_status="all_sources_empty"` + `error`，CLI 打印 ‼️ 并**不**跑报告/决策；
    `update_stats.json` 增加 `days_behind / healthy / index_status / index_sources
    / primary_index_empty / error`，仪表盘「数据状态」页把这三类情况显示为
    error/warning。判断是否异常用 `cli._data_health()`（交易日数差，已排除周末）。
    ⚠️ **注意口径的演变**：这套自述指标在 2026-09-16 **之前**只覆盖指数层 —— 指数当天推进、
    个股大面积没跟上时它照样报 `healthy`（09-16 就是这么漏报的）。现已补上个股口径，
    见下面「`healthy / days_behind` 过去只看指数、会漏报个股（已修）」；
    `updated` 的"文件被写过"语义另见下一条。
  - **指数探测起点用"上一个交易日"而不是"次日"**：存活的源至少会返回上一交易日
    那根 bar，因此"全部空返回"只可能是源挂了，两种"没拿到新数据"从此可区分。
  - **`update_stats.json` 的自述指标会在个股层失效（2026-09-16 实测，是"冻结两天"的新变种）**：
    `updated` 的真实语义是"**这个 parquet 今天被写过**"，不是"拿到了新交易日的 bar"；
    `healthy / days_behind / index_status` 全部只由**指数**日期推导，**完全不校验个股层**。
    09-16 实测自述：`updated=4989 / up_to_date=0 / failed=0 / no_data=371 / healthy=true /
    index_date=2026-09-16`；但逐文件读 parquet 的日期列统计是：
    **数据停在 09-15 的有 5140 只，真到 09-16 的只有 209 只**（按代码段 000=206、002=3）。
    交叉表（数据截止 × 写入日）：`09-15 数据 × 16 时写过 = 4780`、`09-16 数据 × 16 时写过 = 209`；
    当天 16 点后被写过的文件共 **4991** = `updated(4989)` + 2 个非个股文件
    （`features.parquet`、`sh000300.parquet`），而 `no_data(371) = 5362 − 4991` —— 三个数完全自洽。
    → **不是"没取到"，是取回来了、没有新 bar 也照样重写文件、照样计进 `updated`。**
    后果：16:05 生成的面板是"指数=09-16、约 96% 个股=09-15"的**混合日期横截面**，
    当天的决策就建立在这个面板上（`data/tencent/features.meta.json` 记 `as_of=2026-09-16`）。
    - 判断"今天到底更新成功没有"，**不要只看 `update_stats.json`**，也不要只看
      `cli._data_health()`（它同样只看指数）。直读每个 parquet 的**元数据统计**
      （`pyarrow.parquet.ParquetFile(f).metadata` 的 date 列 min/max，不必加载数据，
      全市场 5362 个文件约 2 秒），看**数据截止日的分布**。`manifest.json` 的 `end`
      实测与 parquet 实际 max 一致（5361 条比对 0 处不一致）——manifest 没坏，
      坏的是"文件被重写就记成已更新"的记账口径。
    - **实测**：以上全部数字由两个一次性探针得出（`verify_data_health2.py`、
      `probe_mtime.py`，**属当时的临时脚本、未随仓库发布**），另用上面的元数据统计法
      独立复核，逐项一致。
    - **未证实（不要写成结论）**：为什么 09-16 16:05 只有 209 只（全部 `00` 开头）拿到了当日 bar；
      以及"新浪源 16:05 尚未发布当日 bar"这个时序归因 —— 目前只有 17:40 能取到当日数据的观测，
      加上 `sh000300.parquet` mtime=16:05:12 的旁证，**没有 16:05 时刻的直接采样**。
    - **时效说明**：上面的分布数字是 2026-09-16 那次运行的快照，修好流水线后数字会变；
      但"`updated` 只代表文件被写过、`healthy/days_behind` 只看指数"这个**机制结论与修复无关**，
      改完 daily 也要按上面的元数据统计法重新确认一次，别只看 `update_stats.json`。
      （09-16 18:15 生产补齐后已按此法复测：5350 只到 09-16、仅 10 只真停牌，见「当前状态」。）
  - **"非空即 updated" 是 2026-09-16 事故的根因（已修）**：`daily._fetch_attempt`（旧 daily.py:247 起）
    用 `if not df.empty: return "updated"` 判定成功，**从不校验 df 是否覆盖到目标交易日 `last`**。
    新浪个股当日 bar 滞后数小时，16:05 跑时只返回 D-1 那根 → 非空 → 判"已更新" → 文件被重写而
    数据没有前进一天；而备源链（daily.py:272-293）**只在空返回/异常时才走**，于是永远不会被调用。
    修法：`return "updated" if df.index.max() >= last else "no_data"`（daily.py:265）。
    未覆盖目标日的股票现在会：不判 updated → 继续走备源（腾讯当日即有）→ 都不行才 no_data +
    收盘后写冷却。回归测试 `tests/test_daily_recovery.py::test_main_source_returning_older_bar_is_not_updated`
    （把修复行还原到 scratch 副本上跑，实测必红）。
    ⚠️ **未验证**：这些改造消除了对"新浪当日可得性"的依赖（改走腾讯批量报价），但
    "**16:05 那一刻**腾讯/新浪是否都已发布当日 bar"只在 17:24 之后实测过，**没有 16:05 时刻的
    直接采样**；不要写成"16:05 必然成功"，最终证据要靠生产环境补齐实验给出（见「当前状态」）。
  - **`healthy / days_behind` 过去只看指数、会漏报个股（已修）**：`cli._data_health` 现在还要求
    "预期外落后 ≤ max(10, 1% 股票数)"；`update_stats.json` 新增 `stocks_total / stocks_behind /
    stocks_behind_expected / stocks_behind_unexpected / completeness / stocks_ok`。
    `new_data` 现在要求"确实有股票拿到目标交易日 bar 且 completeness ≥ 50%"，否则 cli
    **跳过报告与决策重算**（不再在"指数 D + 个股 D-1"的塌缩面板上出决策）。
    `index_status / days_behind` 的既有语义未改（它们本来就只描述指数 —— 那正是 09-16 报 healthy 的原因）。
  - **⚠️ `no_data` 不是"停牌数"，冷却表也不是停牌集合**：2026-09-16 修复前那次 `no_data=371`
    （`update_failed.json` 378 条，值是**进入冷却的日期**，不是原因字段：
    `{'2026-09-16': 371, '2026-09-14': 4, '2026-09-11': 3}`），**抽样直接问 akshare/sina：
    30/30 都返回 `last=2026-09-16`**（逐只直问源实测）→ 这 371 只**不是停牌**，是被上面那条
    "非空即 updated + 备源不触发"缺陷误判的正常股票。
    所以 **"冷却中的代码" ≠ "确实停牌"**：任何把冷却表当停牌集合用的口径（含仪表盘的 `expected`）
    都可能被环境误判污染。要不要判真停牌必须另证（腾讯报价 `volume=0`，或直问源）：
    独立复核（另一次隔离会话，在 scratch 副本上）量到 **6 只真停牌**（301390/600825/603159/600301/601238/605303，
    volume=0）、scratch 冷却表共 **10 条**；而**生产当日是 371 条**（其中抽样 30/30 并非停牌）。
    历史记载的 09-11 真停牌是 **8 只**（002870/301390/600825/600929/603159/605577/688291/688432）。
    **这几个数都不要当成"今天的停牌集合"** —— 口径或日子一变就不一样。
  - **盘中（pre/am/lunch/pm）不允许把"今天"的未收盘 bar 当最新交易日推进**
    （`calendar.drop_intraday_today`）：mootdx 盘中可拉当日 bar，但那是盘中价，
    写入后会把决策日期推到当天、污染日线/账户（08-12 中午"缺少决策日基准"
    和 08-11 +0.59% 都是这个根因）。盘中点「每日更新」= 白点，收盘后
    （16:05 计划任务）才正式更新。
  - 手动回滚盘中污染的标准流程：备份小文件 → 批量截掉 `*.parquet` 当日行
    → 移走 panels/features 缓存 → 账户历史去掉当日条目 → `daily --force`
    重建 → **`fetch` 重建 manifest**（否则 manifest 仍显示当日，16:05 会跳过）。
  - **面板缓存指纹只含 manifest 元数据（起止/行数）**，数据回滚后可能指纹
    相同但面板内容滞后（08-12 曾复用 08-11 旧面板导致计划任务决策卡在
    08-11）。已修复：`_save_panel_cache` 记录面板实际最后日期，
    `_load_panel_cache` 校验其 == manifest 指数截止，不符即重建。手改缓存/
    回滚后建议直接移走 panels/ 目录强制重建。
    **（2026-09-18 补强）** 上面两条**都拦不住"值被改写而日期不变"**：修复科创板 volume 时
    改了 604 个 parquet 的数值、日期一行没变 → 缓存被判有效，`daily --force --retrain`
    实际用的仍是被污染的面板（`features.meta.json` 指纹一字未变即铁证）。
    现在 `_load_panel_cache` 还比对 **`source_signature`**（= manifest 指纹 +
    每个个股 parquet 的 `mtime_ns/size`，`pipeline._source_signature`，全市场实测
    ~0.6 秒；旧 meta 无此字段一律视为失效，fail-safe）。
    回归测试 `tests/test_pipeline.py::test_panel_cache_invalidates_when_values_change_without_dates`
    —— **注意该测试第一次写错了**：忘了建指数 manifest，导致 `idx_end` 为空、
    缓存**永远**失效，于是在旧代码上也全绿（一根不会熔断的保险丝）。
    补上 `store.save("sh000300", ...)` 并先断言 `_load_panel_cache(...) is not None`
    后，还原旧行为实测**必红**。
  - **特征缓存犯了与"面板缓存指纹"同款的错（2026-09-16 生产实测复发，已修）**：
    `ml/features.py::load_feature_cache` 旧版**只校验 `as_of` 日期相等**
    （`save_feature_cache` 明明写了 `n_rows`，却从没校验过）。09-16 18:13 补齐数据后，
    面板最后一日横截面从 204 只恢复到 5350 只、**而日期没变** → `as_of` 相等 →
    旧特征缓存被判有效 → 决策在**塌缩的旧特征表**上重算：`decision.json` 的 50 只 picks
    与修复前**逐字相同**（连 rank_score 都一样）。注意每日决策读的是 `features.parquet`、
    **不是 `panels/`**，所以"移走 panels/ 强制重建"治不好这个。
    修法：新增 `panel_fingerprint(close, volume, index_close, horizon, require_target)`，
    对**输入内容**（索引/列名/形状/数值）做 sha256，外加 `FEATURE_CACHE_VERSION`
    （改 `build_dataset` 的特征列/窗口/语义时**必须手动 +1**）；`load_feature_cache` 现在依次校验
    日期 → 版本 → 指纹 → 元数据与文件自洽（有 `target` 列且行数 == `n_rows`），
    **没给指纹一律拒绝**（fail-safe：无法验证就不复用，代价是 ~25s 重算，而不是静默用错数据出决策）。
    每日决策改走 `load_or_build_dataset(...)`，读写同一份判据。指纹成本实测 ~20ms/张（752×5360 面板），
    相对建表 25s 可忽略。
    **覆盖边界（按实现读出来的，别当万能）**：指纹只覆盖"输入面板内容 + horizon +
    require_target + 特征定义版本"；若改了 `build_dataset` 的代码却忘记 bump
    `FEATURE_CACHE_VERSION`，指纹不变、旧缓存仍会被复用。
  - 盘中（`market_session()` 返回 am/lunch/pm）**不等待备源、no_data 不写冷却**
    （备源当日数据未生成，等也白等）；收盘后才走备源链 + 写冷却。
  - 批量层退避只针对 `failed`（风控/断连），**不针对 no_data**（停牌/未生成是正常）。
  - 非终端环境（重定向/仪表盘后台）tqdm 会卡死批量循环，代码用
    `sys.stderr.isatty()` 判断，非 tty 禁用进度条改每 200 只打印。
- **缓存判据不能被自己刷失效（2026-09-18 修 `cache.append`）**：`daily` 每次都会
  `store.append(index_symbol, idx_df)`，而旧版 `append` **无条件 `save`** —— 指数即使没有
  新交易日也会重写 parquet，于是 mtime/size 变化 → `pipeline._source_signature` 随之变化 →
  面板缓存**每次运行都判失效**（重建 ~25 s），特征表也跟着重建：
  实测 17:20 与 18:01 两次真实运行都打印"本次重建特征表"，而两次的面板内容完全一致。
  现在 `append` 在 `old.equals(merged)` 时**不写盘**（只补 manifest）。
  实测（18:09 一次 `--force` 运行）：`panels/meta.json` 与 `features.meta.json` 的 mtime
  **保持不变**（仍是 18:00:44 / 18:01:33）⇒ 缓存真的被复用了；`build_panels` 从 ~25 s 降到
  **3.0 s**，加上不再重建特征表，每次"内容没变"的运行省 ~50 s。
  ⚠️ 判据强度没被削弱：**数值真变了仍会落盘**（反向断言写在测试里）。
  验收：`tests/test_pipeline.py::test_reappending_identical_index_keeps_panel_cache_hit`、
  `tests/test_cache.py::test_append_same_content_does_not_touch_file`，
  还原旧行为实测**红 2/2**。
- `mootdx_fetcher`：**必须线程本地连接**（全局单例并发只有 4只/s 且大量失败；
  线程本地 + xdxr 缓存 ≈ 43只/s）；除权信息 `xdxr` 用模块级缓存。
- `portfolio.py::equity_curve`：决策段覆盖 **(d0, d1]**（含换仓日 d1 当天收益）；
  写成 (d0, d1) 会让连续决策时账户曲线空白。
- 仪表盘 `dashboard.py`：**禁止在 `st.cache_data` 函数里嵌套调用另一个
  `st.cache_data` 函数**（跨刷新会抛 KeyError，见 load_decision 的教训）；
  同一页面多处实时估值共用 `_cached_snapshot`（5s TTL）+ 统一按侧边栏资金预览。
- **实时估值基准 = 决策日累计净资产（`portfolio.account_basis`）**，不是初始资金。
  用初始资金会在每个决策日把「实时总资产」重置回 10 万（决策日收盘建仓，
  成本=现价、本期收益恒为 0），与账户页累计净值（98,599）对不上。账户页/
  总览/实时行情三处口径统一：实时总收益=总资产/初始资金-1（累计），
  本期浮动盈亏=现价相对决策日成本；绩效指标不足 20 个交易日不展示。
- **仪表盘 `st.cache_data` 必须设 TTL（≤60s）**：面板/账户/决策文件每日更新后
  内容会变，无 TTL 会整个会话用旧数据（账户 CSV 曾可能滞留一整天）；TTL 太长
  （如 600s）也会在更新后短暂出现"决策日期 > 面板截止"的假警告。
  **验收边界（2026-09-16）**：仪表盘只在 **1600px 宽 + Chromium + 浅色主题**下验收过
  （8 个 tab 渲染无异常）；**窄屏响应式、暗色主题、Firefox/Safari/Edge、以及盘中的
  10s 自动刷新与当日 bar 动态拼接，均未验证** —— 改样式后别只按"上次验过"推断。
- **等权全市场基准必须带"成分股覆盖度守卫"（2026-09-16 修）**：原实现按日取面板非空均值，
  在"面板最后一行只有 209/5360 只有收盘价"的中间态下算出 **-57% 的假暴跌**
  （当日均值 12.01 元 vs 前一日 28.27 元）。现在要求**当日有效成分股 ≥ 全市场 50%** 才纳入基准，
  并在图上标注"已剔除 N 个覆盖不足的交易日"；**账户自身净值/收益率/绩效一个都没动**，只动对比基准。
  回归测试 `tests/test_dashboard_smoke.py::test_equal_weight_bench_drops_sparse_dates`。
  —— 这是"面板中间态"（见「已知问题」3）的第一个已知受害者；**任何"按日取全市场均值"的下游
  都要按同一标准自查**。
- **计划任务输出要落日志（2026-09-16 改版）**：计划任务现在调 `scripts/run_daily_logged.ps1`，
  日志写 **UTF-8（无 BOM）** 到 `logs\daily_scheduled.utf8.log`，并显式 `--model-dir models\all`。
  **不要再回到 `& python ... *>> logs\xxx.log 2>&1`**：PS 5.1 的 `>>` 写 UTF-16LE，且用 GBK
  解码 python 的 UTF-8 stdout，中文在落盘前就已损坏（历史 `logs\daily_scheduled.log` 即如此）。
  机制、判据与历史文件处置见「已知问题」第 2 条。
  - **改完 .ps1 还要让已注册任务重新指向新命令才算生效**（任务里存的是命令字符串副本，
    改脚本不会自动生效）：`powershell -ExecutionPolicy Bypass -File scripts\schedule_daily.ps1 -Force`。
    实测该任务 Principal = 当前用户 / LogonType=Interactive / RunLevel=Limited；
    `-Force` 已改为 `Set-ScheduledTask` **原地更新**，不再先删后建（旧写法中途失败会永久丢任务，
    见 08-14 事故）。**2026-09-16 实跑 `-Force` 成功**（原地更新、无任务空窗），
    `Get-Command python` 在 `-NoProfile` 非交互 PS 里实测解析到 Anaconda 的 `python.exe`；
    是否需要管理员**未测定**（执行环境是否提权未知），失败只会报 Access denied，不破坏任务。
  - ⚠️ **`New-ScheduledTaskTrigger -Weekly` 会重建触发器，`StartBoundary` 会被改成"执行当天 16:05"**：
    2026-09-16 那次更新后它从 `2026-08-27T16:05` 变成 `2026-09-16T16:05`。触发时间不变、
    下次运行仍是下一个工作日 16:05，但**对账时会看到这个差异**，别以为是被人改过。
    回退依据：更新前的原始 XML 已存档到仓库外的备份目录
    （`<备份目录>/AshareQuantDaily.BEFORE.xml`），
    回退 = `Register-ScheduledTask -Xml (Get-Content <该文件> -Raw) -TaskName 'AshareQuantDaily' -Force`。
- **`daily` 的退出码是"数据侧故障"唯一的对外信号（2026-09-18 新增，`cli.EXIT_DATA_FAILURE = 2`）**：
  约定 `0` = 正常（含"确实没有新交易日"这种非故障空转）、`2` = 数据侧故障
  （主备源全空 / 个股大面积没跟上 / 面板塌缩 —— 都已主动**不**出决策）、`1` = 未捕获异常。
  `main()` 现在把子命令的返回值变成进程退出码（`args.func(args)` 的返回值以前被直接丢掉）。
  **为什么必须这样**：2026-09-17 16:05 那一次（`logs/daily_scheduled.utf8.log` 实录）
  三个指数源**全部空返回**、数据停在 09-16、当天什么都没更新，而
  `Get-ScheduledTaskInfo -TaskName AshareQuantDaily` 的 `LastTaskResult` 仍是 **0** ——
  旧代码只打印 ‼️ 就 `return None`，于是"计划任务一切正常"和仪表盘一样撒同一个谎。
  现在 `update_stats.json` 也带 `exit_code`，自述与实测可对账。
  验收：`python -m pytest tests/test_daily_recovery.py -k exit_code`（6 项，
  把退出码与 `main()` 的传播一起还原成旧行为实测**全红**）。
- **每日决策模型池由 `config.yaml` 的 `models` 字段控制**（2026-08-11 起）：
  空列表回退默认 6 模型。当前池 = lgbm/xgb/rank_lgb/huber_lgb/risk_aware_lgb/
  temporal_decay_lgb；换池必须 `daily --retrain`。
- **决策排序用"截面排名均值融合"**（`ml/decision.py`）：各模型预测转当日
  横截面百分位排名再平均，量纲无关（rank_lgb 分数可安全参与）；`score`
  列是各模型预测的**中位数**（展示用），不是排序依据。改排序逻辑必须
  重跑 `daily --force` 并检查报告里的预期收益量纲是否合理。
- 自研模型基准（全市场 walk-forward，2026-08-11）：risk_aware_lgb 夏普
  2.81 最高、rank_lgb IC 0.097/回撤 -3.6% 最优、huber_lgb 年化 78.5% 最高、
  rank_ensemble 融合年化 77.9%；rank_xgb（IC 0.01）与 mlp_deep 已排除。
  TabPFN v2 需 HF 登录（gated），暂不可用。
- 决策文件多目录兼容：仪表盘读日期最新的一份（`load_decision`），
  计划任务历史可能写 `docs/simulation`。
- **盘中抓取的当日 bar 不是收盘价**：10:35/11:2x 拉取会把 08-11 盘中价
  写成当日 close（如 000779=10.42 地天板低点，真实收盘 12.73）。只要
  `market_session()` 不是收盘后，**不能用当日 bar 做"决策日成本"或估值**；
  实时估值用「决策日成本 × 快照现价」时必须确保成本来自已收盘的决策日，
  否则会复现 2026-08-11 15:58 的 +0.59% 假收益（口径错配：盘中价成本 ×
  收盘价现价，一只 000779 贡献 +0.44pp）。完整溯源见
  `docs/research/2026-08-11-realtime-059-investigation.md`。
- **复现类问题先找"当时的证据文件"再重建**：不要从现有数据反推。
  本次 +0.59% 能精确复现靠的是备份目录 `data/tencent_parquet_bak/` 里
  10:35 写入的旧 `features.parquet`（08-11 当日仅 1880 只有完整特征，
  不是全市场）——先用它重建决策（Top20 与日志完全一致），再用旧面板
  08-11 盘中价 × 修正面板真实收盘 = +0.5915%。git diff 确认决策逻辑
  当天未变；找不到旧特征缓存时"从 parquet 重建"口径会错。
- 腾讯快照：`now`=实时最新价，盘中为最后一笔成交，**15:00 收盘后冻结为
  收盘价**；`close`=昨收。本地不保存分时快照，盘中数字无法事后回放，
  收盘后则等价于收盘价。
- **实时行情已不再经 easyquotation（2026-09-18 修复，曾经整块失效）**：
  该库 0.7.7 把腾讯/新浪接口都硬编码成**明文 http**（`tencent.py:17`、
  `sina.py:26`），而两家现在对 http 一律返回 **400 空响应**（同参数 https
  为 200 且有数据）；库又把空响应当"字段数不足"静默 `continue` →
  **不抛错、返回空 dict**，仪表盘只能显示"可能非交易时段或接口限流"，
  把"接口失效"误报成"休市"。现在 `ashare_quant/realtime.py` 直连 https：
  主源 tencent → 备源 sina 自动降级，**全失败抛 `RealtimeError`**（休市时源
  仍返回最后快照，所以"空"确实等于"坏了"），仪表盘同时给出失败原因 +
  本地收盘兜底表。测试接缝从 `easyquotation.use` 改为 `realtime._get`。
  排查同类问题的最短路径：`https` 与 `http` 各拉一次同一 URL 比状态码。
- **科创板（688/689）成交量单位与全市场相反，别再无条件 ×100**：
  腾讯的 K 线与报价对 688/689 **直接给"股"**，其余板块给"手"（实测
  130 只对撞新浪股数：688 40/40 为股、非 688 90/90 为手）；新浪恒为股。
  唯一事实来源是 `ashare_quant/venues.py::volume_in_shares`，
  `fetchers/tencent_fetcher._parse_kline` 与 `realtime._parse_tencent` 都用它。
  本地历史的量纲台阶（09-10 全市场 手→股、科创板 09-16 额外 100×）已修，见「已知问题」第 4 条。
- **模型库损坏会让每日决策阶段整体崩溃**（08-27 实测）：xgboost 的
  `lib\xgboost.dll`（约 140MB）被删/损坏后，`xgb.joblib` 加载失败 →
  阶段 1/2 正常、阶段 3 崩溃，当天决策缺失（08-26 决策后 08-27 手动运行
  挂掉）。修复：`python -m pip install --force-reinstall --no-deps xgboost`
  （版本以 `pip show xgboost` 为准）。`load_models`（`ml/decision.py`）已改为
  逐模型加载并对失败给出可操作报错（模型名+文件+修复提示），**不静默跳模型**
  （决策悄悄缺模型比崩溃更危险）。若某库彻底不可用：从 `config.yaml` 的
  `models` 移除该模型 → `daily --retrain`；换回后同样 `--retrain`。
- **计划任务可能悄然消失**（08-14 后 AshareQuantDaily 不存在，16:05 不再
  自动运行，期间只有手动 run 维持决策；`daily_scheduled.log` 停更在 08-14）。
  检查：`schtasks /query /tn AshareQuantDaily`（报"找不到文件"= 已消失）；
  恢复：有权限终端跑 `scripts/schedule_daily.ps1 -Force`，注册后验证
  `Get-ScheduledTaskInfo -TaskName AshareQuantDaily` 的 NextRunTime。
- **测试不要硬编码"今天"的日期**：`drop_intraday_today` 按实时钟比较，
  08-12 的回归测试硬编码 08-12 作"今天"，日期一过就挂（08-27 全量测试
  红一个）。涉及当日语义的测试用 `pd.Timestamp.today().normalize() ± 1天`
  构造（见 `tests/test_calendar.py`）。

## 常用命令

```powershell
python -m ashare_quant.cli daily     # 增量更新 + 报告 + 决策（16:05 计划任务）
python -m ashare_quant.cli fetch     # 全量下载（幂等，已有则跳过）
python -m ashare_quant.cli simulate  # 模拟盘回测
python -m ashare_quant.cli decision  # 训练/加载模型 + 今日决策
python -m streamlit run dashboard.py # 仪表盘（端口 8501）
python -m pytest tests/ -q           # 全量测试（当前 224 项，必须全绿再提交）
python -m ruff check .               # 静态检查
python scripts/audit_qfq_anchor.py   # 换数据源后审计前复权锚点一致性（--fix 可整段重建）
python scripts/fix_volume_unit.py    # 成交量量纲体检（dry-run；应始终报"需要修复 0 只"）
python scripts/audit_volume_integrity.py  # 逐 (股票,日期) 成交量完整性审计（--apply/--verify）

# ⚠️ 手动跑 daily / 补数据必须带计划任务的那套路径参数，否则会写进默认
# docs/simulation（决策目录串台、且用错模型池 models/ 而非 models/all）：
# python -m ashare_quant.cli daily --config <abs>/config.yaml `
#   --data-root <abs>/data/tencent --out-dir <abs>/docs/simulation-all `
#   --model-dir <abs>/models/all [--force]
# 另外别用 `| Select-Object -First N` 截断这条命令：PowerShell 会终止上游
# python 进程，决策/报表都写完了但 update_stats.json 不落盘。
```

## 工程约定

- Windows PowerShell 5.1 读中文 .ps1 需要 **UTF-8 BOM**；apply_patch 生成的文件
  无 BOM，写完 .ps1 用 `[System.IO.File]::WriteAllText($p, $c, (New-Object System.Text.UTF8Encoding $true))` 补。
- apply_patch 编辑文件；git commit 消息避免双引号（PowerShell 拆参数）。
- 删除/覆盖类操作先备份（如 `update_failed.json` → `.bak`）。
- 测试：改逻辑必须加/改测试并跑全量；仪表盘改动至少跑 `tests/test_dashboard_smoke.py`
  （含跨刷新回归）。
- 数据目录 `data/tencent`：**5360 个个股 parquet**（2026-09-16 实测；另有
  `features.parquet`、`sh000300.parquet` 两个非个股文件，合计 5362 个 `*.parquet`；
  `universe.json` 记 5358 个代码）+ `panels/` 面板缓存 + `portfolio/` 账户历史；
  `update_stats.json` 记录最近一次每日更新耗时（⚠️ 其数字语义见上文「自述指标会在个股层失效」）。
- **新增的判据/断言，必须先在"还原后的旧行为"上跑出红**，否则它就是一根**不会熔断的保险丝**。
  做法：把修复行还原到 scratch 副本，再跑新测试，确认它真的红（本轮特征缓存 6/8 红、
  daily 早退字段 2/2 红，都是这么自证的）。反面案例也在本轮：一份验收脚本**用自己的产物当判据**
  （自指），故障态下照样报全绿 —— **在最需要它的那天骗过了人**。
- **自述指标必须与实测对撞，不能自证**：`update_stats.json` 的 `healthy=true` 与仪表盘的
  `数据完整率 3.90%` 互相矛盾了整整一个交易日才被发现。验收里应有一条硬断言，例如
  `stats["stocks_behind"] == 用 parquet 元数据统计法算出的实测落后只数`。
- **判据的标准必须写进仓库，不能靠工具的默认值（2026-09-23 事故）**：
  本仓库当时**没有任何 ruff 配置**，`ruff check .` 一直在用 ruff 的默认规则集；
  而 **ruff 0.16 扩大了默认集**，于是同一个 commit、同一份代码：
  **本地（ruff 0.15.13）→ `All checks passed!`；CI（ruff 0.16.8）→ `Found 179 errors.`**
  （CI 里写的是裸 `pip install ruff`，每次都会装最新版）。
  这和上面那条「自述指标不能自证」是**同一类错误**：一条标准如果没被写下来，
  它的含义就会随上游发布**静默漂移** —— "全绿"变成了"取决于你装的是哪个 ruff"。
  修法：`pyproject.toml` 新增 `[tool.ruff]` **显式声明 `select`**，
  并把 ruff 钉在 `>=0.16,<0.17`（`requirements.txt`、dev extra、CI 三处一致）。
  **选规则集的依据来自代码本身**：仓库里本来就有 26 处 `noqa: BLE001`，
  说明作者一直按"要检查盲 except"写 —— 不选 `BLE` 反而会让那些 noqa 变成死注释
  （`RUF100 unused-noqa` 正是这么把它暴露出来的）。
  顺带修掉的真问题：`mootdx_fetcher` 里一处 noqa 写成 `# noqa: BLE001（中文说明）`，
  **全角括号紧跟在规则码后面会让 ruff 无法解析该指令**（一直有 warning，没人看）；
  以及 3 处 B023（闭包捕获循环变量）—— 实测**当前是安全的**（函数只在同一轮迭代内调用），
  但已显式绑成默认参数，免得将来有人把它存起来延后调用时静默拿到下一轮的值。
  ⚠️ 教训：**"本地全绿"不是证据，除非你能说出绿的是哪个版本、哪套规则。**
- **`requirements.txt` 必须保持纯 ASCII（2026-09-23 部署体检发现，是真正的开箱阻断项）**：
  pip 解码 requirements 文件用的是**系统 locale 编码**，不是 UTF-8
  （pip 的 `auto_decode` 只认 BOM，否则用 `locale.getpreferredencoding()`）。
  中文 Windows 的 locale 是 **cp936/GBK**，于是带中文注释的 requirements.txt 会让
  **README 的第一条命令**直接失败：
  `UnicodeDecodeError: 'gbk' codec can't decode byte 0xa1 in position 6`。
  这个坑**只影响中文用户**，英文环境永远复现不出来 —— 而本项目的受众主要就是中文用户。
  实测：`file requirements.txt` = UTF-8 无 BOM + `getpreferredencoding()` = cp936。
  修法二选一，**都实测通过**：① 加 UTF-8 BOM（pip 先读 BOM）；
  ② 注释改纯 ASCII（**采用这条**，无 BOM 副作用、全平台无歧义）。
  中文说明写在 README/AGENTS.md 里，不要写进 requirements.txt。
  同类风险：**任何被工具按"系统 locale"读取的文本文件**（不只是 requirements.txt）。
- **`pip install -e .` ≠ `pip install -r requirements.txt`（同次体检）**：
  前者不含 `[project.optional-dependencies] dashboard` 里的 streamlit，
  于是"按文档装完却起不来仪表盘"。要以包方式装必须写
  `pip install -e ".[dashboard]"`。
- **`config.yaml` 的 `universe_mode: all` 与 README 的 csi300 快速上手不是一回事（同次体检）**：
  照 README 走完（288 只）再跑 `daily`，实测它去拉**全市场 5360 只**
  （`universe.json` 的 `mode` 是 `all`，parquet 从 288 涨到 2182 还在继续），
  首次 15~25 分钟 —— README 里"每日增量约 37 秒"对这批用户不成立。
  已在 README 顶部写成显式警告。

## 当前状态（2026-09-16）

> 本节是 **2026-09-16 的快照**。数据侧一旦有变化（尤其流水线修好、或执行生产补数据），
> 请连同下面的数字一起更新，别让本节变成过期结论。

- **2026-09-18（实时行情修复 + 全市场成交量量纲修复）**：
  ① **实时行情**：`realtime.py` 重写为直连 https + 双源降级 + 失败抛错
  （easyquotation 的 http 被两家 400 拒绝，库又静默返回空 → 实时行情整块失效）；
  ② **成交量量纲**：`tencent_fetcher._parse_kline` 按板块判定单位（科创板不再 ×100），
  并把**历史数据**统一到"股"—— 全市场 4710 只 / 3,388,755 行 ×100（09-10 换源前的"手"）、
  34 行 ÷100；工具 `scripts/fix_volume_unit.py`，备份 `data/tencent_volume_bak_20260918/`；
  ③ **面板缓存判据补强**：新增 `source_signature`（mtime+size），
  让"改数值不改日期"的数据修复也能让缓存失效（否则 `--retrain` 用的是被污染的面板）；
  ④ **面板横截面塌缩修复**（原「已知问题」3；根因=放行判据用"自述完整性"而不是"实测面板覆盖率"）：
  新增 `pipeline.panel_coverage` → 塌缩面板**不落盘**、`cmd_daily` **硬停**不跑报告/决策、
  `screening.covered_dates` 剔除塌缩日期后再算等权基准；生产面板实测 5352/5352、不误报。
  新增/改写测试：`test_realtime.py`（10 项）、`test_tencent_fetcher` 科创板回归、
  `test_dashboard_smoke` 换接缝、`test_pipeline`（面板缓存回归 + coverage 3 项）、
  `test_daily_recovery`（CLI 门禁 2 项）、`test_screening`（基准覆盖度守卫）。
  实测：四大指数 + 持仓 10/10 有数据；成交量对撞新浪股数 4/4 吻合；
  修复后 `features.meta.json` 指纹变化两次（`ce48b973…` → `0b1ba6d2…` → `f9e4018d…`）、
  `vol_ratio` 全市场同量级（0.74~1.41，修复前主板曾因跨台阶虚高到 ~3.5）；
  ⑤ **指数链尾新增"报价端点"兜底源**（`fetchers/index_snapshot.py`，见「数据源」那条）：
  09-17/09-18 连续两天 16:05 数据推不动的根因是腾讯 **K 线**端点被 WAF 拦（HTTP 501），
  而当日指数只有它能给；报价端点（另一个 host）当时仍 200 → 已修，实测链尾生效；
  ⑥ **`daily` 退出码**（`cli.EXIT_DATA_FAILURE = 2`，见「关键机制」那条）：数据侧故障
  现在会被操作系统看见（此前 09-17 那次空转的 `LastTaskResult` 仍是 0）；
  ⑦ **成交量完整性审计**（`scripts/audit_volume_integrity.py`，见「已知问题」4）：
  逐 (股票, 日期) 对撞独立源，且"谁是真值"按实测定（腾讯 `amount` 是估算式，不许当真值）；
  新增/改写测试：`test_volume_integrity.py`（8 项）、`test_index_snapshot.py`（11 项）、
  `test_index_source_chain.py`（+1 项）、`test_daily_recovery.py`（退出码 4 项 + 改 2 项）。
  ⑧ **批量报价把全市场科创板成交量放大 100 倍（当天发现并修复）**：见「已知问题」4 最后一条——
  当日 604 只 688 的 09-18 volume 被 `tencent_quote.parse_quotes` 无条件 ×100 写坏，
  审计脚本先报出 4 只、手工抽查证实是系统性；已修行、修数据（备份 + 三条验收）、重建三件套。
  ⑨ **算法问题⑤（20 日 embargo）已修**：训练集与校准期之间空出 horizon 个交易日，
  旧训练集里 103,505 行（3.37%）的标签曾伸进校准期；生产已重训（`embargo_days: 20`）；
  ⑩ **算法问题⑥（死计算 thresholds）已修**：改为按需（默认不算），省下每轮 6 组校准期预测；
  ⑪ **缓存不再被自己刷失效**：`cache.append` 内容相同时不写盘 → 面板/特征缓存真的会命中
  （实测 `panels/meta.json` mtime 不变、`build_panels` 25 s → 3.0 s）。
  实测：**全量测试 221 项全绿、`ruff check .` 全绿**；六处新判据的熔断丝都自证过
  （audit 旧判据红 3/8、退出码旧行为红 6/6、链尾 append 删掉红 2/2、批量报价 ×100 红 3/5、
  embargo+thresholds 红 2/2、缓存刷新红 2/2）。
  详见「已知问题」3、4。

- **09-17/09-18 连续两天的 16:05 空转与当日修复（重要事故线，别再踩）**：
  - **09-17 16:05**（`logs/daily_scheduled.utf8.log` 实录）：三个指数源**全部无返回**，
    数据停在 09-16；`LastTaskResult=0`（旧代码只打印 ‼️ 就 return）→ **当天什么都没更新，
    而且没人看得见**。日志只跑了 34 秒。
  - **09-18 16:05**（同一天，修复退出码之前就已在跑）：只有 akshare 应答且停在 09-17
    （新浪指数天然滞后一天）、腾讯 K 线 **HTTP 501**、通达信不可用 → 判定"落后 1 个交易日"，
    正确地拒绝推进，退出码 **2**（新约定第一次在真实任务里生效）。
  - **根因**：能给出**当日**指数 bar 的只有腾讯 **K 线** host，而它会 WAF（实测 09-18
    15:41 还是 200、16:07 变 501）；报价 host 同时刻正常。修法 = 链尾 `index_snapshot`。
  - **生产恢复（16:13 手动跑同参数 daily）**：`指数截止=2026-09-18 更新=5352 已最新=0 失败=0
    （指数源：index_snapshot、akshare）`，耗时 `phase1 33.3s / phase2 11.5s / phase3 25.9s /
    total 70.7s`，退出码 0；`update_stats.json` 的 `panel_coverage = {last_count: 5352,
    normal_count: 5352, ratio: 1.0, collapsed: false}`。
  - **自述 vs 实测对撞（本轮必做的一条）**：自述 `updated=5352 / stocks_behind=8 /
    completeness=0.9985`；用 parquet 元数据统计法实测截止日分布 =
    `{09-18: 5352, 09-14: 4, 09-11: 3, 09-09: 1}` → **落后 8 只，逐项吻合**；
    面板最后一日 `09-18` 非空 **5352/5360**；`features.meta.json` 指纹变为 `f6cf503c…`。
  - **账户口径独立复算（防"数据补回来后净值变好看"这类假象）**：09-18 总资产 110,002
    （+10.00%，29 次决策）。**分步验证**：月度段起点 09-01 的 50 只持仓按
    `equity_curve` 同口径（权重随价格漂移）复算，自段起 **+6.940%**，账户自段起 **+6.651%**
    → 差 0.29 pp 正是 09-01 换仓的一次性成本（≈ 卖出 0.175% + 买入 0.125% = 0.30%），
    **且该差额在 09-16/09-17/09-18 三天恒定不变**（成本只在段首扣一次）；
    09-17→09-18 单日复算 **+3.661%** == 账户单日 **+3.661%**（三位小数逐位相同），
    同期沪深300 仅 +1.06% → 当日超额来自持仓（多为 688/科创）。

- **09-16 当日的数据状态：看"补齐后"，别被 16:29 那次的自述骗了**。
  - **16:29 那次（缺陷态，事故快照）**：`update_stats.json` 自述 `last_run=2026-09-16 16:29:11 /
    source=akshare / index_date=2026-09-16 / days_behind=0 / healthy=true / updated=4989 /
    up_to_date=0 / failed=0 / no_data=371`，耗时 `phase1_sec=1382.8 / phase2_sec=20.2 /
    phase3_sec=44.3 / total_sec=1447.3`（报告 27.8s + 决策 16.5s，16:05 起跑、16:29 结束）。
    **但逐文件实测 parquet 日期分布是 5140 只停在 09-15、只有 209 只真到 09-16** ——
    自述指标只反映指数层，当天决策就建立在混合日期面板上。根因与修复见「关键机制」两条
    （"非空即 updated" / "healthy 只看指数"）。
  - **18:15 补齐后（当前）**：逐文件实测 **5350 只到 09-16**，其余 10 只是真停牌
    （4 只停 09-14、3 只停 09-09、3 只停 09-11）；`manifest.json` 的 `end` 与 parquet 实际 max
    **0 处不一致**；新的 `update_stats.json` 自述 `last_run=18:15:27 / index_status=unchanged /
    stocks_total=5360 / stocks_behind=10 / stocks_behind_unexpected=0 / completeness=0.9981 /
    stocks_ok=true / updated=5141 / no_data=10 / failed=0 / phase1_sec=55.1 / total_sec=105.6`。
    ⚠️ `phase1_sec=55.1` 是**一次性补齐 5141 只**的耗时，**不是稳态增量耗时**（稳态见 README
    那张表的"修复后 36.8s / ≈145 只每秒"，那是 scratch 同口径实测）。
  - **18:33 特征表重建 + 决策重生（当前终态）**：`features.meta.json` 变为
    `{as_of: 2026-09-16, n_rows: 3497773, fingerprint: 21718a55…, version: 1}`；
    `features.parquet` 最后一日（09-16）行数 **5236**（塌缩时只有 204，09-14/09-15 分别是 5234/5235）；
    `decision.json`（18:33:48）50 只 picks 的号段已恢复分散
    （300668 / 301396 / 688525 / 688450 / 688783 / 688347 / 920211 / …，不再清一色 `000xxx`）。
    这三条都是**读文件实测**；根因见「关键机制」里特征缓存指纹那条。
- **计划任务 AshareQuantDaily 正常**：实测 `LastRunTime=2026-09-16 16:05:05`、
  `LastTaskResult=0`、`NextRunTime=2026-09-17 16:05:05`、`NumberOfMissedRuns=0`。
  日志改 UTF-8 的新命令**已于 2026-09-16 原地更新进任务**（`Set-ScheduledTask`，无空窗），
  因此 **09-17 16:05 那一次就是首次以 UTF-8 写日志**；跑完后按「已知问题」2 的判据 B 验一次。
- **测试**：**224 项全绿**（`python -m pytest tests/` 实测 `224 passed`）、`python -m ruff check .` 全绿。
  演进：126 →（09-16）155 →（09-18 第一轮）179 →（09-18 第二轮）203 →（09-18 第三轮）221
  →（09-23 部署体检）**224**（新增 3 项：CLI 顶层兜底/中断/退出码透传）；
  ⚠️ 本节此前写「212 项」，而同一段列的演进又止于 203 —— **自相矛盾且都过期**。
  项数以 `python -m pytest tests/ --collect-only -q` 的**实测**为准，别照抄数字。
  新增集中在 daily 正确性/批量快路径、特征与面板缓存判据、抓取源链、成交量审计、
  指数报价兜底源与仪表盘冒烟。
  偶发的 statsmodels 扩展加载失败见「已知问题」1。
- **09-10/09-11 数据缺失事故**（历史，已修复）：全市场 5360 只当时补到 2026-09-11
  （仅 8 只 002870/301390/600825/600929/603159/605577/688291/688432 真停牌，两源都无当日
  bar，走冷却明日重试），指数 749 行、面板 `last_date=2026-09-11`。
  根因与修复见上文「指数取不到 ≠ 没有新交易日」与「跨源追加必须审计」。
- **补数据改变了历史结论**：缺失的两天正好是急跌（全市场中位 -1.52% / -2.24%，
  09-11 仅 11% 个股上涨），账户总收益率从虚假的 **+3.25% 修正为 +0.13%**
  （103,254 → 102,012 → 100,126）。以后凡"数据停在旧日期"都要先怀疑口径，
  不要拿滞后的净值下结论。
- 主源已切到 akshare（`config.yaml`，指数与个股共用链）；mootdx 恢复后改回并
  跑 `scripts/audit_qfq_anchor.py --fix`。
- 已修复并验证：xgboost.dll 缺失导致 08-27 决策崩溃（重装 + `load_models`
  可操作报错）、时间炸弹测试（`test_calendar.py` 按实时钟构造日期）、
  账户 `_prepend_start_point` 空序列 concat 弃用告警。
- 历史修复仍有效：mootdx qfq 未来除权缩放 bug、北交所 920 空 xdxr 丢列、
  实时估值 +0.59% 假收益（根因=盘中价当成本口径错配，已可精确复现）。

## 已知问题

### 1. 偶发：`test_volatility_clustering_on_garch_like_series` 首次加载 statsmodels 编译扩展失败

- **现象**（2026-09-16 观测 1 次 / 全量 4 次）：全量测试偶发只红这一个，报错位置在
  `<site-packages>\statsmodels\tsa\stattools.py:39`
  （`from statsmodels.tsa._innovations import innovations_algo, innovations_filter`）；
  单独跑 `tests/test_research_stats.py` 必绿。
- **已定性（实测）**：该行是 statsmodels 编译扩展
  `statsmodels\tsa\_innovations.cp313-win_amd64.pyd` 的**首次加载点**，且**只在测试体内**触发：
  `volatility_clustering()` → `acorr_ljungbox()`（函数体内的
  `from statsmodels.tsa.stattools import acf`，见 `statsmodels/stats/diagnostic.py:412`，
  注释 "Avoid cyclic import"）→ `stattools.py:39`。
  所以偶发红一定是"该 .pyd 在那一刻首次加载失败"，**不是断言、数据或依赖解析问题**：
  该 .pyd 的 PE 依赖只有 `python313.dll / KERNEL32 / VCRUNTIME140 / api-ms-win-crt-*`，
  这些都早已在进程里。
- **未能证实（不要写成结论）**：具体触发条件。排除实验全部 0 失败：

  | 实验 | 结果 |
  |---|---|
  | 全量 `pytest tests/ -q` × 19（含 3 次在并发加载压力下） | 19/19 全绿 |
  | 独立进程 `import statsmodels.tsa.stattools` × 216（120 串行 + 96 并发） | 216/216 成功 |
  | `test_akshare_fetcher.py + test_research_stats.py` × 40（验 `sys.modules["akshare"]` 假设） | 40/40 绿 |
  | 静态排查 `sys.path` / `os.environ` / `chdir` / `add_dll_directory` 改动 | 0 处 |
  | `__import__` 探针跑全量：stattools 的导入次数与线程 | 恰好 1 次、主线程、无残留线程 |

  216 次全成功 ⇒ 单次加载失败率 < 1.4%（95% 置信上界），与"每 4 次全量红 1 次"不相容 ⇒
  **不是"这个扩展本身不稳定"，更像那一次运行时遇到了瞬时的外部干扰**（杀软扫描/文件占用一类）。
  归因到具体干扰源：**未证实**。
- **让失败变响亮的改动（已做，未放宽任何断言）**：`tests/test_research_stats.py` 在**收集期**
  显式预加载该扩展；失败时抛的 `ImportError` 带 `winerror`、.pyd 路径与是否存在，
  且整轮测试直接变红（不再是测试体深处一个指向 site-packages 的 traceback）。
  该诊断消息路径已自证：用 `sys.meta_path` 故意拦住 `statsmodels.tsa.stattools`，
  得到的消息是 `statsmodels 编译扩展首次加载失败：ImportError('DLL load failed while importing
  _innovations: 参数错误。')；winerror=998；已安装的 _innovations 扩展=['tsa\_innovations.cp313-win_amd64.pyd']；…`
- **下次再红，照着这 4 条排查（逐条可复制；括号里是"正常的样子"）**：

  ```bash
  # 判据 1（5 秒）扩展本身能不能加载 —— 正常输出 LOAD OK
  python -c "import statsmodels.tsa.stattools; print('LOAD OK')"
  # 判据 2（30 秒）单独跑该文件 —— 正常 2 passed；绿 = 那次是瞬时故障（本条目就是记录这个）
  python -m pytest tests/test_research_stats.py -q
  # 判据 3 扩展在不在、版本对不对 —— 正常 ['tsa\\_innovations.cp313-win_amd64.pyd']（statsmodels 0.14.6）
  python -c "import statsmodels, pathlib; p=pathlib.Path(statsmodels.__file__).parent; print(sorted(str(x.relative_to(p)) for x in p.rglob('_innovations*.pyd')))"
  # 判据 4 全量复跑一次（约 1 分钟）—— 正常 全绿；只红这一个即命中本条目
  python -m pytest tests/ -q
  ```

  判据 1/2 报的 `ImportError` 里就带 Windows 的错误文本（`DLL load failed while importing
  _innovations: ...`）；**把整段原文抄进本条目**（2026-09-16 那次的红就是被 `| tail -3` 截掉、
  原文已不可得，所以只能定性到"失败点"）。若判据 1 持续红（不是偶发），按"安装损坏"处理：
  `python -m pip install --force-reinstall --no-deps statsmodels`。

### 2. 历史日志 `logs/daily_scheduled.log` 编码已损坏（新日志已改 UTF-8）

- 该文件有两种叠加的毛病：① `>>` 写成 **UTF-16LE**（grep/tail 花屏）；② **在落盘前就已乱码** ——
  PS 5.1 用 GBK 解码 python 的 UTF-8 stdout，无法映射的字节被换成 `?`，**不可逆**
  （实测 754 行里 456 行含 `?`、共 835 个）。**所以不要试图"转码修复"它，信息已经丢了。**
  只读查看旧文件（人眼扫可以，**不能当数据源**）：
  `python -c "import pathlib;print(pathlib.Path('logs/daily_scheduled.log').read_bytes().decode('utf-16-le').encode('gbk','replace').decode('utf-8','replace'))"`
- 新日志由 `scripts/run_daily_logged.ps1` 写 **UTF-8（无 BOM）** 到 `logs\daily_scheduled.utf8.log`
  （两个编码问题一起修：先 `[Console]::OutputEncoding` 设成 UTF-8，再用 .NET 显式按 UTF-8 追加，
  不用 `>>`；并加 `===== run start/end <时间> exit=<码> =====` 前后缀，便于分辨是哪一次运行）。
  历史文件保持不动。任务命令已于 2026-09-16 原地更新，**09-17 16:05 起的新日志就是 UTF-8**。
- **怎么验（逐条可复制；括号里是"正常的样子"）**：

  ```powershell
  # 判据 A（不用等真实运行，立刻可验）—— 正常：SELFTEST PASS: ... bytes=71 NUL=0 BOM=none UTF8-roundtrip=ok
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_daily_logged.ps1 -SelfTest
  # 判据 B（真实运行之后）—— 正常：NUL 0 BOM False lines <非0>，末尾能看到 run end ... exit=0
  python -c "import pathlib; b=pathlib.Path('logs/daily_scheduled.utf8.log').read_bytes(); print('NUL', b.count(b'\x00'), 'BOM', b[:3]==b'\xef\xbb\xbf', 'lines', b.count(b'\n')); print(b.decode('utf-8')[-400:])"
  # 判据 C 任务实际用的是不是新命令 —— 正常：Arguments 里含 run_daily_logged.ps1 与 daily_scheduled.utf8.log
  powershell -NoProfile -Command "(Get-ScheduledTask -TaskName AshareQuantDaily).Actions.Arguments"
  ```

  失败的样子与处置：判据 A 打 `SELFTEST FAIL: ... NUL=<非0> BOM=<True>`、或判据 B 报
  `UnicodeDecodeError` / `NUL` 非 0 ⇒ 说明又走回 `>>`（判据 C 可确认命令被改回旧写法）；
  若任务确实指向新脚本而 A 仍 FAIL，看日志里有没有 `WARN: cannot set [Console]::OutputEncoding`
  —— 那表示运行环境没有 console，需要改成"不做转码、直接按字节落盘"的方案（`Start-Process
  -RedirectStandardOutput` 是原始字节拷贝，可作后备）。

### 3. 面板横截面塌缩（更新窗口内的中间态）—— **已修复 2026-09-18**

- **根因（本轮定论，此前只记为"推测"）**：不是 `build_panels` 算错，而是**放行判据
  用的是"自述"而不是"实测"**：
  - `daily.py` 的 `new_data` 由 `update_daily` **自述**的 `completeness ≥ 0.5` 决定；
  - 真正被决策与仪表盘消费的**面板横截面**从来没被测量过。
  于是"自述 healthy、面板塌缩"（09-16 实测：指数已到 09-16、96% 个股停在 09-15、
  面板最后一行只有 209/5360）能一路走到**出决策**；而塌缩面板还会被**落盘缓存**，
  仪表盘直接读 `panels/close.parquet` → 等权全市场基准算出 -57% 假暴跌。
  这正是项目自己的工程约定「**自述指标必须与实测对撞，不能自证**」没有落到这条链上。
- **修复（写入侧拒绝 + 消费侧硬停 + 第二道消费守卫）**：
  1. `pipeline.panel_coverage(panels)`：**实测**最后一日有效股票数 vs 近 10 日中位数。
     判定条件（缺一不可，防误伤）：常态 ≥20 只、绝对缺口 ≥10 只、最后一日 < 常态一半。
     相对判定 → 新建库/自选股/少数真停牌（个位数）都不会被误判。
  2. `_save_panel_cache` **拒绝落盘塌缩面板**（宁可不缓存、下次重建 ~25s，也不把坏面板
     留给仪表盘消费一整天）；`_load_panel_cache` 再拒绝一次（防更早版本写下的坏缓存）。
  3. `cmd_daily` 在 `build_panels` 之后**硬停**：打印 ‼️ 并写
     `update_stats.json` 的 `panel_coverage / panel_collapsed / error=panel_cross_section_collapsed`，
     **不跑报告与决策**；健康路径也把实测覆盖率一并写进 stats（自述与实测并列，便于对账）。
  4. 消费侧第二道：`screening.covered_dates` 先剔除塌缩日期再算等权全市场月频基准
     （与仪表盘 `equal_weight_bench` 同口径）——月频观测被污染一个月就足以改变"是否跑赢基准"。
- **验证（都是实测，不是"看起来对"）**：
  - 单测：`tests/test_pipeline.py` 的 coverage 判定（正常日/小样本不误伤、塌缩必判）、
    塌缩不落盘、遗留坏缓存被拒；`tests/test_daily_recovery.py` 的 CLI 门禁两条
    （自述 95% 健康 vs 面板 2/20 塌缩 → 硬停且不调用报告/决策；健康面板照常跑完三段）；
    `tests/test_screening.py::test_covered_dates_drops_collapsed_days`。
  - **旧行为验证**：临时把门禁改成 `if False:` 后，CLI 门禁测试实测**报红** ✓（真熔断丝）。
  - 真实数据：生产面板 5352/5352、`ratio=1.0`、`collapsed=false`（不误报）；
    缓存命中与重建两条路径都带 `_coverage`。
  - 全量测试 179 项全绿、`ruff check .` 全绿。
- **仍然成立的边界（别过度解读）**：
  - 守卫拦的是"**最后一日**相对常态塌缩 ≥50% 且缺口 ≥10 只"；更轻度的部分缺失
    （例如只到 80% 覆盖）仍会照常出决策 —— 与既有 `completeness ≥ 0.5` 口径一致。
  - 面板**当日盘中**的中间态依然存在（那是数据源发布时序，不是缓存问题）；
    本修复保证它不会被**落盘并消费一整天**。
- **排查/规避（仍然有用）**：判断面板是否塌缩，看最后一行非空只数：
  `python -c "import pandas as pd; s=pd.read_parquet('data/tencent/panels/close.parquet').iloc[-1]; print('非空', s.notna().sum(), '/', len(s))"`
  （正常 ≈5350/5360；两三百就是塌缩。）

### 4. 本地日线 `volume` 量纲有两处台阶（**已修复 2026-09-18**）

**全市场一处、科创板额外一处**，两处都会让 `vol_ratio` 在跨台阶的约 20 个交易日内失真：

| 台阶 | 范围 | 表现 | 状态 |
|---|---|---|---|
| **09-10 换源 mootdx → akshare** | **全市场 4710 只 / 3,388,755 行** | 09-09 及以前是"手"（比"股"小 100 倍），09-10 起是"股" | 已修（×100） |
| 09-16 起腾讯兜底写入 | 科创板 604 只 / 1,226 行 | 已是"股"又被 ×100 → 虚高 100 倍 | 已修（÷100） |

> ⚠️ **第一轮排查漏了全市场那处，原因值得记住**：当时的"全市场体检"只算了**最后一行**
> 的 `volume/(amount/close)`，250 只非 688 全部 ≈1.00 → 结论"非 688 无需处理"。
> 但最后一行属于 akshare 时代（本来就是股），**历史那 747 行根本没人看**。
> **量纲体检必须按时段（或按行）分桶，不能只看最后一行** —— 正确做法见下面的检测脚本。
> 是修完科创板后发现"主板 `vol_ratio` ≈3.5、科创板 ≈1.0"这种系统性差异才回头查出来的。

- **实测证据（2026-09-18）**：逐文件算 `volume / (amount/close)`（该比值恒等于
  "volume 相对股数的倍数"）—— 科创板 603/603 只中位数 99.72，其他板块抽样
  250/250 只 ≈ 1.00（**但那是最后一行**，见上面的警告）。分时段看（688525/688111 一致）：

  | 时段 | 比值 | volume 实际单位 | 说明 |
  |---|---|---|---|
  | ≤ 09-09（mootdx 建的） | 0.01 | **手**（比股小 100 倍） | 与全市场"股"口径不一致 |
  | 09-10 ~ 09-15（akshare） | 1.00 | **股** ✓ | 正确 |
  | 09-16 ~ 09-17 | ~100 | **股×100** | 09-16 事故后的补齐/回退路径写入 |

- **根因**：`tencent_fetcher` 旧版 `_parse_kline` **无条件** `volume * 100 # 手 → 股`，
  而腾讯对科创板给的就是"股"（见「关键机制」那条）。该乘法**已修**（改为按
  `venues.volume_in_shares` 判定），但**已写进 parquet 的历史值不会自己变回来**。
  09-16/17 的 688 行是**混源行**：`volume` 来自腾讯（被放大 100 倍），
  `amount` 来自 akshare（正确）—— 所以 `amount/close` 仍可用作股数真值。
- **影响面（务必按此评估，不要只看"成交量柱画错了"）**：`features.py:83` 的
  `vol_ratio = volume.rolling(5).mean() / volume.rolling(20).mean()` 是**比值**，
  尺度恒定时会抵消，**只在跨台阶时出错**；而 20 日窗口意味着每次台阶后错误会
  持续约 20 个交易日（09-16 的台阶要到 ~10-15 才滑出窗口）。
  当前状态（09-18）：09-17 决策的 50 只持仓里 **20 只是科创板**，其 `vol_ratio`
  正在被这条污染影响（09-17 抽查 688525=3.98 vs 600000=3.33，数值巧合地接近，
  别用"看起来正常"当作没坏的证据）。
- **检测（可复制）**：

  ```bash
  # 量纲体检（必须**按时段分桶**，只看最后一行会漏掉历史台阶 —— 本轮就是这么漏的）
  python -c "
  import pathlib, pandas as pd, numpy as np
  root = pathlib.Path('data/tencent')
  bad_hand = bad_infl = n = 0
  for p in list(root.glob('*.parquet')):
      if not (p.stem.isdigit() and len(p.stem) == 6): continue
      d = pd.read_parquet(p, columns=['volume','amount','close'])
      r = (d['volume']/(d['amount']/d['close'].replace(0, np.nan))).replace(0, np.nan)
      n += 1; bad_hand += int((r < 0.1).sum()); bad_infl += int((r > 10).sum())
  print('股票', n, '| 仍为手(需×100)的行', bad_hand, '| 仍虚高(需÷100)的行', bad_infl)
  # 干净应为：0 和 0（修复后实测 0/0）"
  # 或直接跑修复脚本的 dry-run（判据同源，且会自动问独立源）：
  python scripts/fix_volume_unit.py          # 期望：需要修复 0 只
  ```
- **修复记录（2026-09-18，全市场已完成）**：工具 `scripts/fix_volume_unit.py`
  （`--apply` 写入 / `--verify` 全量验收 / `--limit N` 抽查 / 默认 dry-run）。
  两层判据：`ratio = volume/(amount/close)` 偏离 1 → 只改 volume（`>10` 除 100、
  `<0.1` 乘 100）；`ratio ≈ 1` 的行属于"两列可能一起错"（腾讯的
  `amount = volume*close` 是估算式，会跟着 volume 同倍缩放），**必须问独立源定方向**：
  先 akshare，失败再 tencent，两者都取不到就**跳过绝不猜**。
  实际改动：**4710 只**（含科创板 604 只）—— 3,388,755 行 volume×100（手→股）、
  34 行 volume÷100、0 行需要两列同改；另有 1,937 行经真值对撞确认本来就正常（不动）。
  定向询问独立源：akshare 877 只 + tencent 兜底 689009（CDR，akshare 返回非法 JSON）。
  备份 `data/tencent_volume_bak_20260918/`（首轮 604 只 688 的**修复前**原样也在此目录，
  不被后续覆盖）。脚本**幂等**（再跑报"需要修复 0 只"）。
- **改完必须重建三件套**（缺一步就等于没改）：面板 → 特征 → 模型/决策。
  本轮实测的坑：只跑 `daily --force --retrain` **不够** —— 面板缓存当时只校验
  manifest 与面板最后日期，改数值不改日期时被判有效，`--retrain` 实际用的是
  **被污染的旧面板**（`features.meta.json` 指纹一字未变即为铁证）。
  现已给面板缓存加上**源数据签名**（见「关键机制·面板缓存」），
  改完数值直接 `daily --force --retrain` 即可自动失效重建。
  判据：`features.meta.json` 的 `fingerprint` 必须变化；
  抽查 `panels/volume.parquet` 的历史值与日线一致。
- **这次修复踩到的三个坑（都是"判据本身不可靠"，务必记住）**：
  1. **判据必须幂等**：最初用 `amount == volume*close` 判"两列一起放大"，
     两列同除 100 后等式**依然成立** → 再跑一次又除一次（数据会毁成 1/10000）。
  2. **邻域中位只能用来发现问题，不能用来定方向**：用 ±10 日邻域中位判方向时，
     紧挨着被放大行的**干净行**因邻域被污染反而显得"偏小"，实测误判 78 行
     （会把好数据乘 100）。方向必须来自独立源。
  3. **宽 `except` 会吞掉真问题**：脚本放在 `scripts/` 下运行时 `sys.path[0]` 是
     `scripts/`，`import ashare_quant` 抛 ImportError 却被 `except Exception: return None`
     吞成"真值取不到"，表现为"查了 0 只、130 行全部跳过"而看不出原因。
     现在脚本显式补项目根，且真值获取失败会打印首条原因。
  → 同类"批量改写历史数据"的任务，请照这个骨架做：**dry-run 默认 / 备份 / 方向取自
  独立源 / 幂等自检 / 全量对撞验收 / 重建三件套并核对指纹**。
- **遗留（量纲已清，但有个别孤立错行）→ 现已工具化，见下条**：全市场对撞 akshare 后 **5123/5360 只、3,703,187 个交易日零不一致**；
  103 只存在**孤立单日**差异（多为北交所 1~2 天，比值 1.15~7.9×，**不是 100× 量纲错**），
  134 只因新浪限流未取到真值。抽样仲裁：**603559 于 2025-01-03 本地 46,400 vs akshare/腾讯一致为 5,900
  —— 该行确为错值**（mootdx 时代遗留）；北交所样本腾讯无覆盖、无法三方仲裁，故未动。
  这类"个股单日错值"与量纲无关，修法是**按 (股票, 日期) 用独立源重取该行**
  （工具 `scripts/audit_volume_integrity.py`，纪律见下条；改完同样要走"重建三件套 + 核对指纹"）。
  **收官复测（2026-09-18 17:47 全市场重扫，报告 `docs/research/volume_integrity_report.json`）**：
  `confirmed 0`（没有任何一行能被两个独立源共同确证为错）、`amount_only 1`（920128@2025-12-25，
  volume 差 11% 在容差内、只有 amount 超差）、`single_source 84`、`sources_disagree 0`；
  其中 84 行 single_source **以 92xxxx 北交所为主**（920018×11、920163×10、920118×9…），
  腾讯对这些日期无覆盖 → 按纪律**只报告不改**；另有 **749 只 no_truth**（新浪 SSLEOFError 限流，
  连跑两轮全市场扫描后尤其明显）。
  **603559@2025-01-03 仍未修**：本轮先用 akshare+腾讯三方对撞确证过它是错值
  （本地 46,400 vs 两源一致 5,900，且本地行内 `volume×close/amount = 7.864`、改成 5,900 后恰好 1.000），
  但重扫时两个源都被限流取不到真值 → 工具按"绝不猜"只报告。下次两源同时可用时直接
  `python scripts/audit_volume_integrity.py --apply` 即可把它连同其他 confirmed 行一起修掉
  （改完照旧"重建三件套 + 核对指纹"）。
  日后**再换数据源**时必须重跑量纲体检 —— `python scripts/fix_volume_unit.py`（dry-run）
  应始终报"需要修复 0 只"；若报非 0，说明新源的成交量单位与"股"不一致，按脚本判据处理（方向一律问独立源）。
- **成交量完整性审计（2026-09-18 新增 `scripts/audit_volume_integrity.py`）—— 判据的支点是"谁是真值"，而它得实测**：
  逐 (股票, 日期) 对撞独立源，容差 0.05 dex（≈±12%），`volume` 与 `amount` 两列**分开**判。实测结论（600000/000001 各 1628 日）：

  | 源 | volume | amount |
  |---|---|---|
  | akshare（新浪） | 与本地逐日一致 → **真值来源** | **真值来源** |
  | tencent | 与 akshare **1628/1628 日一致** → 可作**独立确证** | ⚠️ **估算式** `volume × 调整后收盘`；实测 **1101/1094 日**与真值不符（中位比 0.80/0.86）→ **绝不可当真值** |

  四类判定，方向绝不猜：`confirmed`（本地 volume 与 akshare 不符 **且腾讯 volume 与 akshare 一致** → 用 akshare 值修，
  amount 仅在同样超差时一并改）/ `amount_only`（只有成交额不符 → **只报告**）/ `single_source`（腾讯无覆盖，如北交所 → 只报告）/
  `sources_disagree`（两源互相矛盾 → 只报告）。判据本身是 `adjudicate()` 纯函数，`tests/test_volume_integrity.py` 钉住 8 条。
  **这个"用腾讯 amount 当真值"的写法曾经真的存在**（本脚本第一版就是），把它还原后新测试实测**红 3/8**（熔断丝自证）。
  另：**旧判据连 603559@2025-01-03 都修不了**（它按"本地 vs 腾讯"判，得到"两源矛盾"而放弃）——错得既危险又无用。

- **同一量纲坑的第二次复发：批量报价路径（2026-09-18 当天发现并修复）—— 本条目最该记住的部分**：
  `fetchers/tencent_quote.py::parse_quotes` 里 `volume = _num(detail[1]) * 100  # 手 → 股`
  **无条件 ×100**，而腾讯报价对科创板给的就是"股"。当天它把**全市场 604 只科创板的当日
  volume 全部放大 100 倍**写进 parquet（688525 写成 1,889,345,500，真值 18,893,455；
  逐文件按 `volume/(amount/close)` 分桶实测：受影响 604 只、每只正好 1 行、日期全部为 09-18、其他板块 0 只）。
  **发现路径值得照抄**：16:47 的全市场成交量审计只报出 `confirmed: 4`（688538/688828/688836/688981，
  比值恰好 100×）；顺着"为什么只有 4 只"手工抽查 688525/688111 才发现是**系统性**的
  —— 其余科创板因为 akshare 真值抓取被新浪限流（572 只 no_truth）而没被比出来。
  → **审计报出的数量是"能取到真值的那些"的下界，不是全貌；判据在少数样本上变红时，先问"这是样本还是总体"。**
  修法：`volume = raw if volume_in_shares(code) else raw * 100`；新增 `tests/test_tencent_quote.py`
  （5 项，用 2026-09-18 实测抓下的真实响应文本；还原成"无条件 ×100"实测**红 3/5**）。
  修数据：604 行 ÷100（复用 `fix_volume_unit.py` 的 `classify/normalize`；行内 ratio≈100 即本地确证，
  不需要独立源），备份 `data/tencent_volume_bak_20260918_star/`（604 个文件）。
  验收三条：全市场体检 **0/0**、akshare 抽检 **8/8 逐位一致**（688525=18,893,455 == ak）、
  审计复测这 6 只科创板 **0 异常**。改完重建三件套：`daily --force --retrain`，
  指纹 `f6cf503c…` → `12c235a3…`，`panels/volume` 688525@09-18 变为 18,893,455；
  ⚠️ **09-18 的决策因此变了**（修正前更偏 688；修正后 `688783/688797/300433/688006/…`），
  账户净值不受影响（价格列没动，仍 +10.00%）。
  → **这个坑在 `tencent_fetcher`、`realtime` 之后第三次出现在新模块（批量报价），而它此前没有任何测试。**
  "成交量单位"不是可以就地判断的事：唯一事实来源是 `venues.volume_in_shares`，
  新增任何解析成交量的模块都必须调用它，并补一条 688 的测试。


## 账户口径与算法审查（2026-09-16 晚，独立审查）

- **账户曾在模拟"每日全额换仓且零成本"，账面收益被系统性高估。**
  实测：27 次 live 决策间隔 {1天:18, 2天:2, 3天:6}（≈每日），单边换手率均值 **59.4%**；
  `portfolio.equity_curve` 按相邻决策日分段，而 `config.rebalance` **声明了却全代码无人读取**（死配置）。
  同时账户**一分钱成本都没扣**（`commission`/`slippage` 在 portfolio.py 命中 0）。
  **注意这不是"少扣成本"这么简单**：按 backtest/engine.py 自己的参数（佣金 0.025%+印花税 0.05%+
  滑点 0.1%），日频换仓的成本会吃掉约 76% 的账面收益 —— 也就是说**每日 churn 本身在毁灭价值**，
  零成本记账只是把这个事实掩盖了。四宫格实测（真实历史 + 真实面板）：

  | 换仓 | 口径 | 总收益 | 年化 | 夏普 | 最大回撤 |
  |---|---|---|---|---|---|
  | 日频（旧行为） | 毛 | +6.31% | +73.50% | 1.60 | -9.09% |
  | 日频（旧行为） | 净 | +1.60% | +15.31% | 0.33 | -11.22% |
  | 月频（与回测一致） | 毛 | +7.70% | +94.94% | 1.95 | -9.68% |
  | **月频（与回测一致）** | **净** | **+7.27%** | **+88.13%** | **1.81** | -9.68% |

  **旧口径是两个错误互相抵消**：不扣成本（虚高）+ 每日换仓（实亏）。改成月频+扣成本后净收益
  反而更高。⚠️ 但月频结论目前只建立在 **2 次调仓**（08-07 与 09-01）上，**统计上还不算数**，
  随历史累积才有意义 —— 不要把它当成"月频更好"的证据，只能说"日频 churn 明显有害"。

- **已改**：`equity_curve(..., costs=, rebalance=)`；`Config.commission/stamp/slippage`；
  `config.yaml` 的 `rebalance` 现在真正生效（M=每月首个决策日换仓）。
  **默认值保持旧行为**（costs=None/rebalance=None 时逐字节等价），只有写盘与仪表盘显式传参。
- **凡是算账户的地方都必须传同一套口径**，否则同一页面会出现两个"总资产"：
  漏改 `account_basis` 时实测实时页 106,313 vs 账户页 107,274。已统一为
  `costs_from_config(cfg)` + `cfg.rebalance`，共 5 处（写盘 1 + 仪表盘 recompute 2 + account_basis 3 个调用点）。
- **仍未修的算法问题（审查发现，未动手）**：
  ① **ML 基准 `backtest/simple.py` 的成本口径 —— 部分已修（2026-09-18 复核更正）**：
     本条目原写"也是零成本"，那在写下时是对的，**现已不准确**，据实更正如下：
     `simple_topn_returns` 已加 `costs=` 参数（2026-09-16），`ml/benchmark.py` 的 **8 处调用
     全部传 `costs=BENCH_COSTS`**（净口径），生成器抬头也已写清成本参数；
     **但 `ml/evaluate.py:41` 与 `screening.py:63,90` 仍不传成本 = 毛收益**，
     这两条路径的数字（模型筛选、单模型评估）仍需按毛收益读。
     ⚠️ **更要注意的是已提交的历史报告**：`docs/research/algorithm-benchmark{,-all}.md`
     生成于 08-10/08-11（**早于成本修复**），数字是毛收益，**而抬头写着"含交易成本"**——
     已在两份报告顶部加「口径更正」横幅（**不改写历史数字**）。
     `docs/research/modern-models-2026.md` 里"年化 78.5% / 夏普 2.81"等招牌数同源，同按毛收益读。
     月频换手下成本影响小于日频，但**毛收益会系统性偏向高换手模型**；
  ② **target 是原始 20 日收益，未做横截面去均值** —— 策略是横截面 Top-N、永远满仓，
     原始收益里绝大部分是市场共同波动（beta），模型会花容量去预测当天排名用不到的东西；
  ③ 特征里 `index_ret_20/index_vol_20/index_state` 是**市场级**量，同一天对所有股票相同，
     **无法影响当日排名**，对纯横截面模型是噪声；
  ④ **幸存者偏差**：股票池是当前在市清单，区间内退市的票不在历史里；
  ⑤ 训练集与校准段之间**没有 20 日 embargo**（目标跨度 20 日，calib_cut 前 20 日的标签伸进校准段）
     → **已修 2026-09-18**：见下条；
  ⑥ `train_and_save` 的 `thresholds`（每轮为 6 个模型各跑一遍最多 3 万行预测）**算完没有任何地方使用**
     → **已修 2026-09-18**：见下条。

### ⑤⑥ 的修复（2026-09-18，含实测数字）

- **⑤ embargo**：`train_and_save` 新增 `horizon`/`embargo_days`，训练集右端从 `calib_cut`
  再往前退 `horizon` 个交易日（这段标签两边都不用，即 purged/embargo 的标准做法），
  `models/*/meta.json` 记 `embargo_days: 20`、`horizon: 20`；`cli._save_decision` 用**同一个**
  `horizon` 常量同时驱动特征表与 embargo（两处不一致就是新 bug）。
  **量化（生产真数据实测）**：674 个交易日、校准期起点 2026-05-26；旧训练集 3,073,213 行里
  有 **103,505 行（3.37%，横跨 20 个交易日 2026-04-23→05-25）的标签窗口伸进了校准期**
  —— 校准期的"样本外残差"（也就是 thresholds 的度量口径）因此是乐观的；新训练集 2,969,708 行。
  生产已 `daily --force --retrain` 重训（`meta.json` 实测 `embargo_days: 20`；
  账户仍 +10.00%，价格列未动）。
  ⚠️ **边界**：这是方法论修正，**没有做 A/B 证明它提高了样本外表现**（20 天里少 3.37% 训练行，
  影响方向不能靠"应该更好"来断言）。
- **⑥ thresholds 改为按需**：默认 `compute_thresholds=False`，字段保留（schema 不变，`{}`），
  真要用传 `True`。依据：全代码 0 处读取（`decide()` 只用 `models`；`decision.json` /
  `account_history.jsonl` 只是把它写下来）。省下每次重训 6 × `min(30k, 校准行数)` 次预测。
- 验收：`tests/test_decision.py::test_train_leaves_embargo_between_train_and_calibration`、
  `::test_thresholds_are_opt_in`；把 embargo 去掉 + thresholds 门控还原成旧行为，实测**红 2/2**。


## ⚠️ 最重要的算法发现：Top-50 没有可证实的选股超额（2026-09-16 实测）

用 walk-forward 样本外做了 target 口径的 A/B（一次性脚本 `ab_target_mode.py`，
**未随仓库发布**；结论与证据如下表），结果**推翻了原有叙事**，请务必先读这一段再引用任何收益数字。

| 模型 | target | 日均截面超额 | t值(未校正) | 月均毛收益 | 年化(毛) |
|---|---|---|---|---|---|
| lgbm | raw | **−0.5632%** | −2.25 | +2.393% | +30.6% |
| lgbm | excess | −0.0892% | −0.34 | +2.651% | +34.5% |
| huber_lgb | raw | **−0.9223%** | −3.78 | +1.851% | +22.9% |
| huber_lgb | excess | **+0.1331%** | +0.48 | +2.930% | +39.2% |
| rank_lgb | raw / excess | −0.8856% | −5.16 | +1.458% | +18.0% |

- **"日均截面超额" = 每日 top-50 的 20 日前瞻收益 − 全市场同期均值。**
  **所有 raw 配置下它都是负的** —— 即选出的前 50 名**跑输市场平均**。
  而同期"月均毛收益"却是 +1.5%~+2.9%（年化 +18%~+39%）：
  **这些正收益几乎全部来自市场 beta（那段时间大盘在涨），不是选股能力。**
  → 引用"年化 78.5% / 夏普 2.81"这类数字时，必须同时说明它们**是毛收益、
  是绝对收益、且未经市场中性检验**。
- **两条必须同时读的警告**：
  ① **t 值被重叠样本高估了约 √20 倍**：20 日前瞻收益在相邻日高度重叠，
     242 个观测的**有效独立样本仅约 12 个**。校正后 lgbm 的 t≈−0.5，
     即"负超额"本身也**没有统计显著性**。正确的结论是"**未证实有超额**"，
     不是"证伪了超额"。
  ② 月频指标只有约 10 个观测，**基本没有统计意义**，仅供方向参考。
- **target 口径 A/B 的结论**：`excess`（同日横截面去均值）在**两个可比较的回归模型上
  都更好**（lgbm −0.56%→−0.09%、huber_lgb −0.92%→**+0.13%**）。
  `rank_lgb` 两种口径结果**逐位相同** —— 这不是 bug，是**机制的自证**：
  LambdaRank 只用当日截面内的相对顺序，而"减去当天常数"是保序变换，
  所以对 rank 类模型天然无影响。这同时说明改动方向是对的（只动共同分量、不动截面次序）。
- **但没有把默认改成 excess**：有效样本仅约 12 个，证据是"方向一致的建议"而非
  "已证实"。`config.target_mode` 已实现（`raw`/`excess`，指纹含此项，切换会自动重建
  特征表），**要切换改一行 + `daily --retrain`**。在"有没有超额"本身都未证实之前，
  换目标口径是二阶问题。
- **下一步该做什么**（留给后来人，不要只调参）：先补**市场中性/行业中性**检验与
  **分年 IC**，确认是否存在任何可用的截面信号；在没有之前，任何"年化 XX%"都应
  默认读作 beta。
