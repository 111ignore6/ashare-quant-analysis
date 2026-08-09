# 全市场数据补齐与研究重跑（M6）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把数据范围从沪深 300 扩展到全 A 股（约 5000+ 只，剔除 ST/退市），并在全市场上重跑研究、模型筛选、模拟盘与报告，让整套系统真正以"全市场"为对象。

**Architecture:** 复用全部现有模块，仅做三处小改动：`pipeline.download_universe` 增加下载进度输出（长任务可视化）；默认配置与计划任务脚本切换到全市场数据目录；README 补充全市场用法。核心计算全部复用 `fetch/research/select/simulate/report` 命令。

**Tech Stack:** 无新增依赖。

---

## Task 1: 下载进度输出

**Files:**
- Modify: `ashare_quant/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pipeline.py 追加
def test_download_universe_progress(capsys, tmp_path):
    store = ParquetStore(tmp_path)
    cfg = Config.from_dict({"years": 1, "retry": 1})

    def fake_fetcher(code, start, end, adjust):
        return _df(["2024-01-02"], [10])

    download_universe(["000001", "000002", "000003"], store, cfg,
                      fetcher=fake_fetcher, progress_every=1)
    captured = capsys.readouterr().out
    assert "progress 3/3" in captured
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: FAIL（无 progress 输出）

- [ ] **Step 3: 写实现**

在 `download_universe` 增加参数与进度输出：

```python
def download_universe(codes, store, cfg, fetcher=None, universe_name="csi300",
                      progress_every: int = 500) -> dict:
    ...
    done = 0
    with ThreadPoolExecutor(...) as ex:
        futures = {...}
        for fut in as_completed(futures):
            done += 1
            if done % progress_every == 0:
                print(f"progress {done}/{len(codes)}")
            ...
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add ashare_quant/pipeline.py tests/test_pipeline.py
git commit -m "feat: 全市场下载进度输出"
```

---

## Task 2: 全市场数据下载（断点续传）

- [ ] **Step 1: 运行全市场下载**

Run: `python -m ashare_quant.cli fetch --universe all --data-root data/all --years 3`
Expected: 逐步输出 `progress ...`；完成后 `ok≈5000、failed≈0`，`data/all` 下生成全部 parquet 与 manifest。

- [ ] **Step 2: 断点续传验证（重复运行应立即全部 skipped）**

Run: `python -m ashare_quant.cli fetch --universe all --data-root data/all --years 3`
Expected: `ok=0 skipped≈5000 failed=0`。

---

## Task 3: 全市场研究、模型筛选、模拟盘与报告重跑

- [ ] **Step 1: 全市场研究报告**

Run: `python -m ashare_quant.cli research --data-root data/all --out docs/research/data-research-all.md`
Expected: 生成报告，样本数量约等于全市场股票数；R1~R6 结论更新。

- [ ] **Step 2: 全市场模型筛选**

Run: `python -m ashare_quant.cli select --data-root data/all --out docs/research/model-selection-all.md`
Expected: 输出保留/淘汰决策（walk-forward）。

- [ ] **Step 3: 全市场模拟盘与 HTML 报告**

Run:

```bash
python -m ashare_quant.cli simulate --data-root data/all --out-dir docs/simulation-all
python -m ashare_quant.cli report --data-root data/all --out-dir docs/simulation-all
```

Expected: `docs/simulation-all/` 下生成 `simulation.md`、`model_returns.csv`、`adjustments.jsonl`、`report.html`。

- [ ] **Step 4: 检查报告关键页并提交**

```bash
git add docs/research/data-research-all.md docs/research/model-selection-all.md docs/simulation-all
git commit -m "docs: 全市场研究/筛选/模拟结果"
```

---

## Task 4: 默认配置与计划任务切换到全市场

- [ ] **Step 1: 修改 config.yaml 默认**

将 `universe_mode` 改为 `all`、`data_root` 改为 `data/all`（全市场为分析对象；需要快速验证时可命令行覆盖）。

- [ ] **Step 2: 修改计划任务脚本默认数据目录**

`scripts/schedule_daily.ps1` 的 `$DataRoot` 默认值改为 `"$ProjectRoot\data\all"`。

- [ ] **Step 3: README 补充全市场说明**

```markdown
## 全市场模式

```bash
python -m ashare_quant.cli fetch --universe all --data-root data/all --years 3
python -m ashare_quant.cli daily --data-root data/all
```
```

- [ ] **Step 4: 全量测试并提交**

Run: `python -m pytest tests/ -q`
Expected: PASS

```bash
git add config.yaml scripts/schedule_daily.ps1 README.md
git commit -m "feat: 默认切换全市场模式并更新文档"
```
