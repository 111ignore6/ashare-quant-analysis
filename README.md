# A股量化研究·模拟分析系统

基于真实 A 股数据（AKShare 主 / BaoStock 备）的历史数据研究项目。

> 本项目为模拟研究，不构成任何投资建议。

## 运行

```bash
pip install -r requirements.txt
python -m ashare_quant.cli fetch --universe csi300 --years 3   # 下载沪深300数据
python -m ashare_quant.cli research                            # 生成研究报告
```

## 阶段

1. 数据底座与历史数据研究（已完成）
2. 候选模型构建与筛选（已完成）
3. 模拟盘与反馈调整（已完成）
4. 每日增量更新与自动执行（已完成）

## 每日更新

```powershell
python -m ashare_quant.cli daily          # 增量更新数据并生成当日 HTML 报告
.\scripts\schedule_daily.ps1              # 注册 Windows 计划任务（周一至五 16:05 自动运行）
```

## 全市场模式

```bash
python -m ashare_quant.cli fetch --universe all --data-root data/all --years 3
python -m ashare_quant.cli daily --data-root data/all
```

## 算法研究

```bash
python -m ashare_quant.cli benchmark --data-root data/all   # 算法表现对比（全市场）
python -m ashare_quant.cli decision --data-root data/all    # 训练模型并生成当日模拟持仓
```

## 仪表盘

```bash
python -m streamlit run dashboard.py
```
