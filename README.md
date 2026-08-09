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

1. 数据底座与历史数据研究（当前）
2. 候选模型构建与筛选（研究结论产出后制定）
3. 模拟盘与反馈调整
4. 每日增量更新与自动执行
