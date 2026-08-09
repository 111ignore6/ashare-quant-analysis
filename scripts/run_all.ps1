param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-Location $ProjectRoot

Write-Output "== 1/4 数据（幂等，已有则跳过）=="
python -m ashare_quant.cli fetch --universe all --data-root data/all --years 3

Write-Output "== 2/4 模拟盘与报告 =="
python -m ashare_quant.cli simulate --data-root data/all --out-dir docs/simulation-all
python -m ashare_quant.cli report --data-root data/all --out-dir docs/simulation-all

Write-Output "== 3/4 训练模型并生成今日模拟持仓 =="
python -m ashare_quant.cli decision --data-root data/all --model-dir models/all --out docs/decision

Write-Output "== 4/4 启动仪表盘 =="
Write-Output "仪表盘地址: http://localhost:8501"
python -m streamlit run dashboard.py --server.port 8501
