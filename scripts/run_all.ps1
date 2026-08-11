param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-Location $ProjectRoot

$DataRoot = Join-Path $ProjectRoot "data\tencent"
$OutDir = Join-Path $ProjectRoot "docs\simulation-all"
$ModelDir = Join-Path $ProjectRoot "models\all"

Write-Output "== 1/4 数据（幂等：已有本地数据则跳过下载）=="
$manifest = Join-Path $DataRoot "manifest.json"
if (-not (Test-Path $manifest)) {
    python -X utf8 -u -m ashare_quant.cli fetch --universe all --data-root $DataRoot --years 3
} else {
    Write-Output "本地数据已存在（$DataRoot），跳过全量下载。"
}

Write-Output "== 2/4 模拟盘与报告 =="
python -X utf8 -u -m ashare_quant.cli simulate --data-root $DataRoot --out-dir $OutDir
python -X utf8 -u -m ashare_quant.cli report --data-root $DataRoot --out-dir $OutDir

Write-Output "== 3/4 训练模型并生成今日模拟持仓 =="
python -X utf8 -u -m ashare_quant.cli decision --data-root $DataRoot --model-dir $ModelDir --out $OutDir

Write-Output "== 4/4 启动仪表盘（后台运行，不阻塞本窗口）=="
$existing = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($existing) {
    Write-Output "仪表盘已在运行：http://localhost:8501（跳过启动）"
} else {
    Start-Process python -ArgumentList @(
        "-m", "streamlit", "run", "dashboard.py",
        "--server.port", "8501", "--server.headless", "false"
    ) -WorkingDirectory $ProjectRoot -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Output "仪表盘已启动：http://localhost:8501（如未自动打开请手动访问）"
}
Write-Output "一条龙完成。"
