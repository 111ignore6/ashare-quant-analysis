param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

# 一键启动：每日更新（增量） -> 启动仪表盘 -> 打开浏览器
Set-Location $ProjectRoot

$DataRoot = Join-Path $ProjectRoot "data\tencent"

if (-not (Test-Path (Join-Path $DataRoot "manifest.json"))) {
    Write-Host ""
    Write-Host "尚未下载数据（$DataRoot 不存在）。" -ForegroundColor Yellow
    Write-Host "首次使用请运行 start.bat，选择 3 下载全市场数据（8-15 分钟，可断点续传）。"
    Write-Host ""
    Read-Host "按回车退出"
    exit 1
}

Write-Host ""
Write-Host "== 1/2 每日增量更新（数据无变化时约 1-2 秒）==" -ForegroundColor Cyan
python -X utf8 -u -m ashare_quant.cli daily --config config.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "每日更新失败（退出码 $LASTEXITCODE），仍尝试启动仪表盘。" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "== 2/2 启动仪表盘（http://localhost:8501）==" -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList @(
    "-m", "streamlit", "run", "dashboard.py",
    "--server.port", "8501", "--server.headless", "true"
) -WindowStyle Hidden

# 等待端口就绪后打开浏览器
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if ($ready) {
    Start-Process "http://localhost:8501"
    Write-Host "仪表盘已就绪并打开浏览器；关闭本窗口不影响仪表盘运行。" -ForegroundColor Green
} else {
    Write-Host "仪表盘启动较慢，请手动打开 http://localhost:8501" -ForegroundColor Yellow
}
Read-Host "按回车关闭本窗口"
