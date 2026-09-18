param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

# 一键启动：每日更新（增量） -> 启动仪表盘 -> 打开浏览器
Set-Location $ProjectRoot

$DataRoot = Join-Path $ProjectRoot "data\tencent"
$OutDir   = Join-Path $ProjectRoot "docs\simulation-all"
$ModelDir = Join-Path $ProjectRoot "models\all"
$Config   = Join-Path $ProjectRoot "config.yaml"
$Port     = 8501

if (-not (Test-Path (Join-Path $DataRoot "manifest.json"))) {
    Write-Host ""
    Write-Host "尚未下载数据（$DataRoot 不存在）。" -ForegroundColor Yellow
    Write-Host "首次使用请运行 start.bat，选择 3 下载全市场数据（8-15 分钟，可断点续传）。"
    Write-Host ""
    Read-Host "按回车退出"
    exit 1
}

Write-Host ""
Write-Host "== 1/2 每日增量更新（无新交易日约 2 秒；有新交易日约 40 秒）==" -ForegroundColor Cyan
Write-Host "   这一步没有进度条时属正常，请勿关窗。" -ForegroundColor DarkGray
# ⚠️ 四个路径参数一个都不能少：只带 --config 会写进默认 docs\simulation（决策串台），
#    并用错模型池 models\ 而非 models\all。见项目 AGENTS.md「手动跑 daily 必须带路径参数」。
python -X utf8 -u -m ashare_quant.cli daily `
    --config $Config --data-root $DataRoot --out-dir $OutDir --model-dir $ModelDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "每日更新失败（退出码 $LASTEXITCODE），仍尝试启动仪表盘。" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "== 2/2 启动仪表盘（http://localhost:$Port）==" -ForegroundColor Cyan
# 这里显式传 --server.headless true：命令行参数优先于 .streamlit\config.toml，
# 所以本脚本自己负责开浏览器（下面 Start-Process），不受全局配置影响。
Start-Process -FilePath "python" -ArgumentList @(
    "-m", "streamlit", "run", "dashboard.py",
    "--server.port", "$Port", "--server.headless", "true"
) -WindowStyle Hidden

# 等待端口就绪后打开浏览器。冷启动要读 341MB 的 features.parquet，给足 90 秒；
# 每 5 秒打印一次进度，避免看起来像卡死。
$ready = $false
for ($i = 1; $i -le 90; $i++) {
    Start-Sleep -Seconds 1
    if ($i % 5 -eq 0) { Write-Host ("   等待仪表盘就绪… {0}s" -f $i) -ForegroundColor DarkGray }
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$Port" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if ($ready) {
    Start-Process "http://localhost:$Port"
    Write-Host "仪表盘已就绪并打开浏览器；关闭本窗口不影响仪表盘运行。" -ForegroundColor Green
} else {
    Write-Host "仪表盘 90 秒内未就绪。请手动打开 http://localhost:$Port" -ForegroundColor Yellow
    Write-Host "（多半是首次加载较慢；也可看该窗口有无 Python 报错。）" -ForegroundColor DarkGray
}
Read-Host "按回车关闭本窗口"
