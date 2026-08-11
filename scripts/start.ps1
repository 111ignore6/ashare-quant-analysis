param(
    [string]$Action = "",
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

# A股量化研究·模拟分析系统 - 一键启动（模拟研究，不构成投资建议）
Set-Location $ProjectRoot

$DataRoot = Join-Path $ProjectRoot "data\tencent"
$OutDir = Join-Path $ProjectRoot "docs\simulation-all"
$ModelDir = Join-Path $ProjectRoot "models\all"
$Config = Join-Path $ProjectRoot "config.yaml"

function Invoke-Step([string]$name, [scriptblock]$block) {
    Write-Host ""
    Write-Host "== $name ==" -ForegroundColor Cyan
    & $block
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
        Write-Host "上一步失败（退出码 $LASTEXITCODE）" -ForegroundColor Red
    }
}

function Invoke-Python([string[]]$Arguments) {
    # -X utf8：强制 UTF-8；-u：无缓冲，保证进度实时显示
    & python -X utf8 -u @Arguments
}

function Start-Daily {
    Invoke-Step "每日增量更新 + 报告 + 决策" {
        Invoke-Python @("-m", "ashare_quant.cli", "daily", "--config", $Config,
                        "--data-root", $DataRoot, "--out-dir", $OutDir,
                        "--model-dir", $ModelDir)
    }
}

function Start-ForceReport {
    Invoke-Step "强制重算报告与决策" {
        Invoke-Python @("-m", "ashare_quant.cli", "daily", "--config", $Config,
                        "--data-root", $DataRoot, "--out-dir", $OutDir,
                        "--model-dir", $ModelDir, "--force")
    }
}

function Start-Simulate {
    Invoke-Step "模拟盘回测（含反馈调整）" {
        Invoke-Python @("-m", "ashare_quant.cli", "simulate", "--config", $Config,
                        "--data-root", $DataRoot, "--out-dir", $OutDir)
        Invoke-Python @("-m", "ashare_quant.cli", "report", "--config", $Config,
                        "--data-root", $DataRoot, "--out-dir", $OutDir)
    }
}

function Start-Research {
    Invoke-Step "生成历史数据研究报告" {
        Invoke-Python @("-m", "ashare_quant.cli", "research", "--config", $Config,
                        "--data-root", $DataRoot)
    }
}

function Start-Fetch {
    Invoke-Step "下载/更新全市场数据（首次约 10-20 分钟）" {
        Invoke-Python @("-m", "ashare_quant.cli", "fetch", "--config", $Config,
                        "--universe", "all", "--data-root", $DataRoot, "--years", "3")
    }
}

function Start-Dashboard {
    Invoke-Step "启动仪表盘（http://localhost:8501，Ctrl+C 停止）" {
        Invoke-Python @("-m", "streamlit", "run", "dashboard.py", "--server.port", "8501")
    }
}

function Start-All {
    Write-Host "== 完整一条龙：数据 -> 模拟盘 -> 决策 -> 仪表盘 ==" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "run_all.ps1")
}

if ($Action -ne "") {
    switch ($Action.ToLower()) {
        "daily"    { Start-Daily }
        "force"    { Start-ForceReport }
        "fetch"    { Start-Fetch }
        "simulate" { Start-Simulate }
        "research" { Start-Research }
        "dashboard"{ Start-Dashboard }
        "all"      { Start-All }
        default {
            Write-Host "未知操作：$Action" -ForegroundColor Red
            Write-Host "可用：daily | force | fetch | simulate | research | dashboard | all"
            exit 1
        }
    }
    exit 0
}

# 首次运行引导
$DataExists = Test-Path (Join-Path $DataRoot "manifest.json")
$ModelExists = Test-Path (Join-Path $ModelDir "meta.json")
if (-not $DataExists) {
    Write-Host ""
    Write-Host "尚未发现本地数据（$DataRoot）。" -ForegroundColor Yellow
    Write-Host "首次使用步骤：" -ForegroundColor Yellow
    Write-Host "  1) 选择 3 下载/更新全市场数据（约 10-20 分钟，有进度显示）"
    Write-Host "  2) 下载完成后选择 1 每日更新，自动生成报告与今日模拟持仓"
} elseif (-not $ModelExists) {
    Write-Host ""
    Write-Host "提示：模型尚未训练。首次选择 1）每日更新 时会自动训练（约 1 分钟）。" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "数据与模型已就绪，可直接选择 1）每日更新。" -ForegroundColor Green
}

while ($true) {
    Write-Host ""
    Write-Host "========== A股量化研究·模拟分析系统 ==========" -ForegroundColor Cyan
    Write-Host " 1) 每日更新 + 报告 + 决策"
    Write-Host " 2) 强制重算报告与决策"
    Write-Host " 3) 下载/更新全市场数据（首次约 10-20 分钟）"
    Write-Host " 4) 模拟盘回测（含反馈调整）"
    Write-Host " 5) 生成历史数据研究报告"
    Write-Host " 6) 启动仪表盘"
    Write-Host " 7) 完整一条龙（数据->模拟->决策->仪表盘）"
    Write-Host " 8) 每日自动更新：查看/开启/关闭（默认开启）"
    Write-Host " 0) 退出"
    $choice = Read-Host "请选择"
    switch ($choice) {
        "1" { Start-Daily }
        "2" { Start-ForceReport }
        "3" { Start-Fetch }
        "4" { Start-Simulate }
        "5" { Start-Research }
        "6" { Start-Dashboard }
        "7" { Start-All }
        "8" {
            & (Join-Path $PSScriptRoot "toggle_auto_update.ps1") -Action status
            $opt = Read-Host "输入 on 开启 / off 关闭（直接回车返回）"
            if ($opt -eq "on" -or $opt -eq "off") {
                & (Join-Path $PSScriptRoot "toggle_auto_update.ps1") -Action $opt
            }
        }
        "0" { Write-Host "再见"; exit 0 }
        default { Write-Host "无效输入，请重新选择" -ForegroundColor Yellow }
    }
}
