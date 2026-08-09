param(
    [string]$Action = "",
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

# A股量化研究·模拟分析系统 - 一键启动（模拟研究，不构成投资建议）
Set-Location $ProjectRoot

$DataRoot = Join-Path $ProjectRoot "data\all"
$OutDir = Join-Path $ProjectRoot "docs\simulation"
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

function Start-Daily {
    Invoke-Step "每日增量更新 + 报告 + 决策" {
        python -m ashare_quant.cli daily --config $Config `
            --data-root $DataRoot --out-dir $OutDir --model-dir $ModelDir
    }
}

function Start-ForceReport {
    Invoke-Step "强制重算报告与决策" {
        python -m ashare_quant.cli daily --config $Config `
            --data-root $DataRoot --out-dir $OutDir --model-dir $ModelDir --force
    }
}

function Start-Simulate {
    Invoke-Step "模拟盘回测（含反馈调整）" {
        python -m ashare_quant.cli simulate --config $Config `
            --data-root $DataRoot --out-dir $OutDir
        python -m ashare_quant.cli report --config $Config `
            --data-root $DataRoot --out-dir $OutDir
    }
}

function Start-Research {
    Invoke-Step "生成历史数据研究报告" {
        python -m ashare_quant.cli research --config $Config `
            --data-root $DataRoot
    }
}

function Start-Dashboard {
    Invoke-Step "启动仪表盘（http://localhost:8501，Ctrl+C 停止）" {
        python -m streamlit run dashboard.py --server.port 8501
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
        "simulate" { Start-Simulate }
        "research" { Start-Research }
        "dashboard"{ Start-Dashboard }
        "all"      { Start-All }
        default {
            Write-Host "未知操作：$Action" -ForegroundColor Red
            Write-Host "可用：daily | force | simulate | research | dashboard | all"
            exit 1
        }
    }
    exit 0
}

while ($true) {
    Write-Host ""
    Write-Host "========== A股量化研究·模拟分析系统 ==========" -ForegroundColor Cyan
    Write-Host " 1) 每日更新 + 报告 + 决策"
    Write-Host " 2) 强制重算报告与决策"
    Write-Host " 3) 模拟盘回测（含反馈调整）"
    Write-Host " 4) 生成历史数据研究报告"
    Write-Host " 5) 启动仪表盘"
    Write-Host " 6) 完整一条龙（数据->模拟->决策->仪表盘）"
    Write-Host " 0) 退出"
    $choice = Read-Host "请选择"
    switch ($choice) {
        "1" { Start-Daily }
        "2" { Start-ForceReport }
        "3" { Start-Simulate }
        "4" { Start-Research }
        "5" { Start-Dashboard }
        "6" { Start-All }
        "0" { Write-Host "再见"; exit 0 }
        default { Write-Host "无效输入，请重新选择" -ForegroundColor Yellow }
    }
}
