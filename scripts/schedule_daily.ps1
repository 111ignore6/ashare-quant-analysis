param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DataRoot = "$ProjectRoot\data\tencent",
    [string]$OutDir = "$ProjectRoot\docs\simulation-all",
    [switch]$Force
)

$taskName = "AshareQuantDaily"
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    if ($Force) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Output "已移除旧任务 $taskName，重新注册（需要管理员权限）。"
    } else {
        Write-Output "计划任务 $taskName 已存在，跳过（如需更新用 -Force）。"
        exit 0
    }
}
$python = (Get-Command python).Source
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 16:05
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
$logDir = "$ProjectRoot\logs"
$logFile = "$logDir\daily_scheduled.log"
# 用 PowerShell 包装：显式指定模型目录 + 输出重定向到日志（任务失败可追溯）
$inner = "New-Item -ItemType Directory -Force -Path '$logDir' | Out-Null; " +
         "& '$python' -X utf8 -u -m ashare_quant.cli daily " +
         "--config '$ProjectRoot\config.yaml' --data-root '$DataRoot' " +
         "--out-dir '$OutDir' --model-dir '$ProjectRoot\models\all' " +
         "*>> '$logFile' 2>&1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$inner`"" `
    -WorkingDirectory $ProjectRoot
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "A股量化研究：每个交易日收盘后增量更新数据并生成报告（模拟，不构成投资建议）"
# 尊重 config.yaml 的 auto_update 开关（默认开启）
$autoUpdate = $true
if (Test-Path "$ProjectRoot\config.yaml") {
    $line = Select-String -Path "$ProjectRoot\config.yaml" -Pattern "^\s*auto_update:\s*(\w+)" | Select-Object -First 1
    if ($line -and $line.Matches[0].Groups[1].Value -eq "false") {
        $autoUpdate = $false
    }
}
if ($autoUpdate) {
    Enable-ScheduledTask -TaskName $taskName | Out-Null
    Write-Output "已注册并开启计划任务 $taskName（每周一至五 16:05）"
} else {
    Disable-ScheduledTask -TaskName $taskName | Out-Null
    Write-Output "已注册计划任务 $taskName，但 config.yaml 的 auto_update 为 false，保持关闭。"
}
