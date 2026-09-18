param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DataRoot = "$ProjectRoot\data\tencent",
    [string]$OutDir = "$ProjectRoot\docs\simulation-all",
    [switch]$Force
)

$taskName = "AshareQuantDaily"
$wrapper = Join-Path $PSScriptRoot "run_daily_logged.ps1"
if (-not (Test-Path $wrapper)) {
    Write-Output "找不到包装脚本 $wrapper，中止（缺了它注册出来的任务不会写日志）。"
    exit 1
}

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    Write-Output "计划任务 $taskName 已存在，跳过（如需更新用 -Force）。"
    exit 0
}

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 16:05
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
$logDir = "$ProjectRoot\logs"
# 日志写 UTF-8（无 BOM）。历史文件 logs\daily_scheduled.log 是 UTF-16LE 且写入前就已乱码，
# 不可逆，保持原样不动；编码问题的机制与判据见 run_daily_logged.ps1 顶部与 AGENTS.md「已知问题」。
$logFile = "$logDir\daily_scheduled.utf8.log"
# 用 -File 调包装脚本：日志编码、时间戳、退出码都在 run_daily_logged.ps1 里处理。
# 不要再回到 `& python ... *>> logs\xxx.log 2>&1`：PS 5.1 的 `>>` 写 UTF-16LE，
# 且会用 GBK 解码 python 的 UTF-8 stdout，中文在落盘前就已损坏。
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`" -ProjectRoot `"$ProjectRoot`" -DataRoot `"$DataRoot`" -OutDir `"$OutDir`" -ModelDir `"$ProjectRoot\models\all`" -LogFile `"$logFile`"" `
    -WorkingDirectory $ProjectRoot

if ($existing) {
    # 原地更新（不先删后建）：旧写法 Unregister + Register 中间有一段"任务不存在"的空窗，
    # 中途失败就永久丢任务（2026-08-14 事故）。Set-ScheduledTask 失败时旧任务仍在。
    Set-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings | Out-Null
    Write-Output "已原地更新计划任务 $taskName（改调 run_daily_logged.ps1，日志 $logFile）。"
} else {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
        -Description "A股量化研究：每个交易日收盘后增量更新数据并生成报告（模拟，不构成投资建议）" | Out-Null
    Write-Output "已注册计划任务 $taskName（每周一至五 16:05）。"
}

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
    Write-Output "已开启计划任务 $taskName（每周一至五 16:05，日志 $logFile）。"
} else {
    Disable-ScheduledTask -TaskName $taskName | Out-Null
    Write-Output "计划任务 $taskName 已就位，但 config.yaml 的 auto_update 为 false，保持关闭。"
}
