param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DataRoot = "$ProjectRoot\data\tencent",
    [string]$OutDir = "$ProjectRoot\docs\simulation"
)

$taskName = "AshareQuantDaily"
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "计划任务 $taskName 已存在，跳过。"
    exit 0
}
$python = (Get-Command python).Source
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 16:05
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
$action = New-ScheduledTaskAction -Execute $python `
    -Argument "-m ashare_quant.cli daily --config `"$ProjectRoot\config.yaml`" --data-root `"$DataRoot`" --out-dir `"$OutDir`"" `
    -WorkingDirectory $ProjectRoot
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "A股量化研究：每个交易日收盘后增量更新数据并生成报告（模拟，不构成投资建议）"
Write-Output "已注册计划任务 $taskName（每周一至五 16:05）"
