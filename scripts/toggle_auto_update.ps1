param(
    [string]$Action = "status"
)

# 每日自动更新开关：on 开启 / off 关闭（保留任务，可随时切回）/ status 查看
$taskName = "AshareQuantDaily"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "计划任务 $taskName 尚未注册。运行 .\scripts\schedule_daily.ps1 注册。" -ForegroundColor Yellow
    exit 1
}

switch ($Action.ToLower()) {
    "on" {
        Enable-ScheduledTask -TaskName $taskName | Out-Null
        Write-Host "已开启每日自动更新（周一至五 16:05，错过补跑）。" -ForegroundColor Green
    }
    "off" {
        Disable-ScheduledTask -TaskName $taskName | Out-Null
        Write-Host "已关闭每日自动更新；任务已保留，可随时用 on 重新开启。" -ForegroundColor Yellow
    }
    "status" {
        $state = (Get-ScheduledTask -TaskName $taskName).State
        Write-Host "每日自动更新：$state"
        if ($state -eq "Ready") {
            Write-Host "（开启中：周一至五 16:05 自动运行，无需手动打开）"
        } elseif ($state -eq "Disabled") {
            Write-Host "（已关闭；需要时运行 .\scripts\toggle_auto_update.ps1 -Action on 重新开启）"
        }
    }
    default {
        Write-Host "用法：.\scripts\toggle_auto_update.ps1 [-Action on | off | status]"
        exit 1
    }
}
