param(
    [string]$ProjectRoot = "",
    [string]$DataRoot = "",
    [string]$OutDir = "",
    [string]$ModelDir = "",
    [string]$LogFile = "",
    [string[]]$ExtraArgs = @(),
    [switch]$SelfTest
)

# A股量化研究·模拟分析系统 —— 每日任务的"带日志包装器"（由计划任务调用）
#
# 为什么需要这个文件（2026-09-16 实测定案）：
#   旧写法 `& python -X utf8 -u -m ashare_quant.cli daily ... *>> logs\daily_scheduled.log 2>&1`
#   有两个叠加的毛病，缺一不可地让日志变成花屏：
#     ① 解码错：PowerShell 5.1 用 [Console]::OutputEncoding（本机 = GBK/936）去解码
#        原生命令的 stdout，而 python 带 -X utf8、吐的是 UTF-8 字节 → 中文在"写进文件
#        之前"就已经是乱码；GBK 无法映射的字节被替换成 '?'，**信息当场丢失**。
#        实测历史日志 logs\daily_scheduled.log：754 行里 456 行含 '?'，共 835 个 '?'。
#     ② 编码错：`>>` 走 Out-File，PS 5.1 默认编码是 UTF-16LE → 文件里全是 NUL 字节，
#        grep/tail/编辑器直接读是花屏（只按 ① 修是不够的，两个都得修）。
#   本脚本的做法：先把 [Console]::OutputEncoding 设成无 BOM 的 UTF-8（修 ①），
#   再用 .NET 显式按"无 BOM UTF-8"追加写文件（修 ②）。
#
# 历史日志 logs\daily_scheduled.log（UTF-16LE + 上述乱码）已损坏且不可逆，本脚本不动它；
# 新日志写到 logs\daily_scheduled.utf8.log。
#
# 自检（不需要管理员、不碰真实数据）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_daily_logged.ps1 -SelfTest
#   打印 SELFTEST PASS/FAIL（纯 ASCII，乱码也能认），并把自检日志留在 logs\_encoding_selftest.log。

if (-not $ProjectRoot) { $ProjectRoot = (Split-Path -Parent $PSScriptRoot) }
if (-not $DataRoot) { $DataRoot = Join-Path $ProjectRoot "data\tencent" }
if (-not $OutDir) { $OutDir = Join-Path $ProjectRoot "docs\simulation-all" }
if (-not $ModelDir) { $ModelDir = Join-Path $ProjectRoot "models\all" }
if (-not $LogFile) { $LogFile = Join-Path $ProjectRoot "logs\daily_scheduled.utf8.log" }
$Config = Join-Path $ProjectRoot "config.yaml"

$utf8 = New-Object System.Text.UTF8Encoding $false

function Write-LogLine([string]$text) {
    [System.IO.File]::AppendAllText($LogFile, $text + [Environment]::NewLine, $utf8)
}

$logDir = Split-Path -Parent $LogFile
if ($logDir -and -not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

# 修 ①：让 PowerShell 按 UTF-8 解码子进程 stdout。
# 计划任务的 LogonType=Interactive（在用户会话内跑，有隐藏 console），正常情况下可用。
# 万一没有 console 导致设置失败：不吞掉、不中断，把 WARN 写进日志（中文可能乱码，但
# ASCII 信息与退出码照旧完整）。
$encWarn = ""
try {
    [Console]::OutputEncoding = $utf8
} catch {
    $encWarn = "WARN: cannot set [Console]::OutputEncoding ($($_.Exception.Message)); 中文可能乱码"
}

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    Write-LogLine "ERROR: 找不到 python（Get-Command python 为空），无法执行 daily"
    exit 127
}

if ($SelfTest) {
    $selfLog = Join-Path (Split-Path -Parent $LogFile) "_encoding_selftest.log"
    if (Test-Path $selfLog) { Remove-Item $selfLog -Force }
    $savedLog = $LogFile
    $LogFile = $selfLog
    $marker = "SELFTEST-中文-utf8-自检"
    Write-LogLine ("===== selftest " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " =====")
    if ($encWarn) { Write-LogLine $encWarn }
    & $python -X utf8 -u -c "print('$marker')" *>&1 | ForEach-Object { Write-LogLine $_.ToString() }
    $bytes = [System.IO.File]::ReadAllBytes($selfLog)
    $nul = 0
    foreach ($b in $bytes) { if ($b -eq 0) { $nul++ } }
    $hasBom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
    $roundtrip = $utf8.GetString($bytes)
    $LogFile = $savedLog
    if ($nul -eq 0 -and -not $hasBom -and $roundtrip.Contains($marker)) {
        Write-Output ("SELFTEST PASS: " + $selfLog + " bytes=" + $bytes.Length + " NUL=0 BOM=none UTF8-roundtrip=ok")
        exit 0
    }
    Write-Output ("SELFTEST FAIL: " + $selfLog + " bytes=" + $bytes.Length + " NUL=" + $nul + " BOM=" + $hasBom)
    exit 1
}

$dailyArgs = @("-X", "utf8", "-u", "-m", "ashare_quant.cli", "daily",
    "--config", $Config, "--data-root", $DataRoot,
    "--out-dir", $OutDir, "--model-dir", $ModelDir) + $ExtraArgs

Write-LogLine ("===== run start " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " =====")
if ($encWarn) { Write-LogLine $encWarn }

# 修 ②：显式按无 BOM UTF-8 追加，不用 `>>`（PS 5.1 会写成 UTF-16LE）。
& $python @dailyArgs *>&1 | ForEach-Object { Write-LogLine $_.ToString() }
$code = $LASTEXITCODE
if ($null -eq $code) { $code = 1 }

Write-LogLine ("===== run end " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " exit=" + $code + " =====")
exit $code
