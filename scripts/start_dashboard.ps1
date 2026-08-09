param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 8501
)

Set-Location $ProjectRoot
Write-Output "启动仪表盘: http://localhost:$Port"
python -m streamlit run dashboard.py --server.port $Port
