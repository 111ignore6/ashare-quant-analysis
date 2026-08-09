@echo off
rem Entry point (double-click). Actual logic lives in scripts\start.ps1
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
pause
