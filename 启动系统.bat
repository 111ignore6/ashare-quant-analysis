@echo off
rem One-click launcher: daily update -> dashboard -> browser
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch_all.ps1"
