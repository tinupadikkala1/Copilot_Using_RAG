@echo off
REM =============================================================================
REM  🚀 Autonomous Customer Support Copilot — Windows Launcher
REM
REM  Double-click this file to start everything:
REM    1. Checks Python + Ollama
REM    2. Installs dependencies (one-time)
REM    3. Builds the vector index
REM    4. Starts the API server + Chat UI + Dashboard
REM
REM  This batch file launches the PowerShell script (start.ps1).
REM  If you get a security error, run this in PowerShell instead:
REM     powershell -ExecutionPolicy Bypass -File start.ps1
REM =============================================================================

title 🚀 Copilot Launcher (Windows)
echo.
echo  ╔════════════════════════════════════════════════╗
echo  ║                                                ║
echo  ║   🚀  Autonomous Customer Support Copilot      ║
echo  ║         Windows Launcher                       ║
echo  ║                                                ║
echo  ╚════════════════════════════════════════════════╝
echo.
echo  Starting PowerShell script...
echo  (If asked, click "Yes" to allow execution)
echo.

REM Launch the PowerShell script
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" %*

REM If PowerShell script exits with error, pause to show the error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ⚠️  The script exited with code %ERRORLEVEL%.
    echo  Try running this in PowerShell instead:
    echo     powershell -ExecutionPolicy Bypass -File start.ps1
    echo.
    pause
)
