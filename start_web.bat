@echo off
title AlphaMaster Web Server
cd /d "C:\Alpha_master"

echo ============================================================
echo   AlphaMaster - Quant Factor Mining Center
echo   Web Console Launcher
echo ============================================================
echo.

REM Check if Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
  echo [ERROR] Python is not found on PATH.
  echo Please install Python 3.11+ and add it to your system PATH.
  echo.
  pause
  exit /b 1
)

echo [1/2] Starting web server on http://127.0.0.1:8765 ...
echo      The browser will open automatically in 3 seconds.
echo.
echo      Press Ctrl+C in this window to stop the server.
echo ============================================================
echo.

REM Open the browser after a 3-second delay (gives the server time to start)
start "" cmd /c "ping 127.0.0.1 -n 4 >nul 2>&1 & start http://127.0.0.1:8765"

REM Start the web server (blocks until Ctrl+C)
python run_web.py --host 127.0.0.1 --port 8765

echo.
echo ============================================================
echo   Server has stopped.
echo ============================================================
pause
