@echo off
rem Start the offline web demo and open it in the default browser. Close this window to stop it.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Environment not found. Run setup.bat first.
    pause
    exit /b 1
)
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start "" http://127.0.0.1:%PORT%"
".venv\Scripts\python.exe" demo\server.py %PORT%
pause
