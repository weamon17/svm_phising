@echo off
rem Run the 44 unit tests (from-scratch SMO vs LIBSVM, KKT, features, data split, demo).
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Environment not found. Run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pytest -v
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" (echo All tests passed.) else (echo Some tests failed: exit code %CODE%.)
pause
exit /b %CODE%
