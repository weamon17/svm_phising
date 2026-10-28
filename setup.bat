@echo off
rem One-time setup: create .venv with Python 3.12 and install the pinned packages.
setlocal
cd /d "%~dp0"

set "PY="
py -3.12 -c "" >nul 2>&1 && set "PY=py -3.12"
if not defined PY python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 && set "PY=python"
if not defined PY (
    echo [ERROR] Python 3.12 is required ^(3.13 is not supported by numpy 1.26.4^).
    echo         Download: https://www.python.org/downloads/release/python-31210/
    goto :fail
)

rem Windows limits paths to 260 characters; files inside .venv add up to ~150 more.
%PY% -c "import os, sys; sys.exit(len(os.getcwd()) > 100)" >nul 2>&1 || (
    echo [ERROR] This folder path is too long for Windows ^(over 100 characters^):
    echo         %CD%
    echo         Move the folder to a short path such as C:\SVM and run setup.bat again.
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    %PY% -m venv .venv || goto :fail
)
echo Installing packages ^(1-3 minutes on first run^) ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || goto :fail

echo.
echo Setup complete. Next: run_tests.bat, run_demo.bat
pause
exit /b 0

:fail
echo.
echo Setup failed. See the message above.
pause
exit /b 1
