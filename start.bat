@echo off
setlocal
cd /d "%~dp0"

py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 goto havepy
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 goto havepython

echo Lentic needs Python 3.10 or newer.
echo Install it from https://www.python.org/downloads/
echo On the first screen, check "Add python.exe to PATH", then run this again.
echo.
pause
exit /b 1

:havepy
set "PYEXE=py"
set "PYARGS=-3"
goto setup

:havepython
set "PYEXE=python"
set "PYARGS="

:setup
if exist ".venv\Scripts\python.exe" goto install
echo Creating a local Python environment...
"%PYEXE%" %PYARGS% -m venv .venv
if not errorlevel 1 goto install
echo Could not create the environment. The messages above say why.
echo.
pause
exit /b 1

:install
echo Installing Lentic. The first time can take a minute.
".venv\Scripts\python.exe" -m pip install -e .
if not errorlevel 1 goto run
echo.
echo Install failed. The messages above say why.
pause
exit /b 1

:run
echo.
".venv\Scripts\python.exe" -m lentic --gui
echo.
pause