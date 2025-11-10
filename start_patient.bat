@echo off
setlocal
cd /d "%~dp0"
echo ==============================================
echo  Launching Patient Communication Client
echo ==============================================

:: Ensure pip is available and up to date (optional)
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
  echo Python/pip not found. Please install Python and make sure it's on PATH.
  pause
  exit /b 1
)

:: Install dependencies for the patient app
echo Installing patient dependencies...
python -m pip install -r "client\requirements.txt" -q

:: Start the client
echo Starting client...
python run_patient.py

pause
endlocal
