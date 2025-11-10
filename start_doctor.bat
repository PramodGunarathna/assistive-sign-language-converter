@echo off
setlocal
cd /d "%~dp0"
echo ==============================================
echo  Launching Doctor Communication Server
echo ==============================================

:: Ensure pip is available and up to date (optional)
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
  echo Python/pip not found. Please install Python and make sure it's on PATH.
  pause
  exit /b 1
)

:: Install dependencies for the doctor app
echo Installing doctor dependencies...
python -m pip install -r "doctor\requirements.txt" -q

:: Start the server
echo Starting server...
python run_doctor.py

pause
endlocal
