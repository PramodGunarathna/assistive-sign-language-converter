@echo off
setlocal
cd /d "%~dp0"
echo ==============================================
echo  Launching Doctor Communication Server
echo  with Text-to-Speech (TTS) Support
echo ==============================================
echo.

:: Ensure pip is available and up to date (optional)
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
  echo ERROR: Python/pip not found. Please install Python and make sure it's on PATH.
  pause
  exit /b 1
)

:: Install dependencies for the doctor app
echo Installing doctor dependencies...
echo This includes: sounddevice, soundfile, and pyttsx3 (TTS)
python -m pip install -r "doctor\requirements.txt" -q

if %errorlevel% neq 0 (
  echo WARNING: Some dependencies may not have installed correctly.
  echo Please check the error messages above.
  echo.
)

:: Check if pyttsx3 is available
python -c "import pyttsx3" >nul 2>&1
if %errorlevel% equ 0 (
  echo [OK] Text-to-Speech (TTS) is available
  echo       Patient messages will be automatically spoken
) else (
  echo [WARNING] Text-to-Speech (pyttsx3) not available
  echo           Install manually: pip install pyttsx3
)
echo.

:: Start the server
echo ==============================================
echo Starting Doctor Communication Server...
echo ==============================================
echo.
echo Features:
echo   - Receive patient sign language translations
echo   - Text-to-Speech: Patient messages read aloud
echo   - Voice recording and transcription
echo   - Text message responses
echo.
echo Press Ctrl+C to stop the server
echo ==============================================
echo.
python run_doctor.py

if %errorlevel% neq 0 (
  echo.
  echo ERROR: Server exited with an error.
  echo Please check the error messages above.
)

pause
endlocal

