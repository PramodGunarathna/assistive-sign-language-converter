@echo off
setlocal
cd /d "%~dp0"
echo ==============================================
echo  Launching Patient Communication Client
echo  with Real-time Sign Language Recognition
echo ==============================================
echo.

:: Ensure pip is available and up to date (optional)
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
  echo ERROR: Python/pip not found. Please install Python and make sure it's on PATH.
  pause
  exit /b 1
)

:: Install dependencies for the patient app
echo Installing patient dependencies...
echo This includes: sounddevice, soundfile, opencv-python, mediapipe, numpy
python -m pip install -r "client\requirements.txt" -q

if %errorlevel% neq 0 (
  echo WARNING: Some dependencies may not have installed correctly.
  echo Please check the error messages above.
  echo.
)

:: Check for optional model integration dependencies
python -c "import cv2; import mediapipe" >nul 2>&1
if %errorlevel% equ 0 (
  echo [OK] Computer vision libraries are available
) else (
  echo [WARNING] Some computer vision libraries may be missing
  echo           Install manually: pip install opencv-python mediapipe numpy
)
echo.

:: Start the client
echo ==============================================
echo Starting Patient Communication Client...
echo ==============================================
echo.
echo Features:
echo   - Real-time sign language capture via webcam
echo   - Automatic translation to text (4-5 words max)
echo   - LLM-enhanced sentence generation
echo   - Receive doctor's voice and text messages
echo.
echo Instructions:
echo   1. Enter the doctor's IP address
echo   2. Enter your name
echo   3. Click Connect
echo   4. Select a trained model (.h5 file)
echo   5. Show your hands to the camera to sign
echo.
echo Press Ctrl+C to exit
echo ==============================================
echo.
python run_patient.py

if %errorlevel% neq 0 (
  echo.
  echo ERROR: Client exited with an error.
  echo Please check the error messages above.
)

pause
endlocal

