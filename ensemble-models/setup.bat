@echo off
REM Setup script for Ensemble Models - Multi-Model Detection System
REM This script sets up both backend and frontend

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ENSEMBLE MODELS - SETUP WIZARD
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    exit /b 1
)

REM Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Node.js not found. Frontend setup will be skipped.
    echo Please install Node.js 16+ to use the React frontend.
    set SKIP_FRONTEND=1
)

echo.
echo Step 1: Setting up Backend...
echo ============================================================
cd backend

if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\Activate.ps1 2>nul || call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo Installing PyTorch (CPU version)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

echo Installing backend dependencies...
python -m pip install -r requirements.txt --quiet

echo Backend setup complete!
echo.

REM Frontend setup
if not defined SKIP_FRONTEND (
    echo Step 2: Setting up Frontend...
    echo ============================================================
    cd ..\frontend
    
    echo Installing Node packages...
    call npm install --silent
    
    echo Frontend setup complete!
    echo.
)

cd ..

echo.
echo ============================================================
echo   SETUP COMPLETE!
echo ============================================================
echo.
echo To run the system:
echo.
echo Terminal 1 (Backend):
echo   cd backend
echo   venv\Scripts\activate.ps1  (or activate.bat on cmd)
echo   python app.py
echo.
echo Terminal 2 (Frontend):
echo   cd frontend
echo   npm start
echo.
echo Then open your browser to: http://localhost:3000
echo.
echo ============================================================
echo.

pause
