# Setup script for Ensemble Models - Multi-Model Detection System
# Run this script from the ensemble-models directory

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   ENSEMBLE MODELS - SETUP WIZARD" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check Node
$nodeFound = $false
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js found: $nodeVersion" -ForegroundColor Green
    $nodeFound = $true
} catch {
    Write-Host "⚠ WARNING: Node.js not found. Frontend setup will be skipped." -ForegroundColor Yellow
    Write-Host "  Please install Node.js 16+ for React frontend." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 1: Setting Up Backend..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Push-Location "backend"

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet | Out-Null

# Install PyTorch
Write-Host "Installing PyTorch (CPU version)..." -ForegroundColor Yellow
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet | Out-Null

# Install dependencies
Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt --quiet | Out-Null

Write-Host "✓ Backend setup complete!" -ForegroundColor Green
Write-Host ""

Pop-Location

# Frontend setup
if ($nodeFound) {
    Write-Host "Step 2: Setting Up Frontend..." -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    
    Push-Location "frontend"
    
    Write-Host "Installing Node packages..." -ForegroundColor Yellow
    npm install --silent | Out-Null
    
    Write-Host "✓ Frontend setup complete!" -ForegroundColor Green
    Write-Host ""
    
    Pop-Location
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "   SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the system:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Terminal 1 (Backend):" -ForegroundColor Yellow
Write-Host "  cd backend" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python app.py" -ForegroundColor White
Write-Host ""
Write-Host "Terminal 2 (Frontend):" -ForegroundColor Yellow
Write-Host "  cd frontend" -ForegroundColor White
Write-Host "  npm start" -ForegroundColor White
Write-Host ""
Write-Host "Then open your browser to: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
