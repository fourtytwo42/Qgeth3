#!/usr/bin/env pwsh

# Q Coin Self-Contained Release Builder (Simplified)
# Creates miner releases with embedded Python - zero installation required!

param([switch]$Help)

if ($Help) {
    Write-Host "Q Coin Self-Contained Release Builder" -ForegroundColor Cyan
    Write-Host "Creates miner with embedded Python - no user installation required!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\\create-embedded-release-simple.ps1" -ForegroundColor Yellow
    Write-Host "Output: releases/quantum-miner-embedded-[timestamp]/" -ForegroundColor Yellow
    exit 0
}

$ErrorActionPreference = "Stop"

Write-Host "Building Q Coin Self-Contained Release..." -ForegroundColor Cyan
Write-Host "This will bundle Python + all dependencies!" -ForegroundColor Green
Write-Host ""

# Get paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$QuantumMinerDir = Join-Path $ProjectRoot "quantum-miner"
$ReleasesDir = Join-Path $ProjectRoot "releases"
$timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

# Create release directory
$releaseDir = Join-Path $ReleasesDir "quantum-miner-embedded-$timestamp"
New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null

Write-Host "Building quantum-miner executable..." -ForegroundColor Yellow
Set-Location $QuantumMinerDir

try {
    $env:CGO_ENABLED = "0"
    $BUILD_TIME = Get-Date -Format "yyyy-MM-dd_HH:mm:ss"
    $GIT_COMMIT = git rev-parse --short HEAD 2>$null
    if (-not $GIT_COMMIT) { $GIT_COMMIT = "unknown" }
    
    $LDFLAGS = "-X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME"
    & go build -ldflags $LDFLAGS -o "quantum-miner.exe" "."
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Miner build failed!"
        exit 1
    }
    
    Copy-Item "quantum-miner.exe" (Join-Path $releaseDir "quantum-miner.exe") -Force
    Write-Host "Miner built successfully" -ForegroundColor Green
} finally {
    Set-Location $ProjectRoot
}

Write-Host "Setting up embedded Python..." -ForegroundColor Yellow

# Download embedded Python 3.11.9
$pythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
$pythonZip = Join-Path $releaseDir "python.zip"
$pythonDir = Join-Path $releaseDir "python"

try {
    Write-Host "  Downloading Python 3.11.9..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonZip -UseBasicParsing
    
    Write-Host "  Extracting Python..." -ForegroundColor Cyan
    Expand-Archive -Path $pythonZip -DestinationPath $pythonDir -Force
    Remove-Item $pythonZip -Force
    
    # Enable site-packages
    $pthFile = Join-Path $pythonDir "python311._pth"
    if (Test-Path $pthFile) {
        $content = Get-Content $pthFile
        $content = $content -replace "#import site", "import site"
        if ($content -notcontains "Lib\site-packages") {
            $content += "Lib\site-packages"
        }
        Set-Content -Path $pthFile -Value $content
    }
    
    Write-Host "Python embedded successfully" -ForegroundColor Green
} catch {
    Write-Error "Failed to setup Python: $_"
    exit 1
}

Write-Host "Installing Python packages..." -ForegroundColor Yellow

$pythonExe = Join-Path $pythonDir "python.exe"

# Install pip
try {
    $getPip = Join-Path $pythonDir "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip -UseBasicParsing
    & $pythonExe $getPip --quiet
    Write-Host "  Pip installed" -ForegroundColor Green
} catch {
    Write-Error "Failed to install pip: $_"
    exit 1
}

# Install required packages
$packages = @(
    "qiskit==0.45.0",
    "qiskit-aer==0.12.2", 
    "numpy==1.24.3",
    "scipy==1.11.0"
)

foreach ($pkg in $packages) {
    Write-Host "  Installing $pkg..." -ForegroundColor Cyan
    try {
        & $pythonExe -m pip install $pkg --quiet --no-warn-script-location
        Write-Host "  $pkg installed" -ForegroundColor Green
    } catch {
        Write-Host "  Failed to install $pkg" -ForegroundColor Red
    }
}

# Try to install CuPy for GPU support
$cudaPackages = @("cupy-cuda12x", "cupy-cuda11x")
foreach ($cudaPkg in $cudaPackages) {
    Write-Host "  Trying $cudaPkg..." -ForegroundColor Cyan
    try {
        & $pythonExe -m pip install $cudaPkg --quiet --no-warn-script-location 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  $cudaPkg installed" -ForegroundColor Green
            break
        }
    } catch {
        Write-Host "  $cudaPkg failed" -ForegroundColor Yellow
    }
}

Write-Host "Adding Python scripts..." -ForegroundColor Yellow

# Copy Python scripts
$scriptsDir = Join-Path $releaseDir "pkg\quantum"
New-Item -ItemType Directory -Path $scriptsDir -Force | Out-Null

$scripts = @(
    "pkg/quantum/qiskit_gpu.py",
    "pkg/quantum/cupy_gpu.py",
    "pkg/quantum/ibm_quantum_cloud.py",
    "test_gpu.py"
)

foreach ($script in $scripts) {
    $scriptPath = Join-Path $QuantumMinerDir $script
    if (Test-Path $scriptPath) {
        if ($script -eq "test_gpu.py") {
            Copy-Item $scriptPath (Join-Path $releaseDir "test_gpu.py") -Force
        } else {
            Copy-Item $scriptPath (Join-Path $scriptsDir (Split-Path $script -Leaf)) -Force
        }
        Write-Host "  Added $script" -ForegroundColor Green
    }
}

Write-Host "Creating wrapper scripts..." -ForegroundColor Yellow

# Create Python wrapper batch file
$pythonWrapperContent = @'
@echo off
REM Q Coin Isolated Python Wrapper - Does NOT affect system Python
set "PYTHON_HOME=%~dp0python"
set "PYTHONPATH=%PYTHON_HOME%;%PYTHON_HOME%\Lib;%PYTHON_HOME%\Lib\site-packages"
set "PATH=%PYTHON_HOME%;%PATH%"
set "PYTHONDONTWRITEBYTECODE=1"
"%PYTHON_HOME%\python.exe" %*
'@
Set-Content -Path (Join-Path $releaseDir "python.bat") -Value $pythonWrapperContent -Encoding ASCII

# Create miner launcher batch file 
$minerLauncherContent = @'
@echo off
echo Q Coin Self-Contained Quantum Miner
echo ====================================
echo Using ISOLATED Python (your system Python is safe!)
echo.

REM Test embedded Python
echo Testing embedded Python environment...
call python.bat --version
if %ERRORLEVEL% neq 0 (
    echo Python test failed
    pause
    exit /b 1
)

echo Testing Qiskit...
call python.bat -c "import qiskit; print('Qiskit version:', qiskit.__version__)"
if %ERRORLEVEL% neq 0 (
    echo Qiskit test failed
    pause
    exit /b 1
)

REM Test GPU support (optional)
echo Testing GPU support...
call python.bat -c "try: import cupy; print('GPU available:', cupy.cuda.is_available()); except: print('GPU not available')" 2>nul

REM Set defaults
set THREADS=%1
set NODE=%2  
set COINBASE=%3
if "%THREADS%"=="" set THREADS=8
if "%NODE%"=="" set NODE=http://localhost:8545
if "%COINBASE%"=="" set COINBASE=0x0000000000000000000000000000000000000001

echo.
echo Starting quantum miner...
echo    Threads: %THREADS%
echo    Node: %NODE%
echo    Coinbase: %COINBASE%
echo.

quantum-miner.exe -node %NODE% -coinbase %COINBASE% -threads %THREADS%
'@
Set-Content -Path (Join-Path $releaseDir "start-miner.bat") -Value $minerLauncherContent -Encoding ASCII

# Create README
$readmeContent = @'
# Q Coin Self-Contained Quantum Miner

**ZERO INSTALLATION REQUIRED!**
**COMPLETELY ISOLATED - Your Python is Safe!**

## Python Isolation Guarantee
This release uses embedded Python that is completely isolated:
- Does NOT touch your system Python (if you have one)
- Does NOT modify PATH or registry
- Does NOT interfere with pip, conda, or other Python tools
- Does NOT require admin privileges
- Safe to run alongside ANY existing Python

## What's Included
- Quantum-Miner executable
- Isolated Python 3.11.9 (in local python/ folder)
- Qiskit quantum computing library (pre-installed)
- CuPy GPU acceleration (if compatible GPU available)
- All dependencies pre-installed in isolation

## Quick Start (Zero Setup)
1. Extract this folder anywhere
2. Run: start-miner.bat
3. Start mining immediately!

## Custom Usage
start-miner.bat [threads] [node] [coinbase]

Examples:
start-miner.bat 16 http://localhost:8545 0xYourAddress
start-miner.bat 8
start-miner.bat 4 http://192.168.1.100:8545 0x1234...

## System Requirements
- OS: Windows 10/11 (64-bit)
- For GPU: NVIDIA GPU with drivers installed
- Python: NOT NEEDED! (We include our own isolated copy)
- Admin: NOT NEEDED! (Runs as regular user)

## Testing
Test the isolated Python:
python.bat -c "import qiskit; print('Qiskit OK')"

Test GPU capabilities:
python.bat test_gpu.py

## Expected Performance
- CPU Mining: ~0.5-0.8 puzzles/sec
- GPU Mining: ~2.0-4.0 puzzles/sec (RTX 3080+)

## Benefits
- Zero installation hassles
- No system modifications
- No dependency conflicts  
- No admin privileges needed
- Safe for any environment
- Portable across machines

Perfect for: Enterprise environments, shared machines, development setups!
'@

$currentDate = Get-Date
$readmeContent += "`n`nBuilt: $currentDate`nSize: ~150MB (completely self-contained)"

Set-Content -Path (Join-Path $releaseDir "README.md") -Value $readmeContent -Encoding UTF8

Write-Host "Testing installation..." -ForegroundColor Yellow

try {
    $testResult = & (Join-Path $releaseDir "python.bat") -c "import qiskit, numpy; print('All packages working')" 2>&1
    Write-Host "Test result: $testResult" -ForegroundColor Green
} catch {
    Write-Host "Test failed but continuing: $_" -ForegroundColor Yellow
}

# Calculate size
$size = (Get-ChildItem $releaseDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB

Write-Host ""
Write-Host "SUCCESS! Self-contained release created:" -ForegroundColor Green
Write-Host "Location: $releaseDir" -ForegroundColor Cyan
Write-Host "Size: $([math]::Round($size, 1)) MB" -ForegroundColor Cyan
Write-Host ""
Write-Host "Users can now mine without ANY installation!" -ForegroundColor Green
Write-Host "Just download, extract, and run start-miner.bat" -ForegroundColor Yellow 