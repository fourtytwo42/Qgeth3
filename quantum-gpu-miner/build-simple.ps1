#!/usr/bin/env pwsh
# Simple Quantum-GPU-Miner Build Script
param(
    [string]$Mode = "cpu",
    [switch]$Clean
)

Write-Host "Quantum-GPU-Miner Build Script" -ForegroundColor Blue
Write-Host "Mode: $Mode" -ForegroundColor Yellow

if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    Remove-Item -Path "quantum-gpu-miner*.exe" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "pkg\quantum\*.dll" -Force -ErrorAction SilentlyContinue
    go clean -cache
    Write-Host "Cleaned successfully" -ForegroundColor Green
}

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Blue

# Check Go
$goVersion = go version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Go not found" -ForegroundColor Red
    exit 1
}
Write-Host "Go: OK" -ForegroundColor Green

# Check Python
$pythonVersion = python --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}
Write-Host "Python: OK" -ForegroundColor Green

# Check Qiskit
python -c "import qiskit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing Qiskit..." -ForegroundColor Yellow
    pip install qiskit qiskit-aer numpy cupy-cuda12x
}
Write-Host "Qiskit: OK" -ForegroundColor Green

if ($Mode -eq "cpu") {
    Write-Host "Building CPU version..." -ForegroundColor Blue
    $env:CGO_ENABLED = "1"
    go build -ldflags "-s -w" -o quantum-gpu-miner-cpu.exe .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: CPU version built" -ForegroundColor Green
        $version = .\quantum-gpu-miner-cpu.exe --version
        Write-Host "Version: $version" -ForegroundColor Cyan
    } else {
        Write-Host "ERROR: CPU build failed" -ForegroundColor Red
        exit 1
    }
}
elseif ($Mode -eq "gpu") {
    # Check CUDA prerequisites
    Write-Host "Checking CUDA prerequisites..." -ForegroundColor Blue
    
    nvcc --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: NVCC not found" -ForegroundColor Red
        exit 1
    }
    Write-Host "NVCC: OK" -ForegroundColor Green
    
    $vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vsPath)) {
        $vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        if (-not (Test-Path $vsPath)) {
            Write-Host "ERROR: Visual Studio Build Tools not found" -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "Visual Studio: OK" -ForegroundColor Green
    
    # Build CUDA library
    Write-Host "Building CUDA library..." -ForegroundColor Blue
    Push-Location "pkg\quantum"
    
    $cudaCmd = "`"$vsPath`" && nvcc -shared -O3 -o libquantum_cuda.dll quantum_cuda.cu"
    cmd /c $cudaCmd
    
    if (Test-Path "libquantum_cuda.dll") {
        Write-Host "CUDA library: OK" -ForegroundColor Green
    } else {
        Write-Host "ERROR: CUDA library build failed" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    
    Pop-Location
    
    # Build GPU version
    Write-Host "Building GPU version..." -ForegroundColor Blue
    $env:CGO_ENABLED = "1"
    $env:CC = "cl"
    $env:CXX = "cl"
    $env:CGO_CFLAGS = "-I."
    $env:CGO_LDFLAGS = "-L."
    
    # Initialize Visual Studio environment for Go build
    cmd /c "`"$vsPath`" && set CGO_ENABLED=1 && set CC=cl && set CXX=cl && go build -tags cuda -ldflags `"-s -w`" -o quantum-gpu-miner-gpu.exe ."
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: GPU version built" -ForegroundColor Green
        $version = .\quantum-gpu-miner-gpu.exe --version
        Write-Host "Version: $version" -ForegroundColor Cyan
    } else {
        Write-Host "ERROR: GPU build failed" -ForegroundColor Red
        exit 1
    }
}
elseif ($Mode -eq "both") {
    # Build CPU first
    Write-Host "Building CPU version..." -ForegroundColor Blue
    $env:CGO_ENABLED = "1"
    go build -ldflags "-s -w" -o quantum-gpu-miner-cpu.exe .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: CPU version built" -ForegroundColor Green
    } else {
        Write-Host "ERROR: CPU build failed" -ForegroundColor Red
        exit 1
    }
    
    # Try GPU build
    nvcc --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Attempting GPU build..." -ForegroundColor Blue
        & .\build-simple.ps1 -Mode gpu
    } else {
        Write-Host "SKIPPING: GPU build (CUDA not available)" -ForegroundColor Yellow
    }
}
else {
    Write-Host "ERROR: Invalid mode '$Mode'. Use: cpu, gpu, or both" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "Usage examples:" -ForegroundColor Cyan
Write-Host "  .\quantum-gpu-miner-cpu.exe -node http://localhost:8545 -threads 4"
if (Test-Path "quantum-gpu-miner-gpu.exe") {
    Write-Host "  .\quantum-gpu-miner-gpu.exe -gpu -node http://localhost:8545 -threads 2"
} 