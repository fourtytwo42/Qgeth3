# Windows Quantum Miner Build Guide

Complete guide for building Q Coin quantum-miner from source on Windows systems with GPU/CPU acceleration support.

## 📋 Prerequisites

### System Requirements
- **OS:** Windows 10 version 1909+ or Windows 11
- **CPU:** 4+ cores (8+ recommended for competitive mining)
- **RAM:** 8GB minimum (16GB+ recommended for GPU mining)
- **Storage:** 5GB free space for tools, source code, and build artifacts
- **GPU:** NVIDIA GPU with CUDA support (optional, for maximum performance)

### Required Software
- **Go 1.21+** (mandatory for building)
- **Python 3.8+** (for quantum algorithms and GPU acceleration)
- **Git for Windows** (for source code management)
- **Visual Studio 2022 Build Tools** (C++ compiler for CGO)
- **PowerShell 5.1+** (for build scripts)

## 🛠️ Installing Dependencies

### Go Installation
```powershell
# Method 1: Download installer from https://golang.org/dl/
# Method 2: Using package managers

# Using winget
winget install GoLang.Go

# Using Chocolatey
choco install golang

# Verify installation
go version  # Should show 1.21 or later
```

### Python Installation
```powershell
# Method 1: Download from https://python.org (ensure "Add to PATH" is checked)
# Method 2: Using package managers

# Using winget
winget install Python.Python.3.11

# Using Chocolatey
choco install python

# Verify installation
python --version  # Should show 3.8 or later
pip --version
```

### Python Quantum Dependencies
```powershell
# Core quantum computing libraries
pip install qiskit qiskit-aer numpy scipy

# For GPU acceleration (NVIDIA GPUs)
pip install cupy-cuda11x  # For CUDA 11.x
# pip install cupy-cuda12x  # For CUDA 12.x

# Additional optimization libraries
pip install numba

# Verify quantum libraries
python -c "import qiskit; print('Qiskit OK')"
python -c "from qiskit_aer import AerSimulator; print('Aer OK')"
```

### Visual Studio Build Tools
```powershell
# Download Visual Studio 2022 Build Tools
# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Required components:
# ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)
# ✅ Windows 10/11 SDK (latest version)
# ✅ CMake tools for Visual Studio

# Verify installation
where cl  # Should show Visual Studio compiler path
```

### NVIDIA GPU Setup (Optional)
```powershell
# Install NVIDIA drivers from https://www.nvidia.com/drivers/

# Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
# Choose version that matches your cupy installation

# Verify CUDA installation
nvcc --version
nvidia-smi

# Test GPU quantum acceleration
python -c "import cupy; print('GPU OK' if cupy.cuda.is_available() else 'No GPU')"
```

## 📥 Getting the Source Code

### Clone Repository
```powershell
# Open PowerShell and clone repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Verify source structure
Get-ChildItem
# Should see: quantum-geth/, scripts/, configs/, etc.

# Check miner source location
Get-ChildItem quantum-geth\tools\solver\
# Should see: main.go, *.go files
```

### Source Code Structure
```
Qgeth3\
├── quantum-geth\
│   └── tools\
│       └── solver\           # Quantum miner source code
│           ├── main.go       # Main miner executable
│           ├── quantum.go    # Quantum algorithm implementation
│           ├── gpu.go        # GPU acceleration code
│           └── ...
├── scripts\windows\         # Windows build scripts
└── ...
```

## 🔨 Building Quantum Miner

### Using Build Script (Recommended)
```powershell
# Set execution policy if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Build miner with automatic GPU detection
.\scripts\windows\build-release.ps1

# Expected outputs based on capabilities:
# ✅ Quantum-Miner built successfully: .\quantum-miner.exe (CuPy-GPU)  # Python GPU
# ✅ Quantum-Miner built successfully: .\quantum-miner.exe (CPU)      # CPU fallback

# Check release directory
Get-ChildItem releases\quantum-miner-*\
```

### Manual Build Process
```powershell
# Navigate to miner source
cd quantum-geth\tools\solver

# Set build environment for Python integration
$env:CGO_ENABLED = "1"  # Required for Python integration
$env:GOOS = "windows"
$env:GOARCH = "amd64"

# Set up Visual Studio environment
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvarsPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
cmd /c """$vcvarsPath"" && set" | ForEach-Object {
    if ($_ -match "=") {
        $var = $_.split("=")
        Set-Item -Path "env:$($var[0])" -Value $var[1]
    }
}

# Set build metadata
$BUILD_TIME = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd_HH:mm:ss")
$GIT_COMMIT = git rev-parse HEAD

# Build with quantum optimizations
go build `
  -ldflags="-s -w -X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME" `
  -o ..\..\..\quantum-miner.exe `
  .

# Return to root directory
cd ..\..\..

# Verify build
Get-ChildItem quantum-miner.exe
```

### GPU Detection and Optimization
The build system automatically detects available acceleration:

```powershell
# GPU detection during build
Write-Host "=== GPU Detection ==="

# Check for NVIDIA CUDA
try {
    nvcc --version | Out-Null
    nvidia-smi | Out-Null
    Write-Host "✅ CUDA detected - Native GPU acceleration available"
    $BUILD_TAG = "cuda"
} catch {
    Write-Host "⚠️  CUDA not available"
}

# Check for CuPy GPU support
try {
    python -c "import cupy; cupy.cuda.is_available()" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ CuPy GPU detected - Python GPU acceleration available"
        $BUILD_TAG = "cupy_gpu"
    }
} catch {
    Write-Host "⚠️  CuPy GPU not available"
}

# Fallback to CPU
if (-not $BUILD_TAG) {
    Write-Host "⚠️  No GPU detected - CPU mode only"
    $BUILD_TAG = "cpu"
}
```

## ⚙️ Build Configuration

### GPU Acceleration Modes

#### CuPy GPU Mode (Best Performance on Windows)
```powershell
# Requirements: Python + cupy-cuda11x/12x + NVIDIA GPU
# Expected performance: ~2.0-4.0 puzzles/sec (RTX 3080)

# Install CuPy for your CUDA version
pip install cupy-cuda11x  # For CUDA 11.x

# Verify GPU support
python -c "import cupy; print('GPU available' if cupy.cuda.is_available() else 'No GPU')"

# Build with CuPy GPU support
cd quantum-geth\tools\solver
go build -tags cupy_gpu -o ..\..\..\quantum-miner.exe .
```

#### CPU Mode (Fallback)
```powershell
# Requirements: Python + qiskit
# Expected performance: ~0.1-0.4 puzzles/sec (8-core CPU)

# Build CPU-only version
cd quantum-geth\tools\solver
go build -tags cpu -o ..\..\..\quantum-miner.exe .
```

### Build Tags and Features
```powershell
# Available build tags
-tags cupy_gpu      # Python GPU acceleration via CuPy
-tags qiskit_cpu    # Python CPU acceleration via Qiskit
-tags cpu           # Pure Go CPU implementation
-tags debug         # Debug logging enabled
-tags profile       # Performance profiling enabled

# Combined tags example
go build -tags "cupy_gpu,debug" -o quantum-miner.exe .
```

### Build Script Details
The `build-release.ps1` script performs comprehensive builds:

1. **Environment Detection:**
   - Detects Visual Studio Build Tools
   - Checks Python and quantum library availability
   - Tests GPU acceleration capabilities

2. **Miner Build:**
   - Sets `CGO_ENABLED=1` for Python integration
   - Configures proper compiler environment
   - Builds with GPU support if available

3. **Release Package:**
   - Creates timestamped release directory
   - Includes miner executable and launchers
   - Adds comprehensive README with usage instructions

## ✅ Verification

### Test Build
```powershell
# Check binary exists
Test-Path quantum-miner.exe
Get-ChildItem quantum-miner.exe

# Test basic functionality
.\quantum-miner.exe --help

# Expected output should include:
# Usage: quantum-miner [options]
# Options:
#   --gpu              Enable GPU acceleration
#   --threads N        Number of CPU threads
#   --coinbase ADDR    Mining reward address
#   --testnet          Use testnet
```

### GPU Acceleration Test
```powershell
# Test GPU capabilities
.\quantum-miner.exe --gpu --test --verbose

# Expected outputs based on build:
# ✅ CuPy GPU detected: GeForce RTX 3080 (10240 MB)
# ✅ Python GPU acceleration enabled
# ⚡ GPU Batch Complete: 512 puzzles in 0.256s (2000.0 puzzles/sec)

# OR for CPU fallback:
# ⚠️  No GPU acceleration available, using CPU
# ⚡ CPU Batch Complete: 64 puzzles in 0.160s (400.0 puzzles/sec)
```

### Performance Benchmarking
```powershell
# Run mining benchmark
.\quantum-miner.exe --benchmark --duration 30s

# CPU benchmark
.\quantum-miner.exe --cpu --threads $env:NUMBER_OF_PROCESSORS --benchmark --duration 30s

# GPU benchmark (if available)
.\quantum-miner.exe --gpu --benchmark --duration 30s

# Expected benchmark results:
# Platform        | Puzzles/sec | Power Usage
# ----------------|-------------|------------
# RTX 4090 (CuPy) | 3.5-5.0    | ~350W
# RTX 3080 (CuPy) | 2.0-3.5    | ~320W
# RTX 2080 (CuPy) | 1.5-2.5    | ~250W
# 16-core CPU     | 0.6-1.0    | ~150W
# 8-core CPU      | 0.2-0.5    | ~95W
```

### Quantum Algorithm Verification
```powershell
# Test quantum computation correctness
.\quantum-miner.exe --verify --test-vectors

# This runs the miner against known test vectors to ensure:
# - Quantum algorithm implementation is correct
# - Hash computations match expected values
# - GPU/CPU results are identical
# - No arithmetic errors in acceleration code
```

## 📂 Build Artifacts

### Generated Files
After successful build, you'll have:

```powershell
# Main executable
quantum-miner.exe           # Main miner binary (8-20MB typically)

# Release package (if using build-release.ps1)
releases\
└── quantum-miner-[timestamp]\
    ├── quantum-miner.exe   # Main binary
    ├── start-miner.ps1     # PowerShell launcher
    ├── start-miner.bat     # Batch launcher
    ├── README.md           # Usage documentation
    └── pkg\                # Supporting files
```

### Installation
```powershell
# Add to system PATH (optional, requires admin)
# Copy to a directory in PATH or add current directory to PATH

# Create Start Menu shortcut
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Q Coin Miner.lnk")
$Shortcut.TargetPath = (Resolve-Path ".\quantum-miner.exe").Path
$Shortcut.Arguments = "--gui"  # If GUI mode available
$Shortcut.Save()
```

## 🔧 Advanced Build Options

### Debug and Profiling Builds
```powershell
# Debug build with symbols
go build -gcflags="-N -l" -tags debug -o quantum-miner.debug.exe .\cmd\miner

# Profiling build
go build -tags profile -o quantum-miner.profile.exe .\cmd\miner

# Race detection build (slower but catches concurrency issues)
go build -race -o quantum-miner.race.exe .\cmd\miner
```

### Cross-Compilation
```powershell
# Build for different architectures
$env:GOARCH = "amd64"    # x86_64 (default)
# $env:GOARCH = "386"    # x86_32
# $env:GOARCH = "arm64"  # ARM64 (Windows 11 ARM)

go build -o quantum-miner-$env:GOARCH.exe .
```

### Custom Python Configuration
```powershell
# Use specific Python installation
$env:PYTHON_HOME = "C:\Python311"
$env:PYTHONPATH = "$env:PYTHON_HOME\Lib\site-packages"

# Build with custom Python path
go build -ldflags="-X main.pythonPath=$env:PYTHONPATH" -o quantum-miner.exe .
```

### Memory and Performance Tuning
```powershell
# For systems with limited RAM
$env:GOMEMLIMIT = "4GiB"
go build -o quantum-miner.exe .

# For high-performance systems
$env:GOMAXPROCS = $env:NUMBER_OF_PROCESSORS
go build -p $env:GOMAXPROCS -o quantum-miner.exe .
```

## 🚀 Post-Build Optimization

### Windows System Tuning
```powershell
# Set high performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable Windows power throttling for mining
Get-Process quantum-miner | ForEach-Object { 
    $_.PriorityClass = "High"
}

# Set GPU performance mode (NVIDIA)
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -pl 300  # Set power limit
```

### Windows Defender Exclusions
```powershell
# Add exclusions to prevent interference
Add-MpPreference -ExclusionPath (Get-Location).Path
Add-MpPreference -ExclusionProcess "quantum-miner.exe"
Add-MpPreference -ExclusionProcess "python.exe"

# Add CUDA temp directories
Add-MpPreference -ExclusionPath "$env:TEMP\cuda-*"
Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\Temp\cuda-*"
```

### Environment Configuration
```powershell
# Create mining-optimized environment script
@'
@echo off
REM Q Coin Mining Environment Setup

set CGO_ENABLED=1
set GOMEMLIMIT=8GiB
set GOMAXPROCS=%NUMBER_OF_PROCESSORS%

REM Python optimization
set PYTHONOPTIMIZE=2
set PYTHONHASHSEED=1

REM GPU optimization
set CUDA_CACHE_PATH=%TEMP%\cuda-cache
set CUDA_CACHE_MAXSIZE=2147483648

echo Mining environment configured!
echo GPU: %GPU_NAME%
echo CPU: %NUMBER_OF_PROCESSORS% cores
echo RAM: Available memory check...

'@ | Out-File -Encoding ASCII setup-mining-env.bat
```

## 📈 Performance Analysis

### Mining Performance Monitoring
```powershell
# Monitor miner performance
.\quantum-miner.exe --stats --interval 10s

# System resource monitoring while mining
while ($true) {
    Clear-Host
    Write-Host "=== CPU ==="
    Get-Counter "\Processor(_Total)\% Processor Time" | Select-Object -ExpandProperty CounterSamples
    
    Write-Host "=== GPU ==="
    try { nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader } catch {}
    
    Write-Host "=== Memory ==="
    Get-Counter "\Memory\Available MBytes" | Select-Object -ExpandProperty CounterSamples
    
    Start-Sleep 5
}
```

### Optimization Recommendations
```powershell
# Optimal configuration per hardware
# Single RTX 4090: --gpu (no CPU threads needed)
# RTX 3080 + 12-core CPU: --gpu --cpu-threads 4
# CPU-only 16 cores: --cpu-threads 14 (leave 2 for system)
# CPU-only 8 cores: --cpu-threads 6

# Memory usage optimization
# 8GB RAM: --batch-size 256
# 16GB RAM: --batch-size 512 (default)
# 32GB+ RAM: --batch-size 1024
```

## 🎯 Windows-Specific Features

### Windows Service Integration
```powershell
# The built quantum-miner.exe can be run as a Windows service
# See the deployment guide for service installation with NSSM

# Test service compatibility
.\quantum-miner.exe --service --help
```

### Registry Configuration
```powershell
# Optional: Add miner to Windows startup (for development)
$registryPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$name = "QCoinMiner"
$value = "`"$(Resolve-Path quantum-miner.exe)`" --background"

# Set-ItemProperty -Path $registryPath -Name $name -Value $value
```

### Event Log Integration
```powershell
# The miner can log to Windows Event Log
# Enable event logging
.\quantum-miner.exe --enable-event-log

# View miner events
Get-WinEvent -LogName Application | Where-Object {$_.ProviderName -eq "QCoinMiner"}
```

## 🔍 Troubleshooting

For build issues, see the [Windows Quantum Miner Build Troubleshooting Guide](troubleshooting-windows-build-quantum-miner.md).

Common quick fixes:
```powershell
# Update Python dependencies
pip install --upgrade qiskit qiskit-aer numpy cupy-cuda11x

# Clear caches and rebuild
go clean -cache
go clean -modcache
.\scripts\windows\build-release.ps1

# Test GPU separately
python -c "import cupy; print('GPU OK' if cupy.cuda.is_available() else 'GPU Failed')"
```

## ✅ Build Checklist

### Pre-Build
- [ ] Go 1.21+ installed (`go version`)
- [ ] Python 3.8+ with quantum libraries (`python -c "import qiskit"`)
- [ ] Visual Studio Build Tools installed and working
- [ ] GPU drivers and CUDA toolkit (if using GPU)
- [ ] PowerShell execution policy allows scripts
- [ ] Source code cloned (`git status`)

### Build Process
- [ ] CGO_ENABLED=1 set for Python integration
- [ ] Visual Studio environment configured
- [ ] Build completes without errors
- [ ] Binary generated (`quantum-miner.exe` exists)
- [ ] Binary shows help when executed

### Post-Build
- [ ] Miner shows correct acceleration mode (`.\quantum-miner.exe --help`)
- [ ] GPU detection working (if available)
- [ ] Benchmark shows reasonable performance
- [ ] Test vectors pass verification
- [ ] Release package created (if using build script)

**Successfully built quantum miner is ready for competitive Q Coin mining on Windows!** ⚡ 