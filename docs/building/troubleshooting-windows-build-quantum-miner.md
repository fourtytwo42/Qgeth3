# Windows Quantum Miner Build Troubleshooting

Comprehensive troubleshooting guide for Q Coin quantum-miner build issues on Windows systems.

## 🚨 Common Build Errors

### Go and CGO Issues

#### Error: `'go' is not recognized as an internal or external command`
**Problem:** Go compiler not installed or not in PATH.

**Solutions:**
```powershell
# Check if Go is installed
Get-Command go -ErrorAction SilentlyContinue

# Install Go using winget
winget install GoLang.Go

# Install Go using Chocolatey
choco install golang

# Manual installation from https://golang.org/dl/
# Ensure Go bin directory is added to PATH

# Verify installation
go version
```

#### Error: `cgo: C compiler not found`
**Problem:** C++ compiler required for Python integration but missing.

**Solutions:**
```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
# Required components:
# - MSVC v143 - VS 2022 C++ x64/x86 build tools
# - Windows 10/11 SDK

# Alternative: Install MinGW
choco install mingw

# Verify compiler availability
where cl  # Should show Visual Studio compiler
# OR
where gcc  # Should show MinGW compiler
```

#### Error: `CGO_ENABLED=0 but CGO required for quantum miner`
**Problem:** Quantum miner requires CGO=1 for Python integration but CGO is disabled.

**Solutions:**
```powershell
# Enable CGO for miner build
$env:CGO_ENABLED = "1"

# Verify setting
Write-Host "CGO_ENABLED: $env:CGO_ENABLED"

# Build miner with CGO enabled
cd quantum-geth\tools\solver
$env:CGO_ENABLED = "1"
go build -o ..\..\..\quantum-miner.exe .

# Note: This is different from geth which requires CGO_ENABLED=0
```

#### Error: `undefined reference to Python symbols`
**Problem:** Python development libraries not found or improperly linked.

**Solutions:**
```powershell
# Ensure Python was installed with development libraries
# Reinstall Python with "Add to PATH" and "pip" options checked

# Check Python installation
python --version
python -c "import sys; print(sys.executable)"

# Find Python include and library directories
python -c "import sysconfig; print(sysconfig.get_path('include'))"
python -c "import sysconfig; print(sysconfig.get_path('stdlib'))"

# Set CGO flags manually if needed
$pythonInclude = python -c "import sysconfig; print(sysconfig.get_path('include'))"
$env:CGO_CFLAGS = "-I$pythonInclude"
$env:CGO_LDFLAGS = "-L$(Split-Path $pythonInclude)\libs -lpython311"  # Adjust version
```

### Python Integration Issues

#### Error: `ModuleNotFoundError: No module named 'qiskit'`
**Problem:** Required Python quantum libraries not installed.

**Solutions:**
```powershell
# Install quantum dependencies
pip install qiskit qiskit-aer numpy scipy

# For GPU support (NVIDIA)
pip install cupy-cuda11x  # For CUDA 11.x
# pip install cupy-cuda12x  # For CUDA 12.x

# Verify installation
python -c "import qiskit; print('Qiskit OK')"
python -c "from qiskit_aer import AerSimulator; print('Aer OK')"

# Check installed packages
pip list | findstr qiskit
```

#### Error: `ImportError: DLL load failed while importing`
**Problem:** Python DLL dependencies not found.

**Solutions:**
```powershell
# Install Microsoft Visual C++ Redistributables
# Download from: https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

# Reinstall Python packages
pip uninstall qiskit qiskit-aer cupy-cuda11x
pip install qiskit qiskit-aer cupy-cuda11x

# Check Python installation integrity
python -m pip check

# Use conda instead of pip if issues persist
conda install qiskit qiskit-aer
```

#### Error: `Failed to load Python DLL`
**Problem:** Python shared libraries not accessible.

**Solutions:**
```powershell
# Add Python directory to PATH
$pythonPath = python -c "import sys; print(sys.executable)"
$pythonDir = Split-Path $pythonPath
$env:PATH += ";$pythonDir;$pythonDir\Scripts"

# Reinstall Python with proper options
# Download from https://python.org/downloads/
# Check "Add Python to PATH" during installation

# Verify DLL availability
Get-ChildItem "$pythonDir\python*.dll"
```

### Visual Studio Build Tools Issues

#### Error: `The system cannot find the path specified` when setting up compiler
**Problem:** Visual Studio Build Tools not properly detected.

**Solutions:**
```powershell
# Check Visual Studio installation
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
} else {
    Write-Host "Visual Studio installer not found"
}

# Manual environment setup
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"  # Adjust path
$vcvarsPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (Test-Path $vcvarsPath) {
    cmd /c """$vcvarsPath"" && set" | ForEach-Object {
        if ($_ -match "=") {
            $var = $_.split("=")
            Set-Item -Path "env:$($var[0])" -Value $var[1]
        }
    }
}

# Verify compiler
where cl
cl  # Should show compiler version
```

#### Error: `'vcvars64.bat' is not recognized`
**Problem:** Visual Studio environment not properly configured.

**Solutions:**
```powershell
# Find vcvars64.bat manually
Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio\" -Recurse -Name "vcvars64.bat"

# Use Developer PowerShell
# Start menu -> "Developer PowerShell for VS 2022"

# Alternative: Use MSBuild from PATH
$msbuildPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin"
$env:PATH += ";$msbuildPath"
```

### GPU and CUDA Issues

#### Error: `CUDA not found` during build
**Problem:** CUDA toolkit not installed or not detected.

**Solutions:**
```powershell
# Check if NVIDIA GPU is present
Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}

# Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
# Choose version compatible with your GPU and cupy version

# Verify CUDA installation
nvcc --version
nvidia-smi

# Add CUDA to PATH if not automatic
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
$env:PATH += ";$cudaPath"
```

#### Error: `ImportError: No module named 'cupy'`
**Problem:** CuPy not installed for GPU acceleration.

**Solutions:**
```powershell
# Check CUDA version first
nvcc --version

# Install CuPy for your CUDA version
pip install cupy-cuda11x  # For CUDA 11.x
# pip install cupy-cuda12x  # For CUDA 12.x

# If installation fails, try pre-compiled wheels
pip install -f https://pip.cupy.dev/aarch64 cupy-cuda11x

# Verify CuPy installation
python -c "import cupy; print('CuPy version:', cupy.__version__)"
python -c "import cupy; print('GPU available:', cupy.cuda.is_available())"
```

#### Error: `cupy.cuda.runtime.CUDARuntimeError: cudaErrorInsufficientDriver`
**Problem:** NVIDIA drivers outdated or incompatible.

**Solutions:**
```powershell
# Update NVIDIA drivers
# Download latest from https://www.nvidia.com/drivers/

# Check driver version
nvidia-smi

# Ensure driver supports your CUDA version
# CUDA 11.8 requires driver 520.61.05+
# CUDA 12.0 requires driver 525.60.13+

# Restart system after driver update
```

### Build Script Issues

#### Error: `execution of scripts is disabled on this system`
**Problem:** PowerShell execution policy prevents running build scripts.

**Solutions:**
```powershell
# Check current execution policy
Get-ExecutionPolicy

# Set execution policy for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Temporary bypass for one session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Alternative: Run specific script with bypass
PowerShell -ExecutionPolicy Bypass -File .\scripts\windows\build-release.ps1
```

#### Error: `quantum-geth\tools\solver directory not found`
**Problem:** Incorrect source code structure or wrong directory.

**Solutions:**
```powershell
# Verify current directory
Get-Location
Get-ChildItem

# Should be in Qgeth3 root directory
Set-Location "C:\path\to\Qgeth3"

# Verify source structure
Test-Path "quantum-geth\tools\solver"
Get-ChildItem "quantum-geth\tools\solver\" -Name "*.go"

# If missing, check repository integrity
git status
git pull origin main
```

### Memory and Performance Issues

#### Error: `Out of memory` or system hanging during build
**Problem:** Insufficient memory for compilation.

**Solutions:**
```powershell
# Check available memory
Get-WmiObject -Class Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory

# Close unnecessary applications
# Use Task Manager to identify memory-heavy processes

# Increase virtual memory (page file)
# Control Panel -> System -> Advanced -> Performance Settings -> Advanced -> Virtual Memory

# Limit Go build parallelism
$env:GOMAXPROCS = "2"
go build -p 2 -o quantum-miner.exe .

# Use build cache
$env:GOCACHE = "$env:LOCALAPPDATA\go-build"
New-Item -ItemType Directory -Force -Path $env:GOCACHE
```

#### Error: Build succeeds but miner performance is poor
**Problem:** Built without optimal GPU support or wrong configuration.

**Solutions:**
```powershell
# Check what acceleration was built
.\quantum-miner.exe --version
.\quantum-miner.exe --help | Select-String -Pattern "gpu"

# Verify GPU support
nvidia-smi
python -c "import cupy; print('GPU OK' if cupy.cuda.is_available() else 'No GPU')"

# Rebuild with explicit GPU tags
cd quantum-geth\tools\solver
go build -tags "cupy_gpu,debug" -o ..\..\..\quantum-miner.exe .

# Test GPU performance
.\quantum-miner.exe --gpu --benchmark --duration 10s
```

## 🔧 Environment Issues

### PowerShell and Execution Policy

#### Error: `UnauthorizedAccess` when running scripts
**Problem:** Insufficient permissions or security restrictions.

**Solutions:**
```powershell
# Run PowerShell as Administrator
Start-Process PowerShell -Verb RunAs

# Check file permissions
Get-Acl .\scripts\windows\build-release.ps1

# Unblock script if downloaded from internet
Unblock-File .\scripts\windows\build-release.ps1

# Grant execution permissions
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### PATH and Environment Variables

#### Error: Environment variables not persistent
**Problem:** Variables not saved permanently.

**Solutions:**
```powershell
# Set permanent user environment variable
[System.Environment]::SetEnvironmentVariable("CGO_ENABLED", "1", "User")

# Set permanent system environment variable (requires admin)
[System.Environment]::SetEnvironmentVariable("CGO_ENABLED", "1", "Machine")

# Add to PATH permanently
$goPath = "C:\Program Files\Go\bin"
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$goPath*") {
    [System.Environment]::SetEnvironmentVariable("PATH", "$currentPath;$goPath", "User")
}

# Refresh environment in current session
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
```

### Windows Defender and Antivirus

#### Error: Build interrupted or files deleted by antivirus
**Problem:** Windows Defender or antivirus interfering with build.

**Solutions:**
```powershell
# Add exclusions to Windows Defender
Add-MpPreference -ExclusionPath (Get-Location).Path
Add-MpPreference -ExclusionProcess "go.exe"
Add-MpPreference -ExclusionProcess "quantum-miner.exe"
Add-MpPreference -ExclusionProcess "python.exe"

# Add temporary directories
Add-MpPreference -ExclusionPath "$env:TEMP"
Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\Temp"

# Temporary disable real-time protection (requires admin)
Set-MpPreference -DisableRealtimeMonitoring $true
# Remember to re-enable: Set-MpPreference -DisableRealtimeMonitoring $false

# Check Windows Defender history
Get-MpThreatDetection | Where-Object {$_.Resources -like "*quantum*"}
```

## 🐛 Advanced Debugging

### Build Debugging

```powershell
# Enable verbose Go build output
go build -v -x -o quantum-miner.exe .\cmd\miner

# Debug CGO compilation
$env:CGO_ENABLED = "1"
go build -work -x -o quantum-miner.exe .

# Check CGO configuration
go env CGO_CFLAGS
go env CGO_LDFLAGS

# Manual CGO test
@'
package main

/*
#include <Python.h>
*/
import "C"

func main() {
    C.Py_Initialize()
    C.Py_Finalize()
}
'@ | Out-File -Encoding UTF8 test_cgo.go

go run test_cgo.go  # Should compile without errors
Remove-Item test_cgo.go
```

### Runtime Debugging

```powershell
# Debug miner startup
.\quantum-miner.exe --debug --verbose --dry-run

# Check Python integration
.\quantum-miner.exe --test-python

# Debug GPU detection
.\quantum-miner.exe --debug-gpu

# Check dependencies
dumpbin /dependents quantum-miner.exe

# Monitor file access
# Use Process Monitor (ProcMon) from Sysinternals
```

### Performance Analysis

```powershell
# Profile miner build
go build -cpuprofile=miner.prof -o quantum-miner.exe .
go tool pprof miner.prof

# Memory profiling
go build -memprofile=mem.prof -o quantum-miner.exe .
go tool pprof mem.prof

# Benchmark different builds
Measure-Command { go build -tags cpu -o quantum-miner-cpu.exe . }
Measure-Command { go build -tags cupy_gpu -o quantum-miner-gpu.exe . }

# Compare performance
.\quantum-miner-cpu.exe --benchmark --duration 30s
.\quantum-miner-gpu.exe --benchmark --duration 30s
```

## 🚀 Performance Troubleshooting

### GPU Performance Issues

```powershell
# Check GPU utilization during mining
while ($true) {
    Clear-Host
    Write-Host "=== GPU Status ==="
    try {
        nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
    } catch {
        Write-Host "nvidia-smi not available"
    }
    Start-Sleep 2
}

# Monitor GPU performance
nvidia-smi -l 1

# Check thermal throttling
nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics,temperature.gpu --format=csv -l 1

# Optimize GPU settings
nvidia-smi -pm 1  # Persistence mode
nvidia-smi -pl 300  # Power limit (adjust for your card)
```

### CPU Performance Issues

```powershell
# Monitor CPU usage
Get-Counter "\Processor(_Total)\% Processor Time" -Continuous

# Check CPU frequency
Get-WmiObject Win32_Processor | Select-Object CurrentClockSpeed, MaxClockSpeed

# Set high performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Check for thermal throttling
Get-WmiObject -Class Win32_PerfRawData_Counters_ThermalZoneInformation
```

### Memory Issues

```powershell
# Monitor memory usage during mining
while ($true) {
    $mem = Get-Counter "\Memory\Available MBytes"
    $proc = Get-Process quantum-miner -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Available Memory: $($mem.CounterSamples[0].CookedValue) MB"
        Write-Host "Miner Memory: $([math]::Round($proc.WorkingSet64/1MB, 2)) MB"
    }
    Start-Sleep 5
}

# Check for memory leaks
# Use Visual Studio Diagnostic Tools or Application Verifier

# Optimize for low memory
$env:GOMEMLIMIT = "4GiB"
.\quantum-miner.exe --cpu --threads 2 --batch-size 128
```

## ✅ Build Health Check

### Pre-Build Health Check

```powershell
# Quantum Miner Windows Build Health Check
Write-Host "=== Quantum Miner Windows Build Health Check ==="

# Check Go installation
try {
    $goVersion = go version
    Write-Host "✅ Go installed: $goVersion"
    if ($goVersion -notmatch "go1\.2[1-9]") {
        Write-Host "❌ Go version too old, need 1.21+"
        exit 1
    }
} catch {
    Write-Host "❌ Go not found"
    exit 1
}

# Check CGO support
if ($env:CGO_ENABLED -eq "1") {
    Write-Host "✅ CGO enabled for Python integration"
} else {
    Write-Host "⚠️  CGO not enabled, setting now"
    $env:CGO_ENABLED = "1"
}

# Check Python and dependencies
try {
    $pythonVersion = python --version
    Write-Host "✅ Python available: $pythonVersion"
    
    python -c "import qiskit" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Qiskit installed"
    } else {
        Write-Host "❌ Qiskit not installed"
        exit 1
    }
} catch {
    Write-Host "❌ Python not found"
    exit 1
}

# Check Visual Studio Build Tools
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsPath) {
        Write-Host "✅ Visual Studio Build Tools found"
    } else {
        Write-Host "⚠️  Visual Studio Build Tools not found"
    }
} else {
    Write-Host "⚠️  VS installer not found"
}

# Check GPU support (optional)
$gpus = Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}
if ($gpus) {
    Write-Host "✅ NVIDIA GPU detected: $($gpus[0].Name)"
    try {
        python -c "import cupy; cupy.cuda.is_available()" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ CuPy GPU support available"
        } else {
            Write-Host "⚠️  CuPy GPU support not available"
        }
    } catch {
        Write-Host "⚠️  CuPy not installed"
    }
} else {
    Write-Host "⚠️  No NVIDIA GPU detected, CPU-only mode"
}

Write-Host "✅ Quantum miner build environment ready!"
```

### Post-Build Verification

```powershell
# Quantum Miner Build Verification
Write-Host "=== Quantum Miner Build Verification ==="

if (Test-Path "quantum-miner.exe") {
    Write-Host "✅ Binary exists"
    
    # Check binary size
    $size = (Get-ChildItem quantum-miner.exe).Length / 1MB
    Write-Host "📦 Binary size: $([math]::Round($size, 1))MB"
    
    if ($size -lt 5 -or $size -gt 50) {
        Write-Host "⚠️  Unusual binary size"
    }
    
    # Test basic functionality
    try {
        $help = .\quantum-miner.exe --help 2>&1
        Write-Host "✅ Binary executes successfully"
    } catch {
        Write-Host "❌ Binary execution failed"
        exit 1
    }
    
    # Check quantum features
    if ($help -match "gpu|quantum") {
        Write-Host "✅ Quantum features detected"
    } else {
        Write-Host "⚠️  Quantum features not found"
    }
    
    # Test GPU support if available
    try {
        $gpuTest = .\quantum-miner.exe --gpu --test 2>&1
        if ($gpuTest -match "GPU.*OK|acceleration.*enabled") {
            Write-Host "✅ GPU acceleration working"
        } else {
            Write-Host "⚠️  GPU acceleration not available"
        }
    } catch {
        Write-Host "⚠️  GPU test failed"
    }
    
} else {
    Write-Host "❌ Binary not found"
    exit 1
}

Write-Host "✅ Quantum miner build verification complete!"
```

## 📞 Getting Help

### Community Support
- **GitHub Issues:** [Report build issues](https://github.com/fourtytwo42/Qgeth3/issues)
- **Documentation:** Check [build guide](windows-build-quantum-miner.md)

### Self-Help Resources
- Run the health check scripts above
- Review [mining troubleshooting](../mining/troubleshooting-windows-mining.md)
- Check [general Windows troubleshooting](../node-operation/troubleshooting-windows-geth.md)

### Debug Information Collection

```powershell
# Create comprehensive debug log
$debugLog = "quantum-miner-debug.log"
@"
=== Debug Information ===
Date: $(Get-Date)
OS: $((Get-WmiObject Win32_OperatingSystem).Caption)
User: $env:USERNAME
PowerShell: $($PSVersionTable.PSVersion)
Working Directory: $(Get-Location)

=== Environment ===
"@ | Out-File $debugLog

go env | Out-File $debugLog -Append

@"

=== Python Environment ===
"@ | Out-File $debugLog -Append

python --version 2>&1 | Out-File $debugLog -Append
pip list | Out-File $debugLog -Append

@"

=== GPU Information ===
"@ | Out-File $debugLog -Append

try {
    nvidia-smi 2>&1 | Out-File $debugLog -Append
} catch {
    "No NVIDIA GPU" | Out-File $debugLog -Append
}

@"

=== Build Attempt ===
"@ | Out-File $debugLog -Append

try {
    Set-Location quantum-geth\tools\solver
    $env:CGO_ENABLED = "1"
    go build -v -x -o ..\..\..\quantum-miner.exe . 2>&1 | Out-File $debugLog -Append
} catch {
    $_.Exception.Message | Out-File $debugLog -Append
}

Write-Host "Debug log created: $debugLog"
Write-Host "Share this file when seeking help"
```

**Most Windows quantum miner build issues are resolved by ensuring proper Python dependencies, Visual Studio Build Tools, and CGO configuration!** ⚡ 