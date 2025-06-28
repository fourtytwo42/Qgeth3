# Windows Q Geth Build Guide

Complete guide for building Q Coin quantum-geth from source on Windows systems.

## 📋 Prerequisites

### System Requirements
- **OS:** Windows 10 version 1909+ or Windows 11
- **CPU:** 2+ cores (4+ recommended for faster builds)
- **RAM:** 4GB minimum (8GB+ recommended for builds)
- **Storage:** 10GB free space for tools, source code, and build artifacts
- **Network:** Internet connection for downloading dependencies

### Required Software
- **Go 1.21+** (mandatory for building)
- **Git for Windows** (for source code management)
- **Visual Studio 2022 Build Tools** or **MinGW-w64** (C++ compiler)
- **PowerShell 5.1+** (for build scripts)

## 🛠️ Installing Dependencies

### Go Installation
```powershell
# Method 1: Download installer
# Visit https://golang.org/dl/ and download Windows installer

# Method 2: Using Chocolatey (if available)
choco install golang

# Method 3: Using winget
winget install GoLang.Go

# Verify installation
go version  # Should show 1.21 or later
```

### Git Installation
```powershell
# Method 1: Download from https://git-scm.com/download/win

# Method 2: Using Chocolatey
choco install git

# Method 3: Using winget
winget install Git.Git

# Verify installation
git --version
```

### Visual Studio Build Tools
```powershell
# Download Visual Studio 2022 Build Tools
# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Required components:
# ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)
# ✅ Windows 10/11 SDK (latest version)
# ✅ CMake tools for Visual Studio (optional but recommended)

# Alternative: Visual Studio Community 2022 (full IDE)
# https://visualstudio.microsoft.com/vs/community/
```

### MinGW-w64 Alternative
```powershell
# If not using Visual Studio, install MinGW-w64
# Download from: https://www.mingw-w64.org/downloads/

# Or using MSYS2:
# Download from: https://www.msys2.org/
# Then run in MSYS2 shell:
# pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make
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

# Check quantum-geth source
Get-ChildItem quantum-geth\
# Should see: cmd/, core/, consensus/, etc.
```

### Source Code Structure
```
Qgeth3\
├── quantum-geth\           # Main geth source code
│   ├── cmd\geth\          # Geth main executable
│   ├── core\              # Blockchain core logic
│   ├── consensus\qmpow\   # Quantum consensus implementation
│   ├── eth\               # Ethereum protocol implementation
│   └── ...
├── scripts\windows\       # Windows build scripts
├── configs\               # Network configurations
└── ...
```

## 🔨 Building Q Geth

### Using Build Script (Recommended)
```powershell
# Set execution policy (if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Build geth with automatic configuration
.\scripts\windows\build-release.ps1

# Expected output:
# Building quantum-geth...
# ENFORCING: CGO_ENABLED=0 for geth build (quantum field compatibility)
# quantum-geth built successfully (CGO_ENABLED=0)
# Created release: .\releases\quantum-geth-[timestamp]
```

### Manual Build Process
```powershell
# Navigate to quantum-geth source
cd quantum-geth

# Set build environment for quantum compatibility
$env:CGO_ENABLED = "0"
$env:GOOS = "windows"
$env:GOARCH = "amd64"

# Set build metadata
$BUILD_TIME = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd_HH:mm:ss")
$GIT_COMMIT = git rev-parse HEAD

# Build geth with quantum consensus
go build `
  -ldflags="-s -w -X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME" `
  -o ..\geth.exe `
  .\cmd\geth

# Return to root directory
cd ..

# Verify build
Get-ChildItem geth.exe
file geth.exe  # Should show: PE32+ executable
```

### Build Script Details
The `build-release.ps1` script performs:

1. **Environment Detection:**
   - Detects Visual Studio Build Tools
   - Sets up proper compiler environment
   - Configures Go build environment

2. **Geth Build:**
   - Sets `CGO_ENABLED=0` for quantum compatibility
   - Builds with optimization flags
   - Creates timestamped release directory

3. **Release Package:**
   - Copies geth.exe to release folder
   - Includes start scripts (PowerShell and batch)
   - Adds comprehensive README documentation

## ⚙️ Build Configuration

### Quantum Field Compatibility
Q Geth requires `CGO_ENABLED=0` for quantum field serialization:

```powershell
# Why CGO_ENABLED=0 is required:
# - Ensures consistent quantum field marshaling
# - Prevents C library dependencies
# - Enables static binary compilation
# - Guarantees cross-platform compatibility

$env:CGO_ENABLED = "0"  # Mandatory for Q Geth builds
```

### Build Environment Setup
```powershell
# Automatic Visual Studio detection
function Find-VisualStudio {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        return $vsPath
    }
    return $null
}

# Set up build environment
$vsPath = Find-VisualStudio
if ($vsPath) {
    $vcvarsPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    cmd /c """$vcvarsPath"" && set" | ForEach-Object {
        if ($_ -match "=") {
            $var = $_.split("=")
            Set-Item -Path "env:$($var[0])" -Value $var[1]
        }
    }
}
```

### Cross-Compilation
```powershell
# Build for different architectures
$env:GOOS = "windows"
$env:GOARCH = "amd64"    # x86_64 (default)
# $env:GOARCH = "386"    # x86_32
# $env:GOARCH = "arm64"  # ARM64 (Windows 11 ARM)

go build -o geth.exe .\cmd\geth
```

## ✅ Verification

### Test Build
```powershell
# Check binary exists
Get-ChildItem geth.exe
Test-Path geth.exe

# Test basic functionality
.\geth.exe version

# Expected output similar to:
# Geth
# Version: 1.13.5-stable
# Git Commit: 916d6a44c9b9b89efdc31b62a78d26a6b84bb9c1
# Git Commit Date: 20231128
# Architecture: amd64
# Go Version: go1.21.5
# Operating System: windows
```

### Quantum Consensus Verification
```powershell
# Check if quantum consensus is available
.\geth.exe help | Select-String -Pattern "quantum" -CaseSensitive

# Check quantum-specific commands
.\geth.exe console --help | Select-String -Pattern "(qmpow|quantum)"

# Test initialization with quantum genesis
.\geth.exe --datadir .\test-qgeth init .\configs\genesis_quantum_testnet.json
Get-ChildItem .\test-qgeth\geth\
Remove-Item -Recurse -Force .\test-qgeth  # Cleanup test
```

### Performance Test
```powershell
# Quick startup test (with timeout)
$job = Start-Job -ScriptBlock { .\geth.exe --datadir .\test --dev console }
Start-Sleep -Seconds 10
Stop-Job $job
Remove-Job $job

# Binary information
$size = (Get-ChildItem geth.exe).Length / 1MB
Write-Host "Binary size: $([math]::Round($size, 1)) MB"

# Dependencies check (should be minimal for static build)
dumpbin /dependents geth.exe 2>$null || echo "Static binary (no dumpbin available)"
```

## 📂 Build Artifacts

### Generated Files
After successful build, you'll have:

```powershell
# Main executable
geth.exe                    # Q Geth binary (typically 15-25MB)

# Release package (if using build-release.ps1)
releases\
└── quantum-geth-[timestamp]\
    ├── geth.exe           # Main binary
    ├── start-geth.ps1     # PowerShell launcher
    ├── start-geth.bat     # Batch launcher
    └── README.md          # Usage documentation
```

### Installation
```powershell
# Add to system PATH (optional, requires admin)
# Copy to a directory in PATH or add current directory to PATH

# Create Start Menu shortcut
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Q Geth.lnk")
$Shortcut.TargetPath = (Resolve-Path ".\geth.exe").Path
$Shortcut.Save()
```

## 🔧 Build Customization

### Debug Builds
```powershell
# Build with debug symbols
go build -gcflags="-N -l" -o geth.debug.exe .\cmd\geth

# Build with race detection (slower but catches concurrency issues)
go build -race -o geth.race.exe .\cmd\geth

# Build with verbose output
go build -v -o geth.exe .\cmd\geth
```

### Optimization Levels
```powershell
# Minimal size build
go build -ldflags="-s -w" -o geth.exe .\cmd\geth

# Standard release build (recommended)
go build -ldflags="-s -w -X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME" -o geth.exe .\cmd\geth

# Debug build (larger, with symbols)
go build -o geth.exe .\cmd\geth
```

### Memory Optimization
```powershell
# For low-memory systems
$env:GOMEMLIMIT = "2GiB"
go build -o geth.exe .\cmd\geth

# Use build cache
$env:GOCACHE = "$env:LOCALAPPDATA\go-build"
go build -o geth.exe .\cmd\geth
```

## 🚀 Advanced Build Options

### Using Custom Temp Directory
```powershell
# Set custom temp directory for builds
$env:QGETH_BUILD_TEMP = "C:\Temp\qgeth-build"
New-Item -ItemType Directory -Force -Path $env:QGETH_BUILD_TEMP

.\scripts\windows\build-release.ps1

# This uses the custom temp directory for:
# - Go cache
# - Go temp files
# - Build artifacts
```

### Parallel Builds
```powershell
# Use multiple CPU cores for faster builds
$env:GOMAXPROCS = [System.Environment]::ProcessorCount
go build -p $env:GOMAXPROCS -o geth.exe .\cmd\geth
```

### Reproducible Builds
```powershell
# Ensure reproducible builds
$env:CGO_ENABLED = "0"
$env:GOOS = "windows"
$env:GOARCH = "amd64"

go build -trimpath -ldflags="-buildid=" -o geth.exe .\cmd\geth
```

## 📝 Build Environment

### Required Environment Variables
```powershell
# Mandatory for Q Geth
$env:CGO_ENABLED = "0"

# Recommended
$env:GOOS = "windows"
$env:GOARCH = "amd64"
$env:GO111MODULE = "on"

# Optional optimizations
$env:GOMAXPROCS = [System.Environment]::ProcessorCount
$env:GOCACHE = "$env:LOCALAPPDATA\go-build"
```

### PowerShell Build Script Environment
The `build-release.ps1` script automatically configures:

```powershell
# Quantum compatibility
$env:CGO_ENABLED = "0"

# Build metadata
$BUILD_TIME = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd_HH:mm:ss")
$GIT_COMMIT = git rev-parse HEAD

# Linker flags
$LDFLAGS = "-s -w -X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME"

# Release directory
$timestamp = [int][double]::Parse((Get-Date -UFormat %s))
$releaseName = "quantum-geth-$timestamp"
```

## 🎯 Windows-Specific Features

### Windows Service Integration
```powershell
# The built geth.exe can be run as a Windows service
# See the deployment guide for service installation

# Test service compatibility
.\geth.exe --help | Select-String "service"
```

### Windows Defender Exclusion
```powershell
# Recommended: Add build directory to Windows Defender exclusions
# This prevents antivirus from interfering with builds

Add-MpPreference -ExclusionPath (Get-Location).Path
Add-MpPreference -ExclusionProcess "go.exe"
Add-MpPreference -ExclusionProcess "geth.exe"
```

### PowerShell Execution Policy
```powershell
# Check current execution policy
Get-ExecutionPolicy

# Set policy for building (if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify scripts can run
Get-ExecutionPolicy -List
```

## 🔍 Troubleshooting

For build issues, see the [Windows Geth Build Troubleshooting Guide](troubleshooting-windows-build-geth.md).

Common quick fixes:
```powershell
# Update Go
winget install GoLang.Go

# Clear caches
go clean -cache
go clean -modcache

# Rebuild clean
.\scripts\windows\build-release.ps1
```

## ✅ Build Checklist

### Pre-Build
- [ ] Go 1.21+ installed and working (`go version`)
- [ ] Visual Studio Build Tools or MinGW installed
- [ ] Git installed (`git --version`)
- [ ] PowerShell execution policy allows scripts
- [ ] Source code cloned (`git status`)

### Build Process
- [ ] CGO_ENABLED=0 set for quantum compatibility
- [ ] Build completes without errors
- [ ] Binary generated (`geth.exe` exists)
- [ ] Binary shows version when executed

### Post-Build
- [ ] Quantum consensus available (`.\geth.exe help | Select-String quantum`)
- [ ] Can initialize with quantum genesis
- [ ] Binary size reasonable (15-25MB)
- [ ] Release package created (if using build script)

**Successfully built Q Geth binary is ready for quantum blockchain operations on Windows!** 🎉 