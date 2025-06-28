# Windows Q Geth Build Troubleshooting

Comprehensive troubleshooting guide for Q Coin quantum-geth build issues on Windows systems.

## 🚨 Common Build Errors

### Go Compiler Issues

#### Error: `'go' is not recognized as an internal or external command`
**Problem:** Go compiler not installed or not in PATH.

**Solutions:**
```powershell
# Check if Go is installed
Get-Command go -ErrorAction SilentlyContinue

# Check PATH
$env:PATH -split ';' | Select-String go

# Install Go using winget
winget install GoLang.Go

# Install Go using Chocolatey
choco install golang

# Manual installation
# Download from https://golang.org/dl/
# Add to PATH: C:\Program Files\Go\bin

# Verify installation
go version
```

#### Error: `go: go.mod file not found in current directory or any parent directory`
**Problem:** Running Go commands outside the quantum-geth directory.

**Solutions:**
```powershell
# Navigate to the correct directory
cd quantum-geth
Get-Location  # Should show: C:\path\to\Qgeth3\quantum-geth

# Verify go.mod exists
Get-ChildItem go.mod

# If go.mod is missing, initialize (rare case)
go mod init quantum-geth
go mod tidy
```

#### Error: `go version go1.x.x: directive requires go 1.21 or later`
**Problem:** Go version too old for quantum-geth requirements.

**Solutions:**
```powershell
# Check current Go version
go version

# Update Go using winget
winget upgrade GoLang.Go

# Update using Chocolatey
choco upgrade golang

# Manual update
# Download latest from https://golang.org/dl/
# Uninstall old version first

# Verify new version
go version
```

### Visual Studio Build Tools Issues

#### Error: `gcc: command not found` or compiler errors
**Problem:** C++ build tools not installed or not configured.

**Solutions:**
```powershell
# Check Visual Studio installation
Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio\" -Recurse -Name "*vcvars64.bat*"

# Install Visual Studio 2022 Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Required components:
# - MSVC v143 - VS 2022 C++ x64/x86 build tools
# - Windows 10/11 SDK

# Alternative: Install MinGW-w64
choco install mingw

# Set up environment manually
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

#### Error: `The system cannot find the path specified` for vcvars
**Problem:** Visual Studio Build Tools not found by build script.

**Solutions:**
```powershell
# Manually locate Visual Studio
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}

# Set environment manually if needed
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"  # Adjust path
$vcvarsPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
cmd /c """$vcvarsPath"" && set" | ForEach-Object {
    if ($_ -match "=") {
        $var = $_.split("=")
        Set-Item -Path "env:$($var[0])" -Value $var[1]
    }
}
```

### CGO Compatibility Issues

#### Error: Build fails with CGO errors despite CGO_ENABLED=0
**Problem:** CGO configuration not properly set.

**Solutions:**
```powershell
# Explicitly disable CGO
$env:CGO_ENABLED = "0"

# Verify setting
Write-Host "CGO_ENABLED: $env:CGO_ENABLED"

# Force static build
$env:CGO_ENABLED = "0"
$env:GOOS = "windows"
$env:GOARCH = "amd64"
go build -a -o geth.exe .\cmd\geth

# Check dependencies (should be minimal)
dumpbin /dependents geth.exe
```

### PowerShell Execution Issues

#### Error: `execution of scripts is disabled on this system`
**Problem:** PowerShell execution policy prevents running build scripts.

**Solutions:**
```powershell
# Check current execution policy
Get-ExecutionPolicy

# Set execution policy for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Temporary bypass for one script
PowerShell -ExecutionPolicy Bypass -File .\scripts\windows\build-release.ps1

# Alternative: Run commands directly
# Copy commands from build-release.ps1 and run manually
```

#### Error: `UnauthorizedAccess` when running scripts
**Problem:** Insufficient permissions or security restrictions.

**Solutions:**
```powershell
# Run PowerShell as Administrator
Start-Process PowerShell -Verb RunAs

# Check file permissions
Get-Acl .\scripts\windows\build-release.ps1

# Unblock script if downloaded
Unblock-File .\scripts\windows\build-release.ps1

# Run from elevated prompt
Set-Location "C:\path\to\Qgeth3"
.\scripts\windows\build-release.ps1
```

### Memory and Space Issues

#### Error: `Out of memory` or system hanging during build
**Problem:** Insufficient RAM for compilation.

**Solutions:**
```powershell
# Check available memory
Get-WmiObject -Class Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-Counter '\Memory\Available MBytes'

# Close unnecessary applications
# Increase virtual memory (page file)

# Limit Go memory usage
$env:GOMEMLIMIT = "2GiB"
go build -o geth.exe .\cmd\geth

# Build with reduced parallelism
$env:GOMAXPROCS = "2"
go build -p 2 -o geth.exe .\cmd\geth
```

#### Error: `No space left on device` equivalent on Windows
**Problem:** Insufficient disk space.

**Solutions:**
```powershell
# Check disk space
Get-PSDrive C

# Clean temporary files
Remove-Item $env:TEMP\* -Recurse -Force -ErrorAction SilentlyContinue

# Clean Go caches
go clean -cache
go clean -modcache

# Use different drive for temp
$env:QGETH_BUILD_TEMP = "D:\qgeth-build"
New-Item -ItemType Directory -Force -Path $env:QGETH_BUILD_TEMP
.\scripts\windows\build-release.ps1
```

### Network and Dependency Issues

#### Error: `fatal: unable to access 'https://github.com/...'`
**Problem:** Git/network authentication issues.

**Solutions:**
```powershell
# Check Git configuration
git config --global user.name
git config --global user.email

# Configure Git credentials
git config --global credential.helper manager-core

# Use HTTPS instead of SSH
git config --global url."https://github.com/".insteadOf git@github.com:

# Clear Git credentials and retry
git config --global --unset credential.helper
git credential-manager-core erase

# Test Git connectivity
git ls-remote https://github.com/fourtytwo42/Qgeth3.git
```

#### Error: `Get "https://proxy.golang.org/...": timeout`
**Problem:** Network timeout downloading Go modules.

**Solutions:**
```powershell
# Configure Go proxy
$env:GOPROXY = "direct"
$env:GOSUMDB = "off"

# Use alternative proxy
$env:GOPROXY = "https://goproxy.cn,direct"

# Download dependencies first
go mod download

# Build offline
go build -mod=readonly -o geth.exe .\cmd\geth

# Check corporate firewall/proxy settings
```

### Windows Defender and Antivirus Issues

#### Error: Build interrupted or files deleted by antivirus
**Problem:** Windows Defender or antivirus interfering with build.

**Solutions:**
```powershell
# Add exclusions to Windows Defender
Add-MpPreference -ExclusionPath (Get-Location).Path
Add-MpPreference -ExclusionProcess "go.exe"
Add-MpPreference -ExclusionProcess "geth.exe"
Add-MpPreference -ExclusionPath "$env:GOPATH"
Add-MpPreference -ExclusionPath "$env:GOCACHE"

# Temporary disable real-time protection (requires admin)
Set-MpPreference -DisableRealtimeMonitoring $true
# Remember to re-enable: Set-MpPreference -DisableRealtimeMonitoring $false

# Check Windows Defender history
Get-MpThreatDetection | Where-Object {$_.Resources -like "*geth*"}
```

### Build Script Issues

#### Error: `quantum-geth directory not found!`
**Problem:** Running script from wrong directory.

**Solutions:**
```powershell
# Check current directory
Get-Location
Get-ChildItem

# Should be in Qgeth3 root directory
Set-Location "C:\path\to\Qgeth3"

# Verify quantum-geth exists
Test-Path "quantum-geth"
Get-ChildItem quantum-geth

# Run from correct location
.\scripts\windows\build-release.ps1
```

#### Error: `The term 'build-release.ps1' is not recognized`
**Problem:** PowerShell cannot find the script.

**Solutions:**
```powershell
# Use full path
C:\path\to\Qgeth3\scripts\windows\build-release.ps1

# Use relative path with .\
.\scripts\windows\build-release.ps1

# Navigate to script directory
Set-Location scripts\windows
.\build-release.ps1
```

## 🔧 Environment Issues

### PATH and Environment Variables

#### Error: Environment variables not persistent
**Problem:** Variables not saved permanently.

**Solutions:**
```powershell
# Set system-wide environment variable (requires admin)
[System.Environment]::SetEnvironmentVariable("CGO_ENABLED", "0", "Machine")

# Set user-level environment variable
[System.Environment]::SetEnvironmentVariable("CGO_ENABLED", "0", "User")

# Add Go to PATH permanently
$goPath = "C:\Program Files\Go\bin"
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$goPath*") {
    [System.Environment]::SetEnvironmentVariable("PATH", "$currentPath;$goPath", "User")
}

# Refresh environment
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
```

### File System and Permissions

#### Error: `Access to the path is denied`
**Problem:** Insufficient file system permissions.

**Solutions:**
```powershell
# Check file permissions
Get-Acl geth.exe

# Run as Administrator if needed
Start-Process PowerShell -Verb RunAs

# Grant full control to current user
$acl = Get-Acl .
$accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME,"FullControl","Allow")
$acl.SetAccessRule($accessRule)
Set-Acl -Path . -AclObject $acl

# Use different build location
$buildPath = "$env:USERPROFILE\qgeth-build"
New-Item -ItemType Directory -Force -Path $buildPath
Set-Location $buildPath
git clone https://github.com/fourtytwo42/Qgeth3.git
```

### Windows-Specific Dependencies

#### Error: Missing Windows SDK or MSVC
**Problem:** Required Windows development components not installed.

**Solutions:**
```powershell
# Install via Visual Studio Installer
# Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Check installed components
Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\" -ErrorAction SilentlyContinue

# Alternative: Install via Chocolatey
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"

# Verify installation
where cl
cl
```

## 🐛 Advanced Debugging

### Verbose Build Information

```powershell
# Enable verbose Go output
go build -v -x -o geth.exe .\cmd\geth

# Show all compiler commands
go build -work -o geth.exe .\cmd\geth

# Keep temporary files for inspection
$env:TMPDIR = "C:\temp\go-build"
New-Item -ItemType Directory -Force -Path $env:TMPDIR
go build -work -o geth.exe .\cmd\geth
```

### Build Environment Diagnosis

```powershell
# Complete environment check
Write-Host "=== Go Environment ==="
go version
go env

Write-Host "=== Build Environment ==="
Write-Host "CGO_ENABLED: $env:CGO_ENABLED"
Write-Host "GOOS: $env:GOOS"
Write-Host "GOARCH: $env:GOARCH"
Write-Host "GOPATH: $env:GOPATH"
Write-Host "GOCACHE: $env:GOCACHE"

Write-Host "=== System Information ==="
Get-WmiObject Win32_OperatingSystem | Select-Object Caption, Version, Architecture
Get-WmiObject Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-PSDrive C | Select-Object Used, Free

Write-Host "=== Dependencies ==="
Get-Command go, git -ErrorAction SilentlyContinue
git --version
```

### Dependency Analysis

```powershell
# Check Go module dependencies
go list -m all
go mod why -m golang.org/x/crypto

# Download all dependencies with verbose output
go mod download -x

# Verify checksums
go mod verify

# Clean and re-download
go clean -modcache
go mod download
```

### Binary Analysis

```powershell
# Analyze built binary
Get-ChildItem geth.exe | Select-Object Name, Length, LastWriteTime

# Check file type
file geth.exe  # If available

# Dependencies check
dumpbin /dependents geth.exe

# Strings analysis (if available)
strings geth.exe | Select-String -Pattern "(version|commit|quantum)"

# Size analysis
$size = (Get-ChildItem geth.exe).Length / 1MB
Write-Host "Binary size: $([math]::Round($size, 1)) MB"
```

## 🚀 Performance Optimization

### Build Speed Improvements

```powershell
# Use build cache
$env:GOCACHE = "$env:LOCALAPPDATA\go-build"
New-Item -ItemType Directory -Force -Path $env:GOCACHE

# Parallel builds
$env:GOMAXPROCS = [System.Environment]::ProcessorCount
go build -p $env:GOMAXPROCS -o geth.exe .\cmd\geth

# Pre-download dependencies
go mod download

# Use SSD for temp directory
$env:TEMP = "C:\temp"  # Ensure this is on SSD
$env:TMP = "C:\temp"
```

### Memory Optimization

```powershell
# Limit memory usage
$env:GOMEMLIMIT = "4GiB"

# Reduce garbage collection pressure
$env:GOGC = "100"

# Monitor memory during build
Get-Process go | Select-Object ProcessName, WorkingSet64
```

## ✅ Build Health Check

### Pre-Build Verification

```powershell
# Q Geth Windows Build Health Check
Write-Host "=== Q Geth Windows Build Health Check ==="

# Check Go installation
try {
    $goVersion = go version
    Write-Host "✅ Go installed: $goVersion"
    if ($goVersion -match "go1\.(\d+)\.") {
        $minorVersion = [int]$matches[1]
        if ($minorVersion -lt 21) {
            Write-Host "❌ Go version too old, need 1.21+"
            exit 1
        }
    }
} catch {
    Write-Host "❌ Go not found"
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

# Check source code
if ((Test-Path "quantum-geth") -and (Test-Path "quantum-geth\go.mod")) {
    Write-Host "✅ Source code present"
} else {
    Write-Host "❌ Source code missing"
    exit 1
}

# Check environment
if ($env:CGO_ENABLED -eq "0") {
    Write-Host "✅ CGO properly disabled"
} else {
    Write-Host "⚠️  CGO not disabled, setting now"
    $env:CGO_ENABLED = "0"
}

Write-Host "✅ Build environment ready!"
```

### Post-Build Verification

```powershell
# Q Geth Build Verification
Write-Host "=== Q Geth Build Verification ==="

if (Test-Path "geth.exe") {
    Write-Host "✅ Binary exists"
    
    # Check size
    $size = (Get-ChildItem geth.exe).Length / 1MB
    Write-Host "📦 Binary size: $([math]::Round($size, 1))MB"
    
    if ($size -lt 10 -or $size -gt 50) {
        Write-Host "⚠️  Unusual binary size"
    }
    
    # Test basic functionality
    try {
        $version = .\geth.exe version
        Write-Host "✅ Binary executes successfully"
        $version | Select-Object -First 5
    } catch {
        Write-Host "❌ Binary execution failed"
        exit 1
    }
    
    # Check for quantum features
    $help = .\geth.exe help 2>&1
    if ($help -match "quantum") {
        Write-Host "✅ Quantum features detected"
    } else {
        Write-Host "⚠️  Quantum features not found in help"
    }
    
} else {
    Write-Host "❌ Binary not found"
    exit 1
}

Write-Host "✅ Build verification complete!"
```

## 📞 Getting Help

### Community Support
- **GitHub Issues:** [Report build issues](https://github.com/fourtytwo42/Qgeth3/issues)
- **Documentation:** Check [main installation guide](../getting-started/quick-start.md)

### Self-Help Resources
- Run the health check scripts above
- Check [build guide](windows-build-geth.md) for correct process
- Review [node operation troubleshooting](../node-operation/troubleshooting-windows-geth.md)

### Last Resort Debug

```powershell
# Create comprehensive debug log
$debugLog = "debug-build.log"
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

=== Build Attempt ===
"@ | Out-File $debugLog -Append

try {
    .\scripts\windows\build-release.ps1 2>&1 | Out-File $debugLog -Append
} catch {
    $_.Exception.Message | Out-File $debugLog -Append
}

Write-Host "Debug log created: $debugLog"
Write-Host "Share this file when seeking help"
```

**Most Windows build issues are resolved by ensuring proper Go installation, Visual Studio Build Tools, and PowerShell execution policy!** 🛠️ 