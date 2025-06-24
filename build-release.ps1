# Quantum-Geth Release Builder
param(
    [string]$Target = "both",  # geth, miner, both
    [switch]$Clean,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host "Quantum-Geth Release Builder" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Description:" -ForegroundColor Yellow
    Write-Host "  Builds distributable release packages with all dependencies" -ForegroundColor White
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\build-release.ps1 [target] [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Targets:" -ForegroundColor Yellow
    Write-Host "  geth    - Build quantum-geth release only" -ForegroundColor White
    Write-Host "  miner   - Build quantum-miner release only" -ForegroundColor White  
    Write-Host "  both    - Build both releases (default)" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Clean  - Clean existing release folders before building" -ForegroundColor White
    Write-Host "  -Help   - Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\build-release.ps1" -ForegroundColor Green
    Write-Host "  .\build-release.ps1 geth" -ForegroundColor Green
    Write-Host "  .\build-release.ps1 miner -Clean" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output:" -ForegroundColor Yellow
    Write-Host "  releases\quantum-geth-<timestamp>\" -ForegroundColor White
    Write-Host "  releases\quantum-miner-<timestamp>\" -ForegroundColor White
}

function Get-UnixTimestamp {
    return [int][double]::Parse((Get-Date -UFormat %s))
}


function Build-WithVisualStudio {
    param(
        [string]$SourceDir,
        [string]$OutputFile,
        [string]$BuildArgs = ""
    )
    
    Write-Host "Setting up Visual Studio build environment..." -ForegroundColor Yellow
    
    # Find vcvarsall.bat with full path
    $vcvarsallPath = Get-ChildItem "C:\Program Files*" -Recurse -Filter "vcvarsall.bat" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    
    if (-not $vcvarsallPath) {
        throw "Visual Studio vcvarsall.bat not found. Please install Visual Studio Build Tools."
    }
    
    Write-Host "Found Visual Studio: $vcvarsallPath" -ForegroundColor Green
    
    # Convert to absolute paths
    $absoluteSourceDir = Resolve-Path $SourceDir
    $absoluteOutputFile = [System.IO.Path]::GetFullPath($OutputFile)
    
    Write-Host "Building in: $absoluteSourceDir" -ForegroundColor Green
    Write-Host "Output file: $absoluteOutputFile" -ForegroundColor Green
    
    # Create a temporary batch file that sets up VS environment and builds
    $tempBat = [System.IO.Path]::GetTempFileName() + ".bat"
    
    $batchContent = @"
@echo off
echo Setting up Visual Studio environment...
call "$vcvarsallPath" x64
if %ERRORLEVEL% neq 0 (
    echo Failed to set up Visual Studio environment
    exit /b %ERRORLEVEL%
)

echo Visual Studio environment ready
echo Changing to source directory: $absoluteSourceDir
cd /d "$absoluteSourceDir"
if %ERRORLEVEL% neq 0 (
    echo Failed to change to source directory
    exit /b %ERRORLEVEL%
)

echo Building without CGO (Visual Studio compatibility)...
set CGO_ENABLED=0
go build $BuildArgs -o "$absoluteOutputFile" .
exit /b %ERRORLEVEL%
"@
    
    try {
        $batchContent | Out-File -FilePath $tempBat -Encoding ASCII
        
        Write-Host "Running build with Visual Studio environment..." -ForegroundColor Yellow
        $process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $tempBat -Wait -PassThru -NoNewWindow
        
        if ($process.ExitCode -ne 0) {
            throw "Build failed with exit code $($process.ExitCode)"
        }
        
        Write-Host "Build completed successfully!" -ForegroundColor Green
        
    } finally {
        # Clean up temp file
        if (Test-Path $tempBat) {
            Remove-Item $tempBat -Force
        }
    }
}

function New-ReleaseFolder {
    param([string]$BaseName)
    
    $timestamp = Get-UnixTimestamp
    $releaseDir = "releases\$BaseName-$timestamp"
    
    if (-not (Test-Path "releases")) {
        New-Item -ItemType Directory -Path "releases" | Out-Null
    }
    
    if (Test-Path $releaseDir) {
        Remove-Item $releaseDir -Recurse -Force
    }
    
    New-Item -ItemType Directory -Path $releaseDir | Out-Null
    return $releaseDir
}

function Build-QuantumGeth {
    Write-Host "Building Quantum-Geth Release..." -ForegroundColor Blue
    
    # Create release directory
    $releaseDir = New-ReleaseFolder "quantum-geth"
    Write-Host "Release directory: $releaseDir" -ForegroundColor Green
    
    # Build geth
    Write-Host "Compiling quantum-geth..." -ForegroundColor Yellow
    
    Build-WithVisualStudio -SourceDir "quantum-geth\cmd\geth" -OutputFile "quantum-geth\geth.exe" -BuildArgs "-ldflags `"-s -w`""
    
    # Copy files to release
    Write-Host "Preparing release package..." -ForegroundColor Yellow
    
    Copy-Item "quantum-geth\geth.exe" "$releaseDir\"
    
    # Create a basic genesis file for the release
    Write-Host "Creating basic genesis file for release..." -ForegroundColor Yellow
    $basicGenesis = @"
{
  "config": {
    "chainId": 1337,
    "homesteadBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip158Block": 0,
    "byzantiumBlock": 0,
    "constantinopleBlock": 0,
    "petersburgBlock": 0,
    "istanbulBlock": 0,
    "berlinBlock": 0,
    "londonBlock": 0,
    "arrowGlacierBlock": 0,
    "grayGlacierBlock": 0,
    "mergeNetsplitBlock": 0,
    "shanghaiTime": 0,
    "cancunTime": 0,
    "pragueTime": null,
    "verkleTime": null,
    "qmpow": {
      "qbits": 16,
      "tcount": 8192,
      "lnet": 48,
      "epochLen": 100,
      "testMode": false
    }
  },
  "nonce": "0x0",
  "timestamp": "0x0",
  "extraData": "0x51756161746756756d2d476574682052656c65617365",
  "gasLimit": "0x2fefd8",
  "difficulty": "0x1f4",
  "mixHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "coinbase": "0x0000000000000000000000000000000000000000",
  "alloc": {
    "0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A": {
      "balance": "300000000000000000000000"
    }
  },
  "number": "0x0",
  "gasUsed": "0x0",
  "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "baseFeePerGas": "0x7",
  "excessBlobGas": "0x0",
  "blobGasUsed": "0x0"
}
"@
    
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText("$releaseDir\genesis_quantum.json", $basicGenesis, $utf8NoBom)
    
    # Create batch scripts
    @"
@echo off
REM Quantum-Geth Node Launcher (Windows Batch)
REM Usage: start-geth.bat [datadir]

set DATADIR=%1
if "%DATADIR%"=="" set DATADIR=qdata

echo Starting Quantum-Geth Node...
echo Data Directory: %DATADIR%
echo Network ID: 1337
echo Mining: DISABLED (use start-geth-mining.bat to mine)
echo.

geth.exe --datadir "%DATADIR%" --networkid 1337 --nodiscover --allow-insecure-unlock --http --http.addr "0.0.0.0" --http.port 8545 --http.corsdomain "*" --http.api "eth,net,web3,personal,miner,qmpow,admin" --ws --ws.addr "0.0.0.0" --ws.port 8546 --ws.origins "*" --ws.api "eth,net,web3,personal,miner,qmpow,admin" --miner.threads 0 --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A

pause
"@ | Out-File -FilePath "$releaseDir\start-geth.bat" -Encoding ASCII

    @"
@echo off
REM Quantum-Geth Mining Node Launcher (Windows Batch)
REM Usage: start-geth-mining.bat [threads] [datadir]

set THREADS=%1
set DATADIR=%2
if "%THREADS%"=="" set THREADS=1
if "%DATADIR%"=="" set DATADIR=qdata

echo Starting Quantum-Geth Mining Node...
echo Data Directory: %DATADIR%
echo Network ID: 1337
echo Mining: ENABLED with %THREADS% threads
echo.

geth.exe --datadir "%DATADIR%" --networkid 1337 --nodiscover --allow-insecure-unlock --http --http.addr "0.0.0.0" --http.port 8545 --http.corsdomain "*" --http.api "eth,net,web3,personal,miner,qmpow,admin" --ws --ws.addr "0.0.0.0" --ws.port 8546 --ws.origins "*" --ws.api "eth,net,web3,personal,miner,qmpow,admin" --mine --miner.threads %THREADS% --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A

pause
"@ | Out-File -FilePath "$releaseDir\start-geth-mining.bat" -Encoding ASCII

    # Create PowerShell scripts
    @"
# Quantum-Geth Node Launcher (PowerShell)
param(
    [string]`$DataDir = "qdata",
    [switch]`$Help
)

if (`$Help) {
    Write-Host "Quantum-Geth Node Launcher" -ForegroundColor Cyan
    Write-Host "Usage: .\start-geth.ps1 [-DataDir <path>] [-Help]" -ForegroundColor Yellow
    Write-Host "  -DataDir: Blockchain data directory (default: qdata)" -ForegroundColor White
    Write-Host "  -Help: Show this help" -ForegroundColor White
    exit 0
}

Write-Host "Starting Quantum-Geth Node..." -ForegroundColor Blue
Write-Host "Data Directory: `$DataDir" -ForegroundColor Green
Write-Host "Network ID: 1337" -ForegroundColor Green  
Write-Host "Mining: DISABLED (use start-geth-mining.ps1 to mine)" -ForegroundColor Yellow
Write-Host ""

& ".\geth.exe" --datadir "`$DataDir" --networkid 1337 --nodiscover --allow-insecure-unlock --http --http.addr "0.0.0.0" --http.port 8545 --http.corsdomain "*" --http.api "eth,net,web3,personal,miner,qmpow,admin" --ws --ws.addr "0.0.0.0" --ws.port 8546 --ws.origins "*" --ws.api "eth,net,web3,personal,miner,qmpow,admin" --miner.threads 0 --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
"@ | Out-File -FilePath "$releaseDir\start-geth.ps1" -Encoding UTF8

    @"
# Quantum-Geth Mining Node Launcher (PowerShell)
param(
    [int]`$Threads = 1,
    [string]`$DataDir = "qdata",
    [switch]`$Help
)

if (`$Help) {
    Write-Host "Quantum-Geth Mining Node Launcher" -ForegroundColor Cyan
    Write-Host "Usage: .\start-geth-mining.ps1 [-Threads <n>] [-DataDir <path>] [-Help]" -ForegroundColor Yellow
    Write-Host "  -Threads: Number of mining threads (default: 1)" -ForegroundColor White
    Write-Host "  -DataDir: Blockchain data directory (default: qdata)" -ForegroundColor White
    Write-Host "  -Help: Show this help" -ForegroundColor White
    exit 0
}

Write-Host "Starting Quantum-Geth Mining Node..." -ForegroundColor Blue
Write-Host "Data Directory: `$DataDir" -ForegroundColor Green
Write-Host "Network ID: 1337" -ForegroundColor Green
Write-Host "Mining: ENABLED with `$Threads threads" -ForegroundColor Green
Write-Host ""

& ".\geth.exe" --datadir "`$DataDir" --networkid 1337 --nodiscover --allow-insecure-unlock --http --http.addr "0.0.0.0" --http.port 8545 --http.corsdomain "*" --http.api "eth,net,web3,personal,miner,qmpow,admin" --ws --ws.addr "0.0.0.0" --ws.port 8546 --ws.origins "*" --ws.api "eth,net,web3,personal,miner,qmpow,admin" --mine --miner.threads `$Threads --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
"@ | Out-File -FilePath "$releaseDir\start-geth-mining.ps1" -Encoding UTF8

    # Create README
    @"
# Quantum-Geth Release Package

## Quick Start

### Windows (Batch Files)
- **start-geth.bat** - Start node (no mining)
- **start-geth-mining.bat [threads]** - Start mining node

### Windows/Linux (PowerShell)
- **start-geth.ps1** - Start node (no mining)  
- **start-geth-mining.ps1 -Threads <n>** - Start mining node

## First Time Setup

1. Initialize blockchain:
   ```
   geth.exe --datadir qdata init genesis_quantum.json
   ```

2. Start node:
   ```
   .\start-geth.ps1
   ```

3. Mine to this node using the optimized quantum-miner:
   ```
   # Download quantum-miner release package
   # Then run:
   .\quantum-miner.exe -gpu -coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -threads 64
   ```

## Network Details
- **Network ID**: 1337
- **HTTP RPC**: http://localhost:8545
- **WebSocket**: ws://localhost:8546
- **Default Coinbase**: 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A

## Mining
Use the **optimized quantum-miner package** for high-performance GPU mining (10-100x faster), or use start-geth-mining scripts for basic built-in mining.
"@ | Out-File -FilePath "$releaseDir\README.md" -Encoding UTF8

    Write-Host "Quantum-Geth release created: $releaseDir" -ForegroundColor Green
    return $releaseDir
}

function Build-QuantumMiner {
    Write-Host "Building Quantum-Miner Release..." -ForegroundColor Blue
    
    # Create release directory
    $releaseDir = New-ReleaseFolder "quantum-miner"
    Write-Host "Release directory: $releaseDir" -ForegroundColor Green
    
    # Build miner with optimized CUDA support
    Write-Host "Compiling quantum-miner with batch CUDA optimization..." -ForegroundColor Yellow
    
    # Check if we can build with CUDA optimization
    $cudaAvailable = $false
    try {
        $nvccVersion = & nvcc --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $cudaAvailable = $true
            Write-Host "CUDA Toolkit detected - building optimized version" -ForegroundColor Green
        }
    } catch {
        Write-Host "CUDA Toolkit not found - building CPU-only version" -ForegroundColor Yellow
    }
    
    # Build with CUDA if available, otherwise CPU-only
    $currentDir = Get-Location
    try {
        Set-Location "quantum-miner"
        
        if ($cudaAvailable) {
            # Try CUDA build first
            try {
                & ".\build-optimized.ps1"
                Write-Host "CUDA-optimized quantum-miner built successfully" -ForegroundColor Green
            } catch {
                Write-Host "CUDA build failed, falling back to CPU-only build" -ForegroundColor Yellow
                & ".\build-simple.ps1"
            }
        } else {
            & ".\build-simple.ps1"
        }
    } finally {
        Set-Location $currentDir
    }
    
    # Copy files to release
    Write-Host "Preparing release package..." -ForegroundColor Yellow
    
    Copy-Item "quantum-miner\quantum-miner.exe" "$releaseDir\"
    Copy-Item "quantum-miner\pkg" "$releaseDir\" -Recurse
    
    # Create batch scripts
    @"
@echo off
REM Quantum-Miner CPU Launcher (Windows Batch)
REM Usage: start-miner-cpu.bat <coinbase> [threads] [node_url]

set COINBASE=%1
set THREADS=%2
set NODE_URL=%3
if "%THREADS%"=="" set THREADS=1
if "%NODE_URL%"=="" set NODE_URL=http://localhost:8545

if "%COINBASE%"=="" (
    echo ERROR: Coinbase address required!
    echo Usage: start-miner-cpu.bat 0xYourAddress [threads] [node_url]
    pause
    exit /b 1
)

echo Starting Quantum-Miner (CPU Mode)...
echo Coinbase: %COINBASE%
echo Threads: %THREADS%
echo Node URL: %NODE_URL%
echo.

quantum-miner.exe -coinbase "%COINBASE%" -threads %THREADS% -node "%NODE_URL%"

pause
"@ | Out-File -FilePath "$releaseDir\start-miner-cpu.bat" -Encoding ASCII

    @"
@echo off
REM Quantum-Miner GPU Launcher (Windows Batch)
REM Usage: start-miner-gpu.bat <coinbase> [threads] [gpu_id] [node_url]

set COINBASE=%1
set THREADS=%2
set GPU_ID=%3
set NODE_URL=%4
if "%THREADS%"=="" set THREADS=1
if "%GPU_ID%"=="" set GPU_ID=0
if "%NODE_URL%"=="" set NODE_URL=http://localhost:8545

if "%COINBASE%"=="" (
    echo ERROR: Coinbase address required!
    echo Usage: start-miner-gpu.bat 0xYourAddress [threads] [gpu_id] [node_url]
    pause
    exit /b 1
)

echo Starting Quantum-Miner (GPU Mode)...
echo Coinbase: %COINBASE%
echo Threads: %THREADS%
echo GPU ID: %GPU_ID%
echo Node URL: %NODE_URL%
echo.

quantum-miner.exe -gpu -coinbase "%COINBASE%" -threads %THREADS% -gpu-id %GPU_ID% -node "%NODE_URL%"

pause
"@ | Out-File -FilePath "$releaseDir\start-miner-gpu.bat" -Encoding ASCII

    # Create PowerShell scripts
    @"
# Quantum-Miner CPU Launcher (PowerShell)
param(
    [string]`$Coinbase = "",
    [int]`$Threads = 1,
    [string]`$NodeURL = "http://localhost:8545",
    [switch]`$Help
)

if (`$Help -or `$Coinbase -eq "") {
    Write-Host "Quantum-Miner CPU Launcher" -ForegroundColor Cyan
    Write-Host "Usage: .\start-miner-cpu.ps1 -Coinbase <address> [options]" -ForegroundColor Yellow
    Write-Host "  -Coinbase: Mining reward address (required)" -ForegroundColor White
    Write-Host "  -Threads: CPU mining threads (default: 1)" -ForegroundColor White
    Write-Host "  -NodeURL: Quantum-Geth node URL (default: http://localhost:8545)" -ForegroundColor White
    Write-Host "Example: .\start-miner-cpu.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A" -ForegroundColor Green
    if (`$Coinbase -eq "") { exit 1 }
    exit 0
}

Write-Host "Starting Quantum-Miner (CPU Mode)..." -ForegroundColor Blue
Write-Host "Coinbase: `$Coinbase" -ForegroundColor Green
Write-Host "Threads: `$Threads" -ForegroundColor Green
Write-Host "Node URL: `$NodeURL" -ForegroundColor Green
Write-Host ""

& ".\quantum-miner.exe" -coinbase "`$Coinbase" -threads `$Threads -node "`$NodeURL"
"@ | Out-File -FilePath "$releaseDir\start-miner-cpu.ps1" -Encoding UTF8

    @"
# Quantum-Miner GPU Launcher (PowerShell)
param(
    [string]`$Coinbase = "",
    [int]`$Threads = 1,
    [int]`$GpuId = 0,
    [string]`$NodeURL = "http://localhost:8545",
    [switch]`$Help
)

if (`$Help -or `$Coinbase -eq "") {
    Write-Host "Quantum-Miner GPU Launcher" -ForegroundColor Cyan
    Write-Host "Usage: .\start-miner-gpu.ps1 -Coinbase <address> [options]" -ForegroundColor Yellow
    Write-Host "  -Coinbase: Mining reward address (required)" -ForegroundColor White
    Write-Host "  -Threads: Mining threads (default: 1)" -ForegroundColor White
    Write-Host "  -GpuId: GPU device ID (default: 0)" -ForegroundColor White
    Write-Host "  -NodeURL: Quantum-Geth node URL (default: http://localhost:8545)" -ForegroundColor White
    Write-Host "Example: .\start-miner-gpu.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A" -ForegroundColor Green
    if (`$Coinbase -eq "") { exit 1 }
    exit 0
}

Write-Host "Starting Quantum-Miner (GPU Mode)..." -ForegroundColor Blue
Write-Host "Coinbase: `$Coinbase" -ForegroundColor Green
Write-Host "Threads: `$Threads" -ForegroundColor Green
Write-Host "GPU ID: `$GpuId" -ForegroundColor Green
Write-Host "Node URL: `$NodeURL" -ForegroundColor Green
Write-Host ""

& ".\quantum-miner.exe" -gpu -coinbase "`$Coinbase" -threads `$Threads -gpu-id `$GpuId -node "`$NodeURL"
"@ | Out-File -FilePath "$releaseDir\start-miner-gpu.ps1" -Encoding UTF8

    # Create README
    @"
# Quantum-Miner Release Package

## Quick Start

### Windows (Batch Files)
- **start-miner-cpu.bat <coinbase>** - CPU mining
- **start-miner-gpu.bat <coinbase>** - GPU mining

### Windows/Linux (PowerShell)  
- **start-miner-cpu.ps1 -Coinbase <address>** - CPU mining
- **start-miner-gpu.ps1 -Coinbase <address>** - GPU mining

## Examples

### CPU Mining
```
start-miner-cpu.bat 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
.\start-miner-cpu.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 4
```

### GPU Mining (CUDA Optimized)
```
start-miner-gpu.bat 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
.\start-miner-gpu.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 64
```

## Performance (Batch CUDA Optimized)
- **CPU Mining**: ~0.36 puzzles/sec (1 thread)
- **GPU Mining**: ~10-100x faster (64-256 threads, batch processing)
- **Expected GPU Utilization**: 80%+ (vs 4% with old version)

## Requirements
- **CPU Mode**: No additional dependencies
- **GPU Mode**: NVIDIA GPU with CUDA support (optional, falls back to CPU)

## Network
Connect to a running quantum-geth node:
- Default: http://localhost:8545
- Custom: Use -NodeURL parameter
"@ | Out-File -FilePath "$releaseDir\README.md" -Encoding UTF8

    Write-Host "Quantum-Miner release created: $releaseDir" -ForegroundColor Green
    return $releaseDir
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

Write-Host "Quantum-Geth Release Builder" -ForegroundColor Cyan
Write-Host "Target: $Target" -ForegroundColor Yellow
Write-Host ""

if ($Clean -and (Test-Path "releases")) {
    Write-Host "Cleaning existing releases..." -ForegroundColor Yellow
    Remove-Item "releases" -Recurse -Force
}

try {
    switch ($Target.ToLower()) {
        "geth" {
            $gethRelease = Build-QuantumGeth
            Write-Host ""
            Write-Host "SUCCESS: Quantum-Geth release ready!" -ForegroundColor Green
            Write-Host "Location: $gethRelease" -ForegroundColor Cyan
        }
        "miner" {
            $minerRelease = Build-QuantumMiner
            Write-Host ""
            Write-Host "SUCCESS: Quantum-Miner release ready!" -ForegroundColor Green
            Write-Host "Location: $minerRelease" -ForegroundColor Cyan
        }
        "both" {
            $gethRelease = Build-QuantumGeth
            Write-Host ""
            $minerRelease = Build-QuantumMiner
            Write-Host ""
            Write-Host "SUCCESS: Both releases ready!" -ForegroundColor Green
            Write-Host "Quantum-Geth: $gethRelease" -ForegroundColor Cyan
            Write-Host "Quantum-Miner: $minerRelease" -ForegroundColor Cyan
        }
        default {
            throw "Invalid target: $Target. Use: geth, miner, or both"
        }
    }
    
    Write-Host ""
    Write-Host "Release packages are ready for distribution!" -ForegroundColor Blue
    Write-Host "Users can run these independently without needing the source code." -ForegroundColor Blue
    
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 