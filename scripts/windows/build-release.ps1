#!/usr/bin/env pwsh

# Q Coin Build Script - Creates timestamped releases
# Usage: ./build-release.ps1 [component]
# Components: geth, miner, both (default: both)

param(
    [Parameter(Position=0)]
    [ValidateSet("geth", "miner", "both")]
    [string]$Component = "both"
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "Building Q Coin Release..." -ForegroundColor Cyan
Write-Host ""

# Get timestamp for folder naming
$timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

# Get the script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$QuantumGethDir = Join-Path $ProjectRoot "quantum-geth"
$QuantumMinerDir = Join-Path $ProjectRoot "quantum-miner"
$ReleasesDir = Join-Path $ProjectRoot "releases"

Write-Host "Project Root: $ProjectRoot"
Write-Host "Releases: $ReleasesDir"

# Create releases directory if it doesn't exist
if (-not (Test-Path $ReleasesDir)) {
    New-Item -ItemType Directory -Path $ReleasesDir -Force | Out-Null
}

# Function to fix Go modules
function Fix-GoModules {
    param([string]$ModuleDir)
    
    Write-Host "Fixing Go modules in $ModuleDir..." -ForegroundColor Yellow
    Push-Location $ModuleDir
    try {
        # Clean any lock files that might be causing issues
        $goModCache = go env GOMODCACHE
        if (Test-Path "$goModCache\cache\vcs\*\shallow.lock") {
            Get-ChildItem "$goModCache\cache\vcs\*\shallow.lock" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        }
        
        # Download and tidy modules
        go mod download 2>$null
        go mod tidy 2>$null
        
        Write-Host "Modules fixed successfully" -ForegroundColor Green
    } catch {
        Write-Host "Module fix failed, continuing anyway..." -ForegroundColor Yellow
    } finally {
        Pop-Location
    }
}

# Build geth
if ($Component -eq "geth" -or $Component -eq "both") {
    Write-Host "Building quantum-geth..." -ForegroundColor Yellow
    
    if (-not (Test-Path $QuantumGethDir)) {
        Write-Error "quantum-geth directory not found at: $QuantumGethDir"
        exit 1
    }
    
    # Fix modules first
    Fix-GoModules $QuantumGethDir
    
    # Build geth
    Set-Location $QuantumGethDir
    try {
        # CRITICAL: Always use CGO_ENABLED=0 for geth to ensure compatibility
        $env:CGO_ENABLED = "0"
        Write-Host "ENFORCING: CGO_ENABLED=0 for geth build (quantum field compatibility)" -ForegroundColor Yellow
        
        $BUILD_TIME = Get-Date -Format "yyyy-MM-dd_HH:mm:ss"
        $GIT_COMMIT = git rev-parse --short HEAD 2>$null
        if (-not $GIT_COMMIT) { $GIT_COMMIT = "unknown" }
        
        $LDFLAGS = "-X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME"
        
        go build -ldflags $LDFLAGS -o "geth.exe" "./cmd/geth"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "quantum-geth built successfully (CGO_ENABLED=0)" -ForegroundColor Green
            
            # Create timestamped release directly in releases directory
            $releaseDir = Join-Path $ReleasesDir "quantum-geth-$timestamp"
            New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
            Copy-Item "geth.exe" (Join-Path $releaseDir "geth.exe") -Force
            
            # Copy genesis JSON files for auto-reset functionality
            Write-Host "Adding genesis configurations for auto-reset..." -ForegroundColor Yellow
            $configsDir = Join-Path $ProjectRoot "configs"
            if (Test-Path $configsDir) {
                Copy-Item (Join-Path $configsDir "genesis_quantum_testnet.json") (Join-Path $releaseDir "genesis_quantum_testnet.json") -Force
                Copy-Item (Join-Path $configsDir "genesis_quantum_dev.json") (Join-Path $releaseDir "genesis_quantum_dev.json") -Force
                Write-Host "Genesis files added successfully" -ForegroundColor Green
            } else {
                Write-Host "Warning: configs directory not found, skipping genesis files" -ForegroundColor Yellow
            }
            
            # Create PowerShell launcher with genesis auto-reset
            @'
param([string]$Network = "testnet", [switch]$Mining, [switch]$Help)

if ($Help) {
    Write-Host "Q Coin Geth Launcher with Auto-Reset" -ForegroundColor Cyan
    Write-Host "Usage: .\start-geth.ps1 [network] [options]"
    Write-Host "Networks: testnet, devnet"
    Write-Host "Features: Automatic blockchain reset when genesis changes"
    exit 0
}

$configs = @{
    "testnet" = @{ 
        chainid = 73235; 
        datadir = "$env:APPDATA\Qcoin\testnet"; 
        port = 30303; 
        genesis = "genesis_quantum_testnet.json" 
    }
    "devnet" = @{ 
        chainid = 73234; 
        datadir = "$env:APPDATA\Qcoin\devnet"; 
        port = 30305; 
        genesis = "genesis_quantum_dev.json" 
    }
}

if (-not $configs.ContainsKey($Network)) {
    Write-Host "Error: Invalid network '$Network'. Use: testnet, devnet" -ForegroundColor Red
    exit 1
}

$config = $configs[$Network]
Write-Host "Starting Q Coin $Network (Chain ID: $($config.chainid))" -ForegroundColor Cyan
Write-Host "Genesis: $($config.genesis)" -ForegroundColor Yellow

if (-not (Test-Path $config.datadir)) {
    New-Item -ItemType Directory -Path $config.datadir -Force | Out-Null
}

# CRITICAL: Initialize with genesis file for auto-reset functionality
Write-Host "Initializing with genesis file (auto-reset if changed)..." -ForegroundColor Yellow
if (Test-Path $config.genesis) {
    & ".\geth.exe" --datadir $config.datadir init $config.genesis
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Genesis initialization successful" -ForegroundColor Green
    } else {
        Write-Host "Genesis initialization failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Error: Genesis file $($config.genesis) not found!" -ForegroundColor Red
    exit 1
}

$threads = if ($Mining) { "1" } else { "0" }
$coinbase = "0x0000000000000000000000000000000000000001"
$args = @("--datadir", $config.datadir, "--networkid", $config.chainid, "--port", $config.port,
    "--http", "--http.addr", "0.0.0.0", "--http.port", "8545", "--http.corsdomain", "*",
    "--http.api", "eth,net,web3,personal,admin,txpool,miner,qmpow", "--mine", "--miner.threads", $threads, "--miner.etherbase", $coinbase)

Write-Host "Starting Q Coin node..." -ForegroundColor Cyan
& ".\geth.exe" @args
'@ | Out-File -FilePath (Join-Path $releaseDir "start-geth.ps1") -Encoding UTF8

            # Create batch launcher with genesis auto-reset
            @'
@echo off
set NETWORK=%1
if "%NETWORK%"=="" set NETWORK=testnet

if "%NETWORK%"=="testnet" (
    set CHAINID=73235
    set DATADIR=%APPDATA%\Qcoin\testnet
    set GENESIS=genesis_quantum_testnet.json
) else if "%NETWORK%"=="devnet" (
    set CHAINID=73234
    set DATADIR=%APPDATA%\Qcoin\devnet
    set GENESIS=genesis_quantum_dev.json
) else (
    echo Error: Invalid network '%NETWORK%'. Use: testnet, devnet
    exit /b 1
)

echo Starting Q Coin %NETWORK% (Chain ID: %CHAINID%)
echo Genesis: %GENESIS%
if not exist "%DATADIR%" mkdir "%DATADIR%"

echo Initializing with genesis file (auto-reset if changed)...
if not exist "%GENESIS%" (
    echo Error: Genesis file %GENESIS% not found!
    exit /b 1
)

geth.exe --datadir "%DATADIR%" init "%GENESIS%"
if %ERRORLEVEL% neq 0 (
    echo Genesis initialization failed
    exit /b 1
)
echo Genesis initialization successful

echo Starting Q Coin node...
geth.exe --datadir "%DATADIR%" --networkid %CHAINID% --http --http.addr 0.0.0.0 --http.port 8545 --http.corsdomain "*" --http.api "eth,net,web3,personal,admin,txpool,miner,qmpow" --mine --miner.threads 0 --miner.etherbase 0x0000000000000000000000000000000000000001
'@ | Out-File -FilePath (Join-Path $releaseDir "start-geth.bat") -Encoding ASCII

            # Create enhanced README
            @"
# Q Coin Geth Release $timestamp

Built: $(Get-Date)
Component: Quantum-Geth (Q Coin Blockchain Node)

## Features
- 🔄 **Auto-Reset**: Automatically wipes and restarts blockchain when genesis changes
- 🚀 **QMPoW Consensus**: Quantum Micro-Puzzle Proof of Work
- 🛡️ **Minimum Difficulty**: Protected against difficulty collapse (minimum 200)
- 🔗 **External Miner Support**: Full qmpow API for external mining

## Quick Start
PowerShell: .\start-geth.ps1 [testnet|devnet] [-mining]
Batch: start-geth.bat [testnet|devnet]

## Network Information
- **Testnet**: Chain ID 73235, genesis_quantum_testnet.json
- **Devnet**: Chain ID 73234, genesis_quantum_dev.json

## Auto-Reset Functionality
The node automatically detects when genesis parameters change and:
1. 🔍 Compares stored vs new genesis hash
2. ⚠️ Logs warning about blockchain reset
3. 🧹 Wipes all blockchain data completely  
4. 🚀 Starts fresh from block 1 with new genesis

## API Access
- **HTTP RPC**: http://localhost:8545
- **APIs**: eth, net, web3, personal, admin, txpool, miner, qmpow
- **Data Directory**: %APPDATA%\Qcoin\[network]\

## Genesis Files Included
- genesis_quantum_testnet.json (Chain ID: 73235)
- genesis_quantum_dev.json (Chain ID: 73234)

See project README for full documentation.
"@ | Out-File -FilePath (Join-Path $releaseDir "README.md") -Encoding UTF8
            
            Write-Host "Created release: $releaseDir" -ForegroundColor Green
        } else {
            Write-Error "quantum-geth build failed!"
            exit 1
        }
    } finally {
        Set-Location $ProjectRoot
    }
    Write-Host ""
}

# Build miner
if ($Component -eq "miner" -or $Component -eq "both") {
    Write-Host "Building quantum-miner..." -ForegroundColor Yellow
    
    if (-not (Test-Path $QuantumMinerDir)) {
        Write-Error "quantum-miner directory not found at: $QuantumMinerDir"
        exit 1
    }
    
    # Fix modules first
    Fix-GoModules $QuantumMinerDir
    
    # Build miner
    Set-Location $QuantumMinerDir
    try {
        # Use CGO_ENABLED=0 for Windows miner (uses CuPy instead of native CUDA)
        $env:CGO_ENABLED = "0"
        Write-Host "INFO: Using CGO_ENABLED=0 for Windows miner (CuPy GPU support)" -ForegroundColor Cyan
        
        $BUILD_TIME = Get-Date -Format "yyyy-MM-dd_HH:mm:ss"
        $GIT_COMMIT = git rev-parse --short HEAD 2>$null
        if (-not $GIT_COMMIT) { $GIT_COMMIT = "unknown" }
        
        $LDFLAGS = "-X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME"
        
        go build -ldflags $LDFLAGS -o "quantum-miner.exe" "."
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "quantum-miner built successfully" -ForegroundColor Green
            
            # Create timestamped release directly in releases directory
            $releaseDir = Join-Path $ReleasesDir "quantum-miner-$timestamp"
            New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
            Copy-Item "quantum-miner.exe" (Join-Path $releaseDir "quantum-miner.exe") -Force
            
            # Copy essential Python scripts for GPU mining support
            Write-Host "Adding Python GPU scripts for GPU acceleration..." -ForegroundColor Yellow
            $pythonScriptsDir = Join-Path $releaseDir "pkg" 
            $quantumScriptsDir = Join-Path $pythonScriptsDir "quantum"
            New-Item -ItemType Directory -Path $quantumScriptsDir -Force | Out-Null
            
            $requiredScripts = @(
                "pkg/quantum/qiskit_gpu.py",
                "pkg/quantum/cupy_gpu.py", 
                "test_gpu.py"
            )
            
            foreach ($script in $requiredScripts) {
                $scriptPath = Join-Path $QuantumMinerDir $script
                if (Test-Path $scriptPath) {
                    if ($script -eq "test_gpu.py") {
                        Copy-Item $scriptPath (Join-Path $releaseDir "test_gpu.py") -Force
                    } else {
                        Copy-Item $scriptPath (Join-Path $quantumScriptsDir (Split-Path $script -Leaf)) -Force
                    }
                    Write-Host "  Added: $script" -ForegroundColor Green
                } else {
                    Write-Host "  Warning: $script not found" -ForegroundColor Yellow
                }
            }
            Write-Host "Python GPU scripts added successfully" -ForegroundColor Green
            
            # Create PowerShell launcher
            @'
param([int]$Threads = 8, [string]$Node = "http://localhost:8545", [string]$Coinbase = "", [switch]$Help)

if ($Help) {
    Write-Host "Q Coin Quantum Miner Launcher" -ForegroundColor Cyan
    Write-Host "Usage: .\start-miner.ps1 [-threads <n>] [-node <url>] [-coinbase <addr>]"
    exit 0
}

Write-Host "Q Coin Quantum Miner Starting..." -ForegroundColor Cyan

# Test connection
try {
    $response = Invoke-RestMethod -Uri $Node -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'
    $chainId = [Convert]::ToInt32($response.result, 16)
    Write-Host "Connected to Chain ID: $chainId" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Cannot connect to $Node" -ForegroundColor Red
    exit 1
}

if ($Threads -eq 0) { $Threads = 8 }
if ($Coinbase -eq "") { $Coinbase = "0x0000000000000000000000000000000000000001" }

Write-Host "Mining with $Threads threads to $Coinbase" -ForegroundColor Cyan
& ".\quantum-miner.exe" -node $Node -coinbase $Coinbase -threads $Threads
'@ | Out-File -FilePath (Join-Path $releaseDir "start-miner.ps1") -Encoding UTF8

            # Create batch launcher
            @'
@echo off
set THREADS=%1
set NODE=%2
set COINBASE=%3
if "%THREADS%"=="" set THREADS=8
if "%NODE%"=="" set NODE=http://localhost:8545
if "%COINBASE%"=="" set COINBASE=0x0000000000000000000000000000000000000001

echo Q Coin Quantum Miner Starting...
echo Threads: %THREADS%
echo Node: %NODE%
echo Coinbase: %COINBASE%

quantum-miner.exe -node %NODE% -coinbase %COINBASE% -threads %THREADS%
'@ | Out-File -FilePath (Join-Path $releaseDir "start-miner.bat") -Encoding ASCII

            # Create README
            @"
# Q Coin Quantum Miner Release $timestamp

Built: $(Get-Date)
Component: Quantum-Miner (Mining Software)

## Quick Start
PowerShell: .\start-miner.ps1 [-threads 8] [-node http://localhost:8545] [-coinbase 0xYourAddress]
Batch: start-miner.bat [threads] [node_url] [coinbase_address]

## Examples
start-miner.bat 8 http://localhost:8545 0x1234567890abcdef1234567890abcdef12345678
start-miner.ps1 -threads 8 -coinbase 0x1234567890abcdef1234567890abcdef12345678

## Performance
- CPU Mining: ~0.3-0.8 puzzles/sec
- GPU Mining: ~2.0 puzzles/sec (with CuPy)

## Requirements
- Q Coin Geth node running
- Valid Ethereum address for coinbase (mining rewards)
- For GPU: Python with CuPy

See project README for full documentation.
"@ | Out-File -FilePath (Join-Path $releaseDir "README.md") -Encoding UTF8
            
            Write-Host "Created release: $releaseDir" -ForegroundColor Green
        } else {
            Write-Error "quantum-miner build failed!"
            exit 1
        }
    } finally {
        Set-Location $ProjectRoot
    }
    Write-Host ""
}

Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "Releases created in:" -ForegroundColor Cyan
if ($Component -eq "geth" -or $Component -eq "both") {
    Write-Host "  Geth: $ReleasesDir\quantum-geth-*\" -ForegroundColor White
}
if ($Component -eq "miner" -or $Component -eq "both") {
    Write-Host "  Miner: $ReleasesDir\quantum-miner-*\" -ForegroundColor White
} 