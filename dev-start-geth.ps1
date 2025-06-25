# Q Coin Dev Network - Start Geth (No Mining)
# Starts Q Coin dev network node (Chain ID 73234) without mining - serves work to external miners
# This script ONLY connects to Q Coin Dev network, never Ethereum!

param(
    [string]$datadir = "qdata",
    [int]$networkid = 73234,
    [int]$port = 30303,
    [int]$httpport = 8545,
    [int]$authrpcport = 8551,
    [string]$etherbase = "0x1234567890123456789012345678901234567890",
    [int]$verbosity = 3
)

Write-Host "🪙 Q COIN DEV NETWORK - STARTING NODE (EXTERNAL MINING) 🪙" -ForegroundColor Cyan
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  * Network: Q Coin Dev/Staging" -ForegroundColor Yellow
Write-Host "  * Chain ID: $networkid (Dev Network)" -ForegroundColor Yellow
Write-Host "  * Data Directory: $datadir" -ForegroundColor Yellow
Write-Host "  * Port: $port" -ForegroundColor Yellow
Write-Host "  * HTTP Port: $httpport" -ForegroundColor Yellow
Write-Host "  * Auth RPC Port: $authrpcport" -ForegroundColor Yellow
Write-Host "  * Etherbase: $etherbase" -ForegroundColor Yellow
Write-Host "  * Verbosity: $verbosity" -ForegroundColor Yellow
Write-Host "  * Mining: EXTERNAL MINERS ONLY (0 internal threads)" -ForegroundColor Green
Write-Host ""

Write-Host "Quantum-Geth Features:" -ForegroundColor Magenta
Write-Host "  * 48 sequential quantum puzzles (16 qubits x 8,192 T-gates)" -ForegroundColor Gray
Write-Host "  * Bitcoin-style halving (50 QGC -> 25 QGC -> 12.5 QGC...)" -ForegroundColor Gray
Write-Host "  * 600,000 block epochs (approximately 6 months)" -ForegroundColor Gray
Write-Host "  * Branch-serial quantum circuit execution" -ForegroundColor Gray
Write-Host "  * Mahadev->CAPSS->Nova proof stack" -ForegroundColor Gray
Write-Host "  * Dilithium-2 self-attestation" -ForegroundColor Gray
Write-Host "  * ASERT-Q difficulty adjustment (12s target)" -ForegroundColor Gray
Write-Host "  * Single RLP quantum blob (197 bytes)" -ForegroundColor Gray
Write-Host ""

# Check if data directory exists, initialize if needed
if (-not (Test-Path "$datadir\geth\chaindata")) {
    Write-Host "🏗️  Initializing Q Coin Dev blockchain..." -ForegroundColor Yellow
    Write-Host "   Data Directory: $datadir" -ForegroundColor Gray
    Write-Host "   Genesis File: genesis_quantum_dev.json" -ForegroundColor Gray
    Write-Host ""
    
    # Create data directory
    if (-not (Test-Path $datadir)) {
        New-Item -ItemType Directory -Path $datadir -Force | Out-Null
    }
    
    # Find geth executable for initialization
    $GethForInit = $null
    $GethReleaseDir = Get-ChildItem -Path "releases\quantum-geth-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if ($GethReleaseDir) {
        $GethForInit = "$($GethReleaseDir.FullName)\geth.exe"
    } elseif (Test-Path ".\geth.exe") {
        $GethForInit = ".\geth.exe"
    } else {
        Write-Host "❌ ERROR: No geth executable found for initialization" -ForegroundColor Red
        Write-Host "   Please build geth first with: .\build-linux.ps1" -ForegroundColor Yellow
        exit 1
    }
    
    # Initialize with genesis
    try {
        & "$GethForInit" --datadir "$datadir" init "genesis_quantum_dev.json" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Q Coin Dev blockchain initialized successfully!" -ForegroundColor Green
            Write-Host "   Node will now start and sync with Q Coin Dev network peers..." -ForegroundColor Green
        } else {
            Write-Host "❌ ERROR: Failed to initialize blockchain!" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "❌ ERROR: Failed to run geth init: $_" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
} else {
    Write-Host "✅ Dev blockchain already initialized" -ForegroundColor Green
}

Write-Host "Starting quantum geth node (NO MINING)..." -ForegroundColor Magenta
Write-Host "This node serves RPC/HTTP endpoints for external miners WITHOUT mining itself." -ForegroundColor Green
Write-Host "Use .\start-geth-mining.ps1 to start mining, or external miners to mine." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the node." -ForegroundColor Yellow
Write-Host ""

# Find the newest quantum-geth release or use development version
$GethReleaseDir = Get-ChildItem -Path "releases\quantum-geth-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
if ($GethReleaseDir) {
    $GethExecutable = "$($GethReleaseDir.FullName)\geth.exe"
    Write-Host "Using geth from release: $($GethReleaseDir.Name)" -ForegroundColor Green
} else {
    $GethExecutable = ".\geth.exe"
    if (-not (Test-Path $GethExecutable)) {
        Write-Host "Geth executable not found. Building release..." -ForegroundColor Yellow
        try {
            & ".\build-release.ps1" geth
            $GethReleaseDir = Get-ChildItem -Path "releases\quantum-geth-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
            if ($GethReleaseDir) {
                $GethExecutable = "$($GethReleaseDir.FullName)\geth.exe"
                Write-Host "SUCCESS: Release built at $($GethReleaseDir.FullName)" -ForegroundColor Green
            } else {
                throw "Failed to create release"
            }
        } catch {
            Write-Host "ERROR: Failed to build quantum-geth release: $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Using development geth from root directory" -ForegroundColor Yellow
    }
}

# Start geth WITHOUT mining - pure RPC node for external miners
& "$GethExecutable" `
    --datadir $datadir `
    --networkid $networkid `
    --port 30305 `
    --http `
    --http.addr 0.0.0.0 `
    --http.port $httpport `
    --http.api "eth,net,web3,personal,miner,qmpow,admin,debug,trace" `
    --http.corsdomain "*" `
    --http.vhosts "*" `
    --mine `
    --miner.threads 0 `
    --miner.etherbase $etherbase `
    --authrpc.port $authrpcport `
    --bootnodes "enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.50.254:30305,enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.50.152:30305" `
    --verbosity $verbosity `
    --log.vmodule "rpc=1" `
    --allow-insecure-unlock

Write-Host ""
Write-Host "Quantum geth node stopped." -ForegroundColor Green 