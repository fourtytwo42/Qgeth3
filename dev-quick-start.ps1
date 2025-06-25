# Q Coin Dev Quick Start
# One-command development shortcut: Build + Start Dev Network
# This script builds geth and immediately starts the Q Coin dev network with peer connections
# Usage: .\dev-quick-start.ps1

param(
    [switch]$clean = $false,           # Clean build (rebuild everything)
    [switch]$mining = $false,          # Start with internal mining enabled
    [switch]$help = $false             # Show help
)

# Show help
if ($help) {
    Write-Host " Q Coin Dev Quick Start" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "One-command development shortcut: Build + Start Dev Network" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Usage: .\dev-quick-start.ps1 [OPTIONS]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Green
    Write-Host "  -clean     Clean build (rebuild everything)"
    Write-Host "  -mining    Start with internal mining enabled"
    Write-Host "  -help      Show this help message"
    Write-Host ""
    Write-Host "What this script does:" -ForegroundColor Yellow
    Write-Host "  1. Builds quantum-geth (clean if -clean specified)"
    Write-Host "  2. Auto-initializes dev blockchain if needed"
    Write-Host "  3. Starts Q Coin Dev Network (Chain ID 73234)"
    Write-Host "  4. Connects to dev peer bootnodes automatically"
    Write-Host "  5. Serves RPC endpoints for external miners"
    Write-Host ""
    Write-Host "Networks:" -ForegroundColor Magenta
    Write-Host "  Dev Network: Chain ID 73234, Port 30305"
    Write-Host "  Peers: 192.168.50.254:30305 & 192.168.50.152:30305"
    Write-Host "  RPC: http://127.0.0.1:8545"
    Write-Host ""
    exit 0
}

Write-Host " Q COIN DEV QUICK START" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Build quantum-geth
Write-Host " Step 1: Building Quantum-Geth..." -ForegroundColor Yellow

if ($clean) {
    Write-Host "   Clean build requested - rebuilding everything" -ForegroundColor Gray
    try {
        if (Test-Path "geth.exe") { Remove-Item "geth.exe" -Force }
        if (Test-Path "geth") { Remove-Item "geth" -Force }
        if (Test-Path "quantum-geth") { Remove-Item "quantum-geth" -Force }
        Write-Host "   Cleaned previous builds" -ForegroundColor Green
    } catch {
        Write-Host "   Warning: Could not clean some files: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Build using existing build script
if (Test-Path "build-linux.sh") {
    Write-Host "   Running build-linux.sh..." -ForegroundColor Gray
    try {
        & ".\build-linux.sh" geth 2>&1 | Out-Host
        if ($LASTEXITCODE -eq 0) {
            Write-Host " Quantum-Geth built successfully!" -ForegroundColor Green
        } else {
            Write-Host " Build failed!" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host " Build error: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host " ERROR: build-linux.sh not found!" -ForegroundColor Red
    Write-Host "   Make sure you're in the Q Coin root directory" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 2: Initialize and start dev network
Write-Host " Step 2: Starting Q Coin Dev Network..." -ForegroundColor Yellow

# Data directory configuration
$datadir = "qdata"
$networkid = 73234
$port = 30305
$httpport = 8545
$wsport = 8546
$etherbase = "0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"

# Find geth executable
$GethExecutable = $null
if (Test-Path "geth.exe") {
    $GethExecutable = ".\geth.exe"
    Write-Host "   Using Windows geth executable" -ForegroundColor Gray
} elseif (Test-Path "geth") {
    $GethExecutable = ".\geth"
    Write-Host "   Using Linux/wrapper geth executable" -ForegroundColor Gray
} else {
    Write-Host " ERROR: No geth executable found!" -ForegroundColor Red
    exit 1
}

# Initialize blockchain if needed
if (-not (Test-Path "$datadir\geth\chaindata")) {
    Write-Host "  Initializing Q Coin Dev blockchain..." -ForegroundColor Yellow
    Write-Host "   Data Directory: $datadir" -ForegroundColor Gray
    Write-Host "   Genesis File: genesis_quantum_dev.json" -ForegroundColor Gray
    
    # Create data directory
    if (-not (Test-Path $datadir)) {
        New-Item -ItemType Directory -Path $datadir -Force | Out-Null
    }
    
    # Initialize with genesis
    try {
        & "$GethExecutable" --datadir "$datadir" init "genesis_quantum_dev.json" 2>&1 | Out-Host
        if ($LASTEXITCODE -eq 0) {
            Write-Host " Q Coin Dev blockchain initialized!" -ForegroundColor Green
        } else {
            Write-Host " ERROR: Failed to initialize blockchain!" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host " ERROR: Failed to run geth init: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host " Dev blockchain already initialized" -ForegroundColor Green
}

Write-Host ""

# Step 3: Start the dev network
Write-Host " Step 3: Starting Dev Network Node..." -ForegroundColor Yellow

# Display configuration
Write-Host ""
Write-Host " Dev Network Configuration:" -ForegroundColor Cyan
Write-Host "  Network: Q Coin Dev/Staging" -ForegroundColor Gray
Write-Host "  Chain ID: $networkid" -ForegroundColor Gray
Write-Host "  Data Directory: $datadir" -ForegroundColor Gray
Write-Host "  P2P Port: $port" -ForegroundColor Gray
Write-Host "  RPC Port: $httpport" -ForegroundColor Gray
Write-Host "  WebSocket Port: $wsport" -ForegroundColor Gray
Write-Host "  Etherbase: $etherbase" -ForegroundColor Gray
if ($mining) {
    Write-Host "  Mining: INTERNAL THREADS ENABLED" -ForegroundColor Green
} else {
    Write-Host "  Mining: EXTERNAL MINERS ONLY" -ForegroundColor Yellow
}
Write-Host ""

Write-Host " Quantum-Geth Features:" -ForegroundColor Magenta
Write-Host "   128 sequential quantum puzzles (16 qubits  20 T-gates)" -ForegroundColor Gray
Write-Host "   Bitcoin-style halving (50 QGC  25 QGC  12.5 QGC...)" -ForegroundColor Gray
Write-Host "   600,000 block epochs (~6 months)" -ForegroundColor Gray
Write-Host "   ASERT-Q difficulty adjustment (12s target)" -ForegroundColor Gray
Write-Host "   Auto-connect to dev bootnodes" -ForegroundColor Gray
Write-Host ""

Write-Host " Starting node and connecting to dev peers..." -ForegroundColor Green
Write-Host "   Bootnodes: 192.168.50.254:30305 & 192.168.50.152:30305" -ForegroundColor Gray
Write-Host "   Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Build geth arguments
$gethArgs = @(
    "--datadir", "$datadir",
    "--networkid", "$networkid",
    "--port", "$port",
    "--http",
    "--http.addr", "0.0.0.0",
    "--http.port", "$httpport",
    "--http.api", "eth,net,web3,personal,miner,qmpow,admin,debug,trace",
    "--http.corsdomain", "*",
    "--http.vhosts", "*",
    "--ws",
    "--ws.addr", "0.0.0.0",
    "--ws.port", "$wsport",
    "--ws.api", "eth,net,web3,personal,admin,miner",
    "--ws.origins", "*",
    "--miner.etherbase", "$etherbase",
    "--bootnodes", "enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.50.254:30305,enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.50.152:30305",
    "--nat", "any",
    "--maxpeers", "25",
    "--verbosity", "3",
    "--allow-insecure-unlock"
)

# Add mining configuration
if ($mining) {
    $gethArgs += "--mine"
    $gethArgs += "--miner.threads"
    $gethArgs += "1"
} else {
    $gethArgs += "--mine"
    $gethArgs += "--miner.threads"
    $gethArgs += "0"
}

# Start geth
try {
    & "$GethExecutable" $gethArgs
} catch {
    Write-Host " ERROR: Failed to start geth: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host " Q Coin Dev Network stopped." -ForegroundColor Green
