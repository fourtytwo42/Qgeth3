# Q Coin Windows Startup Script
# Starts Q Coin node (testnet by default, mainnet with -mainnet)
# This geth ONLY connects to Q Coin networks, never Ethereum!
# Usage: .\qcoin-geth.ps1 [-mainnet] [-etherbase <address>] [-port <port>]

param(
    [switch]$mainnet = $false,                 # Use Q Coin mainnet (default: testnet)
    [string]$etherbase = "0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A",  # Mining address
    [int]$port = 30303,                        # P2P network port
    [int]$rpcport = 8545,                      # HTTP RPC port
    [int]$wsport = 8546,                       # WebSocket port
    [switch]$help = $false                     # Show help
)

# Show help
if ($help) {
    Write-Host "Q Coin Node Startup Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\qcoin-geth.ps1 [OPTIONS]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -mainnet           Use Q Coin mainnet (default: testnet)"
    Write-Host "  -etherbase <addr>  Mining reward address"
    Write-Host "  -port <port>       P2P network port (default: 30303)"
    Write-Host "  -rpcport <port>    HTTP RPC port (default: 8545)"
    Write-Host "  -wsport <port>     WebSocket port (default: 8546)"
    Write-Host "  -help              Show this help message"
    Write-Host ""
    Write-Host "Networks:" -ForegroundColor Yellow
    Write-Host "  Testnet (default): Chain ID 73235, genesis_quantum_testnet.json"
    Write-Host "  Mainnet:          Chain ID 73236, genesis_quantum_mainnet.json"
    Write-Host ""
    Write-Host "Note: This geth ONLY connects to Q Coin networks, never Ethereum!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\qcoin-geth.ps1                    # Start testnet"
    Write-Host "  .\qcoin-geth.ps1 -mainnet           # Start mainnet"
    Write-Host "  .\qcoin-geth.ps1 -etherbase 0x123   # Custom mining address"
    exit 0
}

# Network configuration
if ($mainnet) {
    $networkName = "Q Coin Mainnet"
    $chainId = 73236
    $genesisFile = "genesis_quantum_mainnet.json"
    $datadir = "$env:APPDATA\Qcoin\mainnet"
} else {
    $networkName = "Q Coin Testnet"
    $chainId = 73235
    $genesisFile = "genesis_quantum_testnet.json"
    $datadir = "$env:APPDATA\Qcoin"
}

Write-Host "🪙 $networkName - Windows Startup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Find geth executable
$GethExecutable = $null
$searchPaths = @(
    ".\geth.exe",
    ".\releases\quantum-geth-*\geth.exe"
)

foreach ($path in $searchPaths) {
    $found = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($found) {
        $GethExecutable = $found.FullName
        break
    }
}

if (-not $GethExecutable -or -not (Test-Path $GethExecutable)) {
    Write-Host "❌ ERROR: Q Coin geth executable not found!" -ForegroundColor Red
    Write-Host "   Build it first: .\build-release.ps1" -ForegroundColor Yellow
    exit 1
}

# Check genesis file
if (-not (Test-Path $genesisFile)) {
    Write-Host "❌ ERROR: Genesis file not found: $genesisFile" -ForegroundColor Red
    exit 1
}

# Initialize blockchain if needed
if (-not (Test-Path "$datadir\geth\chaindata")) {
    Write-Host "🏗️  Initializing $networkName blockchain..." -ForegroundColor Yellow
    Write-Host "   Data Directory: $datadir" -ForegroundColor Gray
    Write-Host "   Genesis File: $genesisFile" -ForegroundColor Gray
    Write-Host ""
    
    # Create data directory
    if (-not (Test-Path $datadir)) {
        New-Item -ItemType Directory -Path $datadir -Force | Out-Null
    }
    
    # Initialize with Q Coin genesis
    try {
        & "$GethExecutable" --datadir "$datadir" init "$genesisFile" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $networkName blockchain initialized successfully!" -ForegroundColor Green
        } else {
            Write-Host "❌ ERROR: Failed to initialize blockchain!" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "❌ ERROR: Failed to run geth init: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ Blockchain already initialized" -ForegroundColor Green
}

Write-Host ""

# Build geth command with Q Coin configuration
$gethArgs = @(
    "--datadir", "$datadir",
    "--networkid", "$chainId",
    "--port", "$port",
    "--http",
    "--http.addr", "0.0.0.0",
    "--http.port", "$rpcport",
    "--http.api", "eth,net,web3,personal,admin,miner,debug,txpool,qmpow",
    "--http.corsdomain", "*",
    "--ws",
    "--ws.addr", "0.0.0.0",
    "--ws.port", "$wsport",
    "--ws.api", "eth,net,web3,personal,admin,miner,debug,txpool,qmpow",
    "--ws.origins", "*",
    "--nat", "any",
    "--maxpeers", "50",
    "--syncmode", "full",
    "--gcmode", "archive",
    "--mine",
    "--miner.threads", "0",
    "--miner.etherbase", "$etherbase"
)

# Add bootnodes for testnet and mainnet (not dev network)
if ($chainId -ne 73234) {
    $gethArgs += "--bootnodes"
    $gethArgs += "enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.50.254:30303,enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.50.152:30303"
}

# Display startup information
Write-Host "🔧 Node Configuration:" -ForegroundColor Cyan
Write-Host "  Network: $networkName" -ForegroundColor Gray
Write-Host "  Chain ID: $chainId" -ForegroundColor Gray
Write-Host "  Currency: Q (Q Coin)" -ForegroundColor Gray
Write-Host "  Data Directory: $datadir" -ForegroundColor Gray
Write-Host "  P2P Port: $port" -ForegroundColor Gray
Write-Host "  RPC Port: $rpcport" -ForegroundColor Gray
Write-Host "  WebSocket Port: $wsport" -ForegroundColor Gray
Write-Host "  Mining: EXTERNAL MINERS (0 internal threads)" -ForegroundColor Green
Write-Host "  Etherbase: $etherbase" -ForegroundColor Gray
Write-Host ""
Write-Host "🚀 Starting $networkName node..." -ForegroundColor Green
Write-Host "   Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Start geth
try {
    & "$GethExecutable" $gethArgs
} catch {
    Write-Host "❌ ERROR: Failed to start geth: $_" -ForegroundColor Red
    exit 1
} 