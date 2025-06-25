# Q Coin Testnet - Start Geth Node
# Starts a Q Coin testnet node with external mining enabled (0 internal threads)
# Uses default blockchain location like standard geth
# Usage: .\start-geth.ps1 [-etherbase <address>]

param(
    [string]$etherbase = "0x1234567890123456789012345678901234567890", # Mining address (default provided)
    [int]$port = 4294,                         # P2P port (Q Coin testnet)
    [int]$rpcport = 8545,                      # RPC port
    [int]$wsport = 8546,                       # WebSocket port
    [string]$datadir = "",                     # Data directory (empty = default)
    [switch]$mainnet = $false,                 # Use mainnet (default: testnet)
    [switch]$help = $false                     # Show help
)

# Show help
if ($help) {
    Write-Host "Q Coin Testnet - Geth Node" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\start-geth.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Green
    Write-Host "  -etherbase <addr>  Mining reward address (default: 0x1234...)"
    Write-Host "  -port <port>       P2P network port (default: 4294 testnet, 4295 mainnet)"
    Write-Host "  -rpcport <port>    HTTP-RPC server port (default: 8545)"
    Write-Host "  -wsport <port>     WebSocket server port (default: 8546)"
    Write-Host "  -datadir <path>    Custom data directory (default: system default)"
    Write-Host "  -mainnet           Use mainnet instead of testnet"
    Write-Host "  -help              Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\start-geth.ps1                                    # Start with external mining"
    Write-Host "  .\start-geth.ps1 -etherbase 0x123...               # Custom mining address"
    Write-Host ""
    Write-Host "Q Coin Testnet Details:" -ForegroundColor Magenta
    Write-Host "  Chain ID: 73235"
    Write-Host "  Currency: Q (Q Coin)"
    Write-Host "  Block Time: 12 seconds"
    Write-Host "  Consensus: QMPoW (Quantum Proof of Work)"
    Write-Host "  Mining: External miners only (0 internal threads)"
    Write-Host "  Puzzles: 128 chained quantum puzzles per block"
    exit 0
}

# Determine network configuration
if ($mainnet) {
    $networkName = "Q Coin Mainnet"
    $chainId = "73236"
    $genesisFile = "genesis_quantum_mainnet.json"
    $defaultDataDir = "$env:APPDATA\Qcoin\mainnet"
    if ($port -eq 4294) { $port = 4295 }  # Switch to mainnet port if using default
} else {
    $networkName = "Q Coin Testnet"
    $chainId = "73235"
    $genesisFile = "genesis_quantum_testnet.json"
    $defaultDataDir = "$env:APPDATA\Qcoin\testnet"
}

# Validate etherbase address format (basic check)
if ($etherbase -notmatch "^0x[0-9a-fA-F]{40}$") {
    Write-Host "ERROR: Invalid etherbase address format!" -ForegroundColor Red
    Write-Host "Expected format: 0x followed by 40 hex characters" -ForegroundColor Yellow
    Write-Host "Example: 0x1234567890123456789012345678901234567890" -ForegroundColor Yellow
    exit 1
}

# Find the latest quantum-geth release
Write-Host "$networkName - Starting Geth Node" -ForegroundColor Cyan
Write-Host ""

$GethReleaseDir = Get-ChildItem -Path "releases\quantum-geth-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
if (-not $GethReleaseDir) {
    Write-Host "ERROR: No quantum-geth release found!" -ForegroundColor Red
    Write-Host "Please run: .\build-release.ps1 geth" -ForegroundColor Yellow
    exit 1
}

$GethExecutable = "$($GethReleaseDir.FullName)\geth.exe"
Write-Host "Using geth from: $($GethReleaseDir.Name)" -ForegroundColor Green

# Determine data directory
if ($datadir -eq "") {
    # Use network-specific default directory
    $datadir = $defaultDataDir
    Write-Host "Using default data directory: $datadir" -ForegroundColor Green
} else {
    Write-Host "Using custom data directory: $datadir" -ForegroundColor Green
}

# Check if blockchain is initialized
if (-not (Test-Path "$datadir\geth\chaindata")) {
    Write-Host ""
    Write-Host "Initializing $networkName blockchain..." -ForegroundColor Yellow
    
    # Create data directory if it doesn't exist
    if (-not (Test-Path $datadir)) {
        New-Item -ItemType Directory -Path $datadir -Force | Out-Null
    }
    
    # Initialize with testnet genesis
    try {
        & "$GethExecutable" --datadir "$datadir" init "$genesisFile" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$networkName blockchain initialized successfully!" -ForegroundColor Green
        } else {
            Write-Host "ERROR: Failed to initialize blockchain!" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "ERROR: Failed to run geth init: $_" -ForegroundColor Red
        exit 1
    }
}

# Build geth command with external mining enabled
# For other nodes to connect to this one, use:
# --bootnodes "enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@69.243.132.233:4294"
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

# Display startup information
Write-Host ""
Write-Host "$networkName Configuration:" -ForegroundColor Cyan
Write-Host "  Chain ID: $chainId" -ForegroundColor Gray
Write-Host "  Currency: Q (Q Coin)" -ForegroundColor Gray
Write-Host "  Data Directory: $datadir" -ForegroundColor Gray
Write-Host "  P2P Port: $port" -ForegroundColor Gray
Write-Host "  RPC Port: $rpcport" -ForegroundColor Gray
Write-Host "  WebSocket Port: $wsport" -ForegroundColor Gray
Write-Host "  Mining: EXTERNAL MINERS (0 internal threads)" -ForegroundColor Green
Write-Host "  Etherbase: $etherbase" -ForegroundColor Gray
Write-Host ""
Write-Host "Starting $networkName node..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Start geth
try {
    & "$GethExecutable" $gethArgs
} catch {
    Write-Host "ERROR: Failed to start geth: $_" -ForegroundColor Red
    exit 1
} 