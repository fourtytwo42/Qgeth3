# Q Coin Testnet - Start Geth Node
# Starts a Q Coin testnet node with mining support (mining disabled by default)
# Uses default blockchain location like standard geth
# Usage: .\start-geth.ps1 [-mine] [-etherbase <address>]

param(
    [switch]$mine = $false,                    # Enable mining
    [string]$etherbase = "",                   # Mining address (optional)
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
    Write-Host "  -mine              Enable mining (disabled by default)"
    Write-Host "  -etherbase <addr>  Mining reward address (required if mining)"
    Write-Host "  -port <port>       P2P network port (default: 4294 testnet, 4295 mainnet)"
    Write-Host "  -rpcport <port>    HTTP-RPC server port (default: 8545)"
    Write-Host "  -wsport <port>     WebSocket server port (default: 8546)"
    Write-Host "  -datadir <path>    Custom data directory (default: system default)"
    Write-Host "  -mainnet           Use mainnet instead of testnet"
    Write-Host "  -help              Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\start-geth.ps1                                    # Start node (no mining)"
    Write-Host "  .\start-geth.ps1 -mine -etherbase 0x123...         # Start with mining"
    Write-Host ""
    Write-Host "Q Coin Testnet Details:" -ForegroundColor Magenta
    Write-Host "  Chain ID: 73235"
    Write-Host "  Currency: Q (Q Coin)"
    Write-Host "  Block Time: 12 seconds"
    Write-Host "  Consensus: QMPoW (Quantum Proof of Work)"
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

# Validate mining parameters
if ($mine -and $etherbase -eq "") {
    Write-Host "ERROR: Mining requires an etherbase address!" -ForegroundColor Red
    Write-Host "Use: .\start-geth.ps1 -mine -etherbase <your_address>" -ForegroundColor Yellow
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

# Build geth command
$gethArgs = @(
    "--datadir", "$datadir",
    "--networkid", "$chainId",
    "--port", "$port",
    "--http",
    "--http.addr", "0.0.0.0",
    "--http.port", "$rpcport",
    "--http.api", "eth,net,web3,personal,admin,miner,debug,txpool",
    "--http.corsdomain", "*",
    "--ws",
    "--ws.addr", "0.0.0.0", 
    "--ws.port", "$wsport",
    "--ws.api", "eth,net,web3,personal,admin,miner,debug,txpool",
    "--ws.origins", "*",
    "--nat", "any",
    "--maxpeers", "50",
    "--allow-insecure-unlock",
    "--syncmode", "full",
    "--gcmode", "archive"
)

# Add mining parameters if enabled
if ($mine) {
    $gethArgs += @("--mine", "--miner.etherbase", "$etherbase")
    Write-Host "Mining enabled - rewards go to: $etherbase" -ForegroundColor Yellow
}

# Display startup information
Write-Host ""
Write-Host "$networkName Configuration:" -ForegroundColor Cyan
Write-Host "  Chain ID: $chainId" -ForegroundColor Gray
Write-Host "  Currency: Q (Q Coin)" -ForegroundColor Gray
Write-Host "  Data Directory: $datadir" -ForegroundColor Gray
Write-Host "  P2P Port: $port" -ForegroundColor Gray
Write-Host "  RPC Port: $rpcport" -ForegroundColor Gray
Write-Host "  WebSocket Port: $wsport" -ForegroundColor Gray
Write-Host "  Mining: $(if ($mine) { 'ENABLED' } else { 'DISABLED' })" -ForegroundColor $(if ($mine) { 'Green' } else { 'Yellow' })
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