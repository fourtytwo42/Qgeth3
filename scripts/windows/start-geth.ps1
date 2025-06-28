# Q Coin Geth Node Starter
# Usage: ./start-geth.ps1 [network] [options]
# Networks: mainnet, testnet, devnet (default: testnet)
# Options: -mining (enable mining with single thread)

param(
    [Parameter(Position=0)]
    [ValidateSet("mainnet", "testnet", "devnet")]
    [string]$Network = "testnet",
    
    [switch]$Mining,
    [switch]$Help,
    
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

if ($Help) {
    Write-Host "Q Coin Geth Node Starter" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: ./start-geth.ps1 [network] [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Networks:" -ForegroundColor Yellow
    Write-Host "  mainnet   - Q Coin Mainnet (Chain ID 73236)"
    Write-Host "  testnet   - Q Coin Testnet (Chain ID 73235) [DEFAULT]"
    Write-Host "  devnet    - Q Coin Dev Network (Chain ID 73234)"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -mining   - Enable mining with single thread"
    Write-Host "  -help     - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ./start-geth.ps1                 # Start testnet node"
    Write-Host "  ./start-geth.ps1 mainnet         # Start mainnet node"
    Write-Host "  ./start-geth.ps1 devnet -mining  # Start dev node with mining"
    exit 0
}

# Find latest geth release
function Get-LatestGethRelease {
    $gethReleases = Get-ChildItem "releases" -Directory | Where-Object { $_.Name -like "quantum-geth-*" } | Sort-Object Name -Descending
    if ($gethReleases.Count -eq 0) {
        return $null
    }
    return Join-Path $gethReleases[0].FullName "geth.exe"
}

# Build if latest geth release doesn't exist
$latestGeth = Get-LatestGethRelease
if (-not $latestGeth -or -not (Test-Path $latestGeth)) {
    Write-Host "Building Q Coin Geth Release..." -ForegroundColor Yellow
    & .\build-release.ps1 geth
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed!" -ForegroundColor Red
        exit 1
    }
    $latestGeth = Get-LatestGethRelease
    if (-not $latestGeth) {
        Write-Host "ERROR: No geth release found after build!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "* Using latest geth release: $latestGeth" -ForegroundColor Green

# Network configurations - bootnodes auto-selected based on chainid
$configs = @{
    "mainnet" = @{
        chainid = 73236
        datadir = "$env:APPDATA\Qcoin\mainnet"
        genesis = "genesis_quantum_mainnet.json"
        port = 30303
        name = "Q Coin Mainnet"
    }
    "testnet" = @{
        chainid = 73235
        datadir = "$env:APPDATA\Qcoin\testnet"
        genesis = "genesis_quantum_testnet.json"
        port = 30303
        name = "Q Coin Testnet"
    }
    "devnet" = @{
        chainid = 73234
        datadir = "$env:APPDATA\Qcoin\devnet"
        genesis = "genesis_quantum_dev.json"
        port = 30305
        name = "Q Coin Dev Network"
    }
}

$config = $configs[$Network]
Write-Host "Starting $($config.name) (Chain ID: $($config.chainid))" -ForegroundColor Cyan

# Create data directory if it doesn't exist
if (-not (Test-Path $config.datadir)) {
    New-Item -ItemType Directory -Path $config.datadir -Force | Out-Null
    Write-Host "* Created data directory: $($config.datadir)" -ForegroundColor Green
}

# Initialize with genesis if needed
$genesisPath = Join-Path $config.datadir "geth\chaindata"
if (-not (Test-Path $genesisPath)) {
    Write-Host "Initializing blockchain with genesis file..." -ForegroundColor Yellow
    & $latestGeth --datadir $config.datadir init $config.genesis
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Genesis initialization failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "* Blockchain initialized successfully" -ForegroundColor Green
}

# Prepare geth arguments
$gethArgs = @(
    "--datadir", $config.datadir,
    "--networkid", $config.chainid,
    "--port", $config.port,
    "--nat", "any",
    "--http",
    "--http.addr", "0.0.0.0",
    "--http.port", "8545",
    "--http.corsdomain", "*",
    "--http.api", "eth,net,web3,personal,admin,txpool,miner,qmpow",
    "--ws",
    "--ws.addr", "0.0.0.0", 
    "--ws.port", "8546",
    "--ws.origins", "*",
    "--ws.api", "eth,net,web3,personal,admin,txpool,miner,qmpow",
    "--maxpeers", "25",
    "--verbosity", "3"
)

# Add mining if requested
if ($Mining) {
    $gethArgs += @("--mine", "--miner.threads", "1", "--miner.etherbase", "0x0000000000000000000000000000000000000001")
    Write-Host "Mining enabled with 1 thread" -ForegroundColor Yellow
    Write-Host "NOTE: Use miner_setEtherbase RPC call to set your actual mining address" -ForegroundColor Cyan
} else {
    # Enable mining interface for external miners (0 threads = no CPU mining, external only)
    $gethArgs += @("--mine", "--miner.threads", "0", "--miner.etherbase", "0x0000000000000000000000000000000000000001")
    Write-Host "Mining interface enabled for external miners (no CPU mining)" -ForegroundColor Green
    Write-Host "NOTE: External miners will set their own coinbase addresses via RPC" -ForegroundColor Cyan
}

# Add any extra arguments passed to the script
if ($ExtraArgs) {
    $gethArgs += $ExtraArgs
    Write-Host "Extra arguments: $($ExtraArgs -join ' ')" -ForegroundColor Cyan
}

Write-Host "Network: $($config.name)" -ForegroundColor White
Write-Host "Chain ID: $($config.chainid)" -ForegroundColor White
Write-Host "Data Directory: $($config.datadir)" -ForegroundColor White
Write-Host "Port: $($config.port)" -ForegroundColor White
Write-Host "NAT: Automatic discovery (UPnP/NAT-PMP)" -ForegroundColor White
Write-Host "Bootnodes: Auto-selected for $Network network" -ForegroundColor White
Write-Host ""
Write-Host "Starting Q Coin Geth node..." -ForegroundColor Green

# Start geth
& $latestGeth @gethArgs 
