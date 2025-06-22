# Start Geth with Mining
# Starts the quantum geth node with mining enabled
# Usage: .\start-geth-mining.ps1 -threads 6

param(
    [int]$threads = 1,
    [string]$datadir = "qdata",
    [int]$networkid = 73428,
    [int]$port = 30303,
    [int]$httpport = 8545,
    [int]$authrpcport = 8551,
    [string]$etherbase = "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    [int]$verbosity = 4
)

Write-Host "*** STARTING QUANTUM GETH MINING NODE ***" -ForegroundColor Cyan
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  * Data Directory: $datadir" -ForegroundColor Yellow
Write-Host "  * Network ID: $networkid" -ForegroundColor Yellow
Write-Host "  * Port: $port" -ForegroundColor Yellow
Write-Host "  * HTTP Port: $httpport" -ForegroundColor Yellow
Write-Host "  * Auth RPC Port: $authrpcport" -ForegroundColor Yellow
Write-Host "  * Verbosity: $verbosity" -ForegroundColor Yellow
Write-Host "  * Mining: ENABLED" -ForegroundColor Green
Write-Host "  * Mining Threads: $threads" -ForegroundColor Green
Write-Host "  * Etherbase: $etherbase" -ForegroundColor Yellow
Write-Host ""

Write-Host "Bitcoin-Style Quantum Mining Features:" -ForegroundColor Magenta
Write-Host "  * Fixed 48 quantum puzzles (1,152-bit security)" -ForegroundColor Gray
Write-Host "  * Nonce iteration (0 to 4 billion attempts)" -ForegroundColor Gray
Write-Host "  * Quantum proof quality targeting" -ForegroundColor Gray
Write-Host "  * Bitcoin-style difficulty adjustment" -ForegroundColor Gray
Write-Host "  * Target-based validation" -ForegroundColor Gray
Write-Host ""

# Check if data directory exists
if (-not (Test-Path $datadir)) {
    Write-Host "Data directory '$datadir' not found!" -ForegroundColor Red
    Write-Host "Please initialize with genesis first:" -ForegroundColor Yellow
    Write-Host "  .\geth.exe --datadir $datadir init quantum-geth\eth\configs\genesis_qmpow.json" -ForegroundColor Gray
    exit 1
}

Write-Host "Starting quantum mining with $threads thread(s)..." -ForegroundColor Magenta

# Start geth with mining
& .\geth.exe `
    --datadir $datadir `
    --networkid $networkid `
    --port $port `
    --http `
    --http.port $httpport `
    --http.api "eth,net,web3,miner" `
    --http.corsdomain "*" `
    --authrpc.port $authrpcport `
    --mine `
    --miner.threads $threads `
    --miner.etherbase $etherbase `
    --nodiscover `
    --maxpeers 0 `
    --verbosity $verbosity `
    --gcmode archive `
    console

Write-Host ""
Write-Host "Quantum mining stopped." -ForegroundColor Green 