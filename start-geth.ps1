# Start Geth (No Mining)
# Starts the quantum geth node without mining enabled

param(
    [string]$datadir = "qdata",
    [int]$networkid = 73428,
    [int]$port = 30303,
    [int]$httpport = 8545,
    [int]$authrpcport = 8551,
    [int]$verbosity = 4
)

Write-Host "*** STARTING QUANTUM GETH NODE (NO MINING) ***" -ForegroundColor Cyan
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  * Data Directory: $datadir" -ForegroundColor Yellow
Write-Host "  * Network ID: $networkid" -ForegroundColor Yellow
Write-Host "  * Port: $port" -ForegroundColor Yellow
Write-Host "  * HTTP Port: $httpport" -ForegroundColor Yellow
Write-Host "  * Auth RPC Port: $authrpcport" -ForegroundColor Yellow
Write-Host "  * Verbosity: $verbosity" -ForegroundColor Yellow
Write-Host "  * Mining: DISABLED" -ForegroundColor Red
Write-Host ""

# Check if data directory exists
if (-not (Test-Path $datadir)) {
    Write-Host "Data directory '$datadir' not found!" -ForegroundColor Red
    Write-Host "Please initialize with genesis first:" -ForegroundColor Yellow
    Write-Host "  .\geth.exe --datadir $datadir init quantum-geth\eth\configs\genesis_qmpow.json" -ForegroundColor Gray
    exit 1
}

Write-Host "Starting geth node..." -ForegroundColor Magenta

# Start geth without mining
& .\geth.exe `
    --datadir $datadir `
    --networkid $networkid `
    --port $port `
    --http `
    --http.port $httpport `
    --http.api "eth,net,web3,miner" `
    --http.corsdomain "*" `
    --authrpc.port $authrpcport `
    --nodiscover `
    --maxpeers 0 `
    --verbosity $verbosity `
    --gcmode archive `
    console

Write-Host ""
Write-Host "Geth node stopped." -ForegroundColor Green 