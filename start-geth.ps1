# Start Geth (No Mining) - Quantum-Geth v0.9 BareBones+Halving
# Starts the quantum geth node without mining enabled - serves work to external miners

param(
    [string]$datadir = "qdata",
    [int]$networkid = 73428,
    [int]$port = 30303,
    [int]$httpport = 8545,
    [int]$authrpcport = 8551,
    [string]$etherbase = "0x1234567890123456789012345678901234567890",
    [int]$verbosity = 3
)

Write-Host "*** STARTING QUANTUM-GETH v0.9 BareBones+Halving NODE (EXTERNAL MINING) ***" -ForegroundColor Cyan
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  * Data Directory: $datadir" -ForegroundColor Yellow
Write-Host "  * Network ID: $networkid" -ForegroundColor Yellow
Write-Host "  * Port: $port" -ForegroundColor Yellow
Write-Host "  * HTTP Port: $httpport" -ForegroundColor Yellow
Write-Host "  * Auth RPC Port: $authrpcport" -ForegroundColor Yellow
Write-Host "  * Etherbase: $etherbase" -ForegroundColor Yellow
Write-Host "  * Verbosity: $verbosity" -ForegroundColor Yellow
Write-Host "  * Mining: EXTERNAL MINERS ONLY (0 internal threads)" -ForegroundColor Green
Write-Host ""

Write-Host "Quantum-Geth v0.9 BareBones+Halving Features:" -ForegroundColor Magenta
Write-Host "  * 48 sequential quantum puzzles (16 qubits x 8,192 T-gates)" -ForegroundColor Gray
Write-Host "  * Bitcoin-style halving (50 QGC -> 25 QGC -> 12.5 QGC...)" -ForegroundColor Gray
Write-Host "  * 600,000 block epochs (approximately 6 months)" -ForegroundColor Gray
Write-Host "  * Branch-serial quantum circuit execution" -ForegroundColor Gray
Write-Host "  * Mahadev->CAPSS->Nova proof stack" -ForegroundColor Gray
Write-Host "  * Dilithium-2 self-attestation" -ForegroundColor Gray
Write-Host "  * ASERT-Q difficulty adjustment (12s target)" -ForegroundColor Gray
Write-Host "  * Single RLP quantum blob (197 bytes)" -ForegroundColor Gray
Write-Host ""

# Check if data directory exists
if (-not (Test-Path $datadir)) {
    Write-Host "Data directory '$datadir' not found!" -ForegroundColor Red
    Write-Host "Initializing with v0.9 BareBones+Halving genesis..." -ForegroundColor Yellow
    Write-Host "  .\reset-blockchain.ps1 -difficulty 1 -force" -ForegroundColor Gray
    
    Write-Host ""
    Write-Host "Please run the reset script first to initialize the blockchain:" -ForegroundColor Yellow
    Write-Host "   .\reset-blockchain.ps1 -difficulty 1 -force" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "Starting v0.9 BareBones+Halving geth node for external miners..." -ForegroundColor Magenta
Write-Host "This node enables mining infrastructure with 0 threads for external miners." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the node." -ForegroundColor Yellow
Write-Host ""

# Start geth WITH mining enabled but NO mining threads (for external miners)
& .\geth.exe `
    --datadir $datadir `
    --networkid $networkid `
    --port $port `
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
    --nodiscover `
    --maxpeers 0 `
    --verbosity $verbosity `
    --log.vmodule "rpc=1" `
    --allow-insecure-unlock

Write-Host ""
Write-Host "v0.9 BareBones+Halving geth node stopped." -ForegroundColor Green 