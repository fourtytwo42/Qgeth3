# Start Geth (No Mining) - Quantum-Geth v0.9-rc3-hw0
# Starts the quantum geth node without mining enabled

param(
    [string]$datadir = "qdata_quantum",
    [int]$networkid = 73428,
    [int]$port = 30303,
    [int]$httpport = 8545,
    [int]$authrpcport = 8551,
    [int]$verbosity = 4
)

Write-Host "*** STARTING QUANTUM-GETH v0.9-rc3-hw0 NODE (NO MINING) ***" -ForegroundColor Cyan
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  * Data Directory: $datadir" -ForegroundColor Yellow
Write-Host "  * Network ID: $networkid" -ForegroundColor Yellow
Write-Host "  * Port: $port" -ForegroundColor Yellow
Write-Host "  * HTTP Port: $httpport" -ForegroundColor Yellow
Write-Host "  * Auth RPC Port: $authrpcport" -ForegroundColor Yellow
Write-Host "  * Verbosity: $verbosity" -ForegroundColor Yellow
Write-Host "  * Mining: DISABLED" -ForegroundColor Red
Write-Host ""

Write-Host "Quantum-Geth v0.9-rc3-hw0 Features:" -ForegroundColor Magenta
Write-Host "  * Unified Quantum Blob Architecture (197 bytes)" -ForegroundColor Gray
Write-Host "  * QBits = 12 + glide steps (epochic progression)" -ForegroundColor Gray
Write-Host "  * Fixed 48 puzzles (1,152-bit security)" -ForegroundColor Gray
Write-Host "  * Branch-serial template selection" -ForegroundColor Gray
Write-Host "  * Real quantum computation with Qiskit" -ForegroundColor Gray
Write-Host "  * Backward compatibility maintained" -ForegroundColor Gray
Write-Host "  * RLP encoding with quantum blob marshaling" -ForegroundColor Gray
Write-Host ""

# Check if data directory exists
if (-not (Test-Path $datadir)) {
    Write-Host "Data directory '$datadir' not found!" -ForegroundColor Red
    Write-Host "Initializing with v0.9-rc3-hw0 genesis..." -ForegroundColor Yellow
    Write-Host "  .\quantum-geth\build\bin\geth.exe --datadir $datadir init genesis_quantum.json" -ForegroundColor Gray
    
    # Initialize the blockchain
    & .\quantum-geth\build\bin\geth.exe --datadir $datadir init genesis_quantum.json
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to initialize blockchain!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Blockchain initialized successfully!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Starting v0.9-rc3-hw0 geth node..." -ForegroundColor Magenta
Write-Host "Press Ctrl+C to stop the node." -ForegroundColor Yellow
Write-Host ""

# Start geth without mining
& .\quantum-geth\build\bin\geth.exe `
    --datadir $datadir `
    --networkid $networkid `
    --port $port `
    --http `
    --http.addr 127.0.0.1 `
    --http.port $httpport `
    --http.api "eth,net,web3,personal,miner,qmpow" `
    --http.corsdomain "*" `
    --authrpc.port $authrpcport `
    --nodiscover `
    --maxpeers 0 `
    --verbosity $verbosity

Write-Host ""
Write-Host "v0.9-rc3-hw0 geth node stopped." -ForegroundColor Green 