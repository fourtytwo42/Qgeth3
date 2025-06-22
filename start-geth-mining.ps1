# Start Geth with Mining - Quantum-Geth v0.9-rc3-hw0
# Starts the quantum geth node with v0.9-rc3-hw0 mining enabled
# Usage: .\start-geth-mining.ps1 -threads 1

param(
    [int]$threads = 1,
    [string]$datadir = "qdata_quantum",
    [int]$networkid = 73428,
    [int]$port = 30303,
    [int]$httpport = 8545,
    [int]$authrpcport = 8551,
    [string]$etherbase = "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    [int]$verbosity = 4,
    [string]$quantumSolver = ".\quantum-geth\tools\solver\qiskit_solver.py"
)

Write-Host "*** STARTING QUANTUM-GETH v0.9-rc3-hw0 MINING NODE ***" -ForegroundColor Cyan
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
Write-Host "  * Quantum Solver: $quantumSolver" -ForegroundColor Magenta
Write-Host ""

Write-Host "Quantum-Geth v0.9-rc3-hw0 Features:" -ForegroundColor Magenta
Write-Host "  * Unified Quantum Blob Architecture (197 bytes)" -ForegroundColor Gray
Write-Host "  * QBits = 12 + glide steps (epochic progression)" -ForegroundColor Gray
Write-Host "  * Fixed 48 puzzles (1,152-bit security)" -ForegroundColor Gray
Write-Host "  * Bitcoin-style QNonce64 iteration" -ForegroundColor Gray
Write-Host "  * Branch-serial template selection" -ForegroundColor Gray
Write-Host "  * Real quantum computation with Qiskit" -ForegroundColor Gray
Write-Host "  * Backward compatibility maintained" -ForegroundColor Gray
Write-Host "  * RLP encoding with quantum blob marshaling" -ForegroundColor Gray
Write-Host ""

# Check if quantum solver exists
if (-not (Test-Path $quantumSolver)) {
    Write-Host "⚠️  Quantum solver '$quantumSolver' not found!" -ForegroundColor Yellow
    Write-Host "Mining will use fallback mode." -ForegroundColor Yellow
}

# Check if data directory exists
if (-not (Test-Path $datadir)) {
    Write-Host "Data directory '$datadir' not found!" -ForegroundColor Red
    Write-Host "Initializing with v0.9-rc3-hw0 genesis..." -ForegroundColor Yellow
    Write-Host "  .\quantum-geth\build\bin\geth.exe --datadir $datadir init genesis_quantum_minimal.json" -ForegroundColor Gray
    
    # Initialize the blockchain
    & .\quantum-geth\build\bin\geth.exe --datadir $datadir init genesis_quantum_minimal.json
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to initialize blockchain!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Blockchain initialized successfully!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Starting v0.9-rc3-hw0 quantum mining with $threads thread(s)..." -ForegroundColor Magenta
Write-Host "Press Ctrl+C to stop mining." -ForegroundColor Yellow
Write-Host ""

# Start geth with mining and quantum solver
& .\quantum-geth\build\bin\geth.exe `
    --datadir $datadir `
    --networkid $networkid `
    --port $port `
    --http `
    --http.addr 127.0.0.1 `
    --http.port $httpport `
    --http.api "eth,net,web3,personal,miner,qmpow" `
    --http.corsdomain "*" `
    --mine `
    --miner.threads $threads `
    --miner.etherbase $etherbase `
    --quantum.solver $quantumSolver `
    --nodiscover `
    --maxpeers 0 `
    --verbosity $verbosity

Write-Host ""
Write-Host "v0.9-rc3-hw0 quantum mining stopped." -ForegroundColor Green 