# Start Quantum-Geth v0.9-rc3-hw0 with Mining
# Quantum-Geth v0.9-rc3-hw0 — Unified, Branch-Serial Quantum Proof-of-Work
# Usage: .\start-geth-v09rc3.ps1 -threads 6

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

Write-Host "*** QUANTUM-GETH v0.9-rc3-hw0 MINING NODE ***" -ForegroundColor Cyan
Write-Host "Unified, Branch-Serial Quantum Proof-of-Work — Canonical-Compile Edition" -ForegroundColor Green
Write-Host ""
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

Write-Host "v0.9-rc3-hw0 Quantum Features:" -ForegroundColor Magenta
Write-Host "  * Epochic n-qubit glide (starts at 12, +1 every 12,500 blocks)" -ForegroundColor Gray
Write-Host "  * Fixed 48 quantum puzzles (1,152-bit security)" -ForegroundColor Gray
Write-Host "  * Branch-dependent template selection" -ForegroundColor Gray
Write-Host "  * Canonical compiler with gate hash verification" -ForegroundColor Gray
Write-Host "  * Tier-A/B/C proof stack with Nova batching" -ForegroundColor Gray
Write-Host "  * Deterministic Dilithium attestation" -ForegroundColor Gray
Write-Host "  * Bitcoin-style nonce iteration (0 to 4 billion)" -ForegroundColor Gray
Write-Host ""

# Check if data directory exists
if (-not (Test-Path $datadir)) {
    Write-Host "Data directory '$datadir' not found!" -ForegroundColor Red
    Write-Host "Initializing with v0.9-rc3-hw0 genesis..." -ForegroundColor Yellow
    
    & .\geth.exe --datadir $datadir init genesis_qmpow_v09rc3.json
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Genesis initialization failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Genesis initialized successfully" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Starting Quantum-Geth v0.9-rc3-hw0 mining with $threads thread(s)..." -ForegroundColor Magenta

# Start geth with mining
& .\geth.exe `
    --datadir $datadir `
    --networkid $networkid `
    --port $port `
    --http `
    --http.port $httpport `
    --http.api "eth,net,web3,miner,qmpow" `
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
Write-Host "Quantum-Geth v0.9-rc3-hw0 mining stopped." -ForegroundColor Green 