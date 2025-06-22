# Start Quantum-Geth Mining Script
# Real Quantum Proof-of-Work Blockchain - Local Only

Write-Host "Starting Quantum-Geth with Real Qiskit Integration (Local Only)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if data directory exists
if (-not (Test-Path "qdata_quantum")) {
    Write-Host "Data directory not found. Initializing..." -ForegroundColor Yellow
    .\geth.exe --datadir qdata_quantum init genesis_quantum.json
    Write-Host "Genesis block initialized" -ForegroundColor Green
}

# Start Quantum-Geth - Completely Local/Isolated
Write-Host "Starting Quantum-Geth mining (Local Only Mode)..." -ForegroundColor Green
.\geth.exe --datadir qdata_quantum `
    --networkid 73428 `
    --mine `
    --miner.threads 1 `
    --miner.etherbase 0x8b61271473f14c80f2B1381Db9CB13b2d5306200 `
    --nodiscover `
    --maxpeers 0 `
    --netrestrict "127.0.0.1/32" `
    --nat none `
    --port 0 `
    --verbosity 4 `
    --http `
    --http.port 8545 `
    --http.addr "127.0.0.1" `
    --http.api "eth,net,web3,miner,qmpow" `
    --http.corsdomain "localhost" 