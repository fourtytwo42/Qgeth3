# Test Real Qiskit Quantum Mining - v0.9-rc3-hw0
# This tests the genuine quantum computation using Qiskit-Aer

Write-Host "*** TESTING REAL QISKIT QUANTUM MINING ***" -ForegroundColor Cyan
Write-Host "Quantum-Geth v0.9-rc3-hw0 with genuine Qiskit-Aer computation" -ForegroundColor Green
Write-Host ""

# Clean up any existing data
if (Test-Path "qdata_v09rc3") {
    Write-Host "Cleaning up existing blockchain data..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "qdata_v09rc3"
}

Write-Host "Initializing blockchain with genesis..." -ForegroundColor Yellow
& .\geth.exe --datadir qdata_v09rc3 init genesis_qmpow_v09rc3.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "Genesis initialization failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting Quantum-Geth with real Qiskit quantum mining..." -ForegroundColor Green
Write-Host "This will use genuine quantum circuit execution via Qiskit-Aer!" -ForegroundColor Magenta
Write-Host ""
Write-Host "Expected behavior:" -ForegroundColor Yellow
Write-Host "  * Each quantum puzzle will take several seconds (real quantum computation)" -ForegroundColor Gray
Write-Host "  * You'll see Qiskit execution logs in the background" -ForegroundColor Gray
Write-Host "  * Mining will be much slower but genuinely quantum" -ForegroundColor Gray
Write-Host "  * Press Ctrl+C to stop when you see quantum computation working" -ForegroundColor Gray
Write-Host ""

# Start with reduced parameters for testing
& .\geth.exe --datadir qdata_v09rc3 --networkid 73428 --mine --miner.threads 1 --miner.etherbase 0x8b61271473f14c80f2B1381Db9CB13b2d5306200 --nodiscover --maxpeers 0 --verbosity 4 --http --http.port 8545 --http.api "eth,net,web3,miner,qmpow" --http.corsdomain "*"

Write-Host ""
Write-Host "Real quantum mining test completed." -ForegroundColor Green 