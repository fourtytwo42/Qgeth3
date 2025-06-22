# Bitcoin-Style Quantum Mining Test
# The World's First Bitcoin-Style Quantum Proof-of-Work Test
# Fixed 48 puzzles (1,152-bit security) + Nonce target + Proof quality

Write-Host "*** BITCOIN-STYLE QUANTUM MINING TEST ***" -ForegroundColor Cyan
Write-Host "True Bitcoin-Style Implementation:" -ForegroundColor Green
Write-Host "  * Fixed 48 puzzles (1,152-bit security)" -ForegroundColor Yellow
Write-Host "  * Nonce iteration (like Bitcoin)" -ForegroundColor Yellow
Write-Host "  * Quantum proof quality targeting" -ForegroundColor Yellow
Write-Host "  * Bitcoin-style difficulty adjustment" -ForegroundColor Yellow
Write-Host "  * Competitive mining with targets" -ForegroundColor Yellow
Write-Host ""

# Clean up previous data
Write-Host "Cleaning up previous blockchain data..." -ForegroundColor Magenta
if (Test-Path "qdata") {
    Remove-Item -Recurse -Force "qdata"
}

# Initialize with Bitcoin-style genesis
Write-Host "Initializing Bitcoin-style quantum blockchain..." -ForegroundColor Magenta
& .\quantum-geth\build\bin\geth.exe --datadir qdata init genesis_qmpow_bitcoin_style.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "Genesis initialization failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Genesis initialized successfully" -ForegroundColor Green
Write-Host ""

# Start Bitcoin-style quantum mining
Write-Host "Starting Bitcoin-style quantum mining..." -ForegroundColor Cyan
Write-Host "Features:" -ForegroundColor White
Write-Host "  * Nonce iteration (0 to 4 billion attempts)" -ForegroundColor Gray
Write-Host "  * Quantum proof quality calculation" -ForegroundColor Gray
Write-Host "  * Target-based validation (like Bitcoin)" -ForegroundColor Gray
Write-Host "  * Difficulty retargeting every 100 blocks" -ForegroundColor Gray
Write-Host "  * Fixed 1,152-bit security (48 puzzles always)" -ForegroundColor Gray
Write-Host ""

# Run Bitcoin-style mining with enhanced logging
& .\quantum-geth\build\bin\geth.exe `
    --datadir qdata `
    --networkid 1337 `
    --mine `
    --miner.etherbase 0x965e15c0d7fa23fe70d760b380ae60b204f289f2 `
    --miner.threads 1 `
    --nodiscover `
    --maxpeers 0 `
    --verbosity 3 `
    --log.debug `
    --rpc.allow-unprotected-txs `
    console

Write-Host ""
Write-Host "Bitcoin-style quantum mining test completed!" -ForegroundColor Green 