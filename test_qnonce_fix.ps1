# Test QNonce RLP Fix
Write-Host "*** TESTING QNONCE RLP FIX ***" -ForegroundColor Cyan
Write-Host "Bitcoin-style QuantumNonce implementation test" -ForegroundColor Green
Write-Host ""

# Clean up previous data
Write-Host "Cleaning up previous blockchain data..." -ForegroundColor Magenta
if (Test-Path "qdata") {
    Remove-Item -Recurse -Force "qdata"
}

# Initialize with Bitcoin-style genesis
Write-Host "Initializing Bitcoin-style quantum blockchain..." -ForegroundColor Magenta
.\geth.exe --datadir qdata init genesis_qmpow_bitcoin_style.json
if ($LASTEXITCODE -ne 0) {
    Write-Host "Genesis initialization failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Genesis initialized successfully" -ForegroundColor Green
Write-Host ""

# Start mining for a short test (10 seconds)
Write-Host "Starting short mining test to verify QNonce RLP fix..." -ForegroundColor Cyan
Write-Host "Will run for 10 seconds then exit automatically" -ForegroundColor Yellow
Write-Host ""

# Start mining in background
$miningJob = Start-Job -ScriptBlock {
    param($gethPath)
    & $gethPath --datadir qdata --networkid 1337 --mine --miner.etherbase 0x965e15c0d7fa23fe70d760b380ae60b204f289f2 --miner.threads 1 --nodiscover --maxpeers 0 --verbosity 3 --log.debug --rpc.allow-unprotected-txs 2>&1
} -ArgumentList (Resolve-Path ".\geth.exe")

# Wait 10 seconds
Start-Sleep -Seconds 10

# Stop mining
Write-Host "Stopping mining test..." -ForegroundColor Yellow
Stop-Job $miningJob
$output = Receive-Job $miningJob
Remove-Job $miningJob

# Check for RLP errors
$rlpErrors = $output | Select-String "Invalid block header RLP.*QNonce"
if ($rlpErrors.Count -gt 0) {
    Write-Host "RLP QNonce errors still present:" -ForegroundColor Red
    $rlpErrors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
} else {
    Write-Host "No RLP QNonce errors detected!" -ForegroundColor Green
}

# Check for successful mining
$successfulBlocks = $output | Select-String "Bitcoin-style quantum block mined"
if ($successfulBlocks.Count -gt 0) {
    Write-Host "Successfully mined blocks with Bitcoin-style QNonce!" -ForegroundColor Green
    $successfulBlocks | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
} else {
    Write-Host "No blocks mined in test period (normal for short test)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "QNonce RLP fix test completed!" -ForegroundColor Green 