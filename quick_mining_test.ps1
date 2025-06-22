# Quick Mining Test with QNonce Fix
Write-Host "*** QUICK MINING TEST - QNONCE FIX ***" -ForegroundColor Cyan
Write-Host ""

# Clean up previous data
if (Test-Path "qdata") {
    Remove-Item -Recurse -Force "qdata"
}

# Initialize blockchain
Write-Host "Initializing blockchain..." -ForegroundColor Yellow
.\geth.exe --datadir qdata init genesis_qmpow_bitcoin_style.json | Out-Null

# Start mining for 30 seconds
Write-Host "Starting mining for 30 seconds..." -ForegroundColor Green
$miningJob = Start-Job -ScriptBlock {
    param($gethPath)
    & $gethPath --datadir qdata --networkid 1337 --mine --miner.etherbase 0x965e15c0d7fa23fe70d760b380ae60b204f289f2 --miner.threads 1 --nodiscover --maxpeers 0 --verbosity 3 2>&1
} -ArgumentList (Resolve-Path ".\geth.exe")

Start-Sleep -Seconds 30

Write-Host "Stopping mining..." -ForegroundColor Yellow
Stop-Job $miningJob
$output = Receive-Job $miningJob
Remove-Job $miningJob

# Check results
$rlpErrors = $output | Select-String "Invalid block header RLP.*QNonce"
$minedBlocks = $output | Select-String "Bitcoin-style quantum block mined"
$sealedBlocks = $output | Select-String "Successfully sealed new block"

Write-Host ""
Write-Host "=== RESULTS ===" -ForegroundColor Cyan

if ($rlpErrors.Count -gt 0) {
    Write-Host "❌ RLP QNonce errors found:" -ForegroundColor Red
    $rlpErrors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
} else {
    Write-Host "✅ No RLP QNonce errors!" -ForegroundColor Green
}

if ($minedBlocks.Count -gt 0) {
    Write-Host "✅ Mined blocks successfully:" -ForegroundColor Green
    $minedBlocks | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
} else {
    Write-Host "ℹ️  No blocks mined (may need longer time)" -ForegroundColor Yellow
}

if ($sealedBlocks.Count -gt 0) {
    Write-Host "✅ Sealed blocks:" -ForegroundColor Green
    $sealedBlocks | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
}

Write-Host ""
Write-Host "Test completed!" -ForegroundColor Cyan 