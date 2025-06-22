# Simple Quantum Blockchain Test
Write-Host "🔬 Testing Quantum Blockchain" -ForegroundColor Cyan

# Stop any running processes
Stop-Process -Name "geth" -Force -ErrorAction SilentlyContinue

# Clean and reinitialize
Remove-Item -Path "qdata/geth/chaindata/*" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "⚡ Initializing blockchain..."
& ./geth.exe --datadir qdata init quantum-geth/eth/configs/genesis_qmpow.json

# Start blockchain in background
Write-Host "🚀 Starting quantum mining..."
$gethProcess = Start-Process -FilePath "./geth.exe" -ArgumentList @(
    "--datadir", "qdata",
    "--mine",
    "--miner.threads", "1", 
    "--unlock", "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    "--password", "qdata/password.txt",
    "--allow-insecure-unlock",
    "--networkid", "73428",
    "--nodiscover",
    "--maxpeers", "0",
    "--verbosity", "4",
    "--http",
    "--http.port", "8545",
    "--http.addr", "localhost",
    "--http.api", "eth,net,web3,miner"
) -PassThru -NoNewWindow

Write-Host "⏳ Waiting for startup..."
Start-Sleep -Seconds 10

# Check block progression
Write-Host "🔍 Checking block progression..."
$blockNumbers = @()

for ($i = 1; $i -le 20; $i++) {
    try {
        $body = '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
        $result = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 3
        $blockHex = $result.result
        $blockNumber = [Convert]::ToInt32($blockHex, 16)
        $blockNumbers += $blockNumber
        Write-Host "Check $i`: Block $blockNumber" -ForegroundColor Green
    }
    catch {
        Write-Host "Check $i`: RPC failed" -ForegroundColor Yellow
    }
    Start-Sleep -Seconds 3
}

# Cleanup
$gethProcess.Kill()
$gethProcess.WaitForExit(5000)

# Results
$uniqueBlocks = $blockNumbers | Sort-Object -Unique
Write-Host "`n📊 Results:" -ForegroundColor Cyan
Write-Host "Block progression: $($uniqueBlocks -join ' -> ')" -ForegroundColor White

if ($uniqueBlocks.Count -ge 3) {
    Write-Host "🎉 SUCCESS: Multiple blocks mined!" -ForegroundColor Green
    Write-Host "✅ Quantum blockchain working correctly!" -ForegroundColor Green
} elseif ($uniqueBlocks -contains 1) {
    Write-Host "⚠️ WARNING: Only block 1 mined repeatedly" -ForegroundColor Yellow
} else {
    Write-Host "FAILURE: No progression detected" -ForegroundColor Red
} 