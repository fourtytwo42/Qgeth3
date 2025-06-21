# Test quantum mining for 20 seconds
Write-Host "Starting quantum mining test..." -ForegroundColor Green

# Start geth process
$process = Start-Process -FilePath "./geth.exe" -ArgumentList @(
    "--datadir", "qdata",
    "--networkid", "73428",
    "--mine",
    "--miner.threads", "1",
    "--miner.etherbase", "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    "--unlock", "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    "--password", "qdata/password.txt",
    "--http",
    "--http.api", "eth,net,web3,miner,admin,debug,qmpow",
    "--http.corsdomain", "*",
    "--allow-insecure-unlock",
    "--verbosity", "4",
    "--nodiscover",
    "--maxpeers", "0"
) -PassThru -NoNewWindow

Write-Host "Started geth process ID: $($process.Id)" -ForegroundColor Yellow

# Wait 5 seconds for startup
Start-Sleep 5

Write-Host "Checking block number every 3 seconds..." -ForegroundColor Cyan

# Check block number for 20 seconds
$startTime = Get-Date
while ((Get-Date) - $startTime -lt [TimeSpan]::FromSeconds(20)) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8545" -Method POST -Body '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' -ContentType "application/json" -TimeoutSec 2
        $result = $response.Content | ConvertFrom-Json
        $blockHex = $result.result
        $blockNumber = [Convert]::ToInt32($blockHex, 16)
        Write-Host "Block: $blockNumber (hex: $blockHex)" -ForegroundColor Green
    }
    catch {
        Write-Host "HTTP request failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Start-Sleep 3
}

Write-Host "Test complete. Stopping geth..." -ForegroundColor Yellow

# Stop the process
Stop-Process -Id $process.Id -Force
Write-Host "Done." -ForegroundColor Green 