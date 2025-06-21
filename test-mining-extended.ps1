# Extended quantum mining test - 60 seconds
Write-Host "Starting EXTENDED quantum mining test (60 seconds)..." -ForegroundColor Green

# Start geth process
$process = Start-Process -FilePath "./geth.exe" -ArgumentList @(
    "--datadir", "qdata",
    "--networkid", "73428",
    "--mine",
    "--miner.threads", "2",
    "--miner.etherbase", "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    "--unlock", "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    "--password", "qdata/password.txt",
    "--http",
    "--http.api", "eth,net,web3,miner,admin,debug,qmpow",
    "--http.corsdomain", "*",
    "--allow-insecure-unlock",
    "--verbosity", "3",
    "--nodiscover",
    "--maxpeers", "0"
) -PassThru -NoNewWindow

Write-Host "Started geth process ID: $($process.Id)" -ForegroundColor Yellow

# Wait 8 seconds for startup
Write-Host "Waiting for startup..." -ForegroundColor Cyan
Start-Sleep 8

Write-Host "Monitoring block production every 2 seconds for 60 seconds..." -ForegroundColor Cyan

# Check block number for 60 seconds
$startTime = Get-Date
$lastBlock = -1
$blockCount = 0

while ((Get-Date) - $startTime -lt [TimeSpan]::FromSeconds(60)) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8545" -Method POST -Body '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' -ContentType "application/json" -TimeoutSec 3
        $result = $response.Content | ConvertFrom-Json
        $blockHex = $result.result
        $blockNumber = [Convert]::ToInt32($blockHex, 16)
        
        if ($blockNumber -ne $lastBlock) {
            if ($lastBlock -ge 0) {
                $blockCount++
                Write-Host "NEW BLOCK #$blockNumber mined! (Total blocks mined: $blockCount)" -ForegroundColor Green
            } else {
                Write-Host "Starting at block: $blockNumber" -ForegroundColor Yellow
            }
            $lastBlock = $blockNumber
        } else {
            Write-Host "Block: $blockNumber (waiting for next block...)" -ForegroundColor White
        }
    }
    catch {
        Write-Host "HTTP request failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Start-Sleep 2
}

Write-Host "" -ForegroundColor White
Write-Host "MINING TEST COMPLETE!" -ForegroundColor Green
Write-Host "Final block number: $lastBlock" -ForegroundColor Green  
Write-Host "Total blocks mined during test: $blockCount" -ForegroundColor Green
Write-Host "Mining rate: $($blockCount/60.0) blocks per second" -ForegroundColor Green

Write-Host "Stopping geth..." -ForegroundColor Yellow
Stop-Process -Id $process.Id -Force
Write-Host "Done." -ForegroundColor Green 