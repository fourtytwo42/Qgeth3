# Monitor Realistic Quantum Mining Progress

Write-Host "📊 Monitoring Realistic Quantum Mining Progress..." -ForegroundColor Cyan
Write-Host "Target: 12-second block times" -ForegroundColor Yellow

$startTime = Get-Date
$startBlock = ./geth.exe --datadir ./qdata attach --exec "eth.blockNumber" 2>$null
$startBlock = [int]$startBlock.Trim()
Write-Host "Starting monitoring at block: $startBlock" -ForegroundColor Green

$previousBlock = $startBlock
$previousTime = $startTime

Write-Host ""
Write-Host "📈 Block Time Analysis:" -ForegroundColor Cyan
Write-Host "Block | Time   | Block Time | Status" -ForegroundColor Gray
Write-Host "------|--------|------------|--------" -ForegroundColor Gray

for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 5
    
    $currentTime = Get-Date
    $currentBlock = ./geth.exe --datadir ./qdata attach --exec "eth.blockNumber" 2>$null
    if ($currentBlock) {
        $currentBlock = [int]$currentBlock.Trim()
        
        if ($currentBlock -gt $previousBlock) {
            # New block found
            $timeSinceLastBlock = ($currentTime - $previousTime).TotalSeconds
            $blocksMinedSinceLastCheck = $currentBlock - $previousBlock
            $averageBlockTime = $timeSinceLastBlock / $blocksMinedSinceLastCheck
            
            # Determine status based on block time
            $status = "OK"
            if ($averageBlockTime -lt 10) {
                $status = "FAST"
            } elseif ($averageBlockTime -gt 14) {
                $status = "SLOW" 
            }
            
            $timeStr = $averageBlockTime.ToString("F1") + "s"
            
            $color = "Green"
            if ($status -eq "FAST") { $color = "Yellow" }
            if ($status -eq "SLOW") { $color = "Red" }
            
            $line = "{0,5} | {1,6} | {2,10} | {3}" -f $currentBlock, $timeStr, $timeStr, $status
            Write-Host $line -ForegroundColor $color
            
            $previousBlock = $currentBlock
            $previousTime = $currentTime
        } else {
            # No new block yet
            $waitTime = ($currentTime - $previousTime).TotalSeconds
            $line = "{0,5} | {1,6} | {2,10} | {3}" -f "...", ($waitTime.ToString("F1") + "s"), "MINING", "WAIT"
            Write-Host $line -ForegroundColor Cyan
        }
    }
}

$endTime = Get-Date
$totalTime = ($endTime - $startTime).TotalSeconds
$endBlock = ./geth.exe --datadir ./qdata attach --exec "eth.blockNumber" 2>$null
$endBlock = [int]$endBlock.Trim()
$totalBlocks = $endBlock - $startBlock
$averageBlockTime = if ($totalBlocks -gt 0) { $totalTime / $totalBlocks } else { 0 }

Write-Host ""
Write-Host "📊 Summary:" -ForegroundColor Cyan
Write-Host "Total monitoring time: $($totalTime.ToString('F1'))s" -ForegroundColor White
Write-Host "Blocks mined: $totalBlocks" -ForegroundColor White

$avgColor = "Yellow"
if ($averageBlockTime -ge 10 -and $averageBlockTime -le 14) { $avgColor = "Green" }
Write-Host "Average block time: $($averageBlockTime.ToString('F1'))s" -ForegroundColor $avgColor

Write-Host "Target block time: 12.0s" -ForegroundColor Gray

$deviation = $averageBlockTime - 12
$devColor = "Red"
if ([Math]::Abs($deviation) -le 2) { $devColor = "Green" }
Write-Host "Deviation: $($deviation.ToString('F1'))s" -ForegroundColor $devColor

Write-Host ""
Write-Host "Monitoring complete!" -ForegroundColor Green