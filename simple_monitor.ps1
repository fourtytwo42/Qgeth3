# Simple Quantum Mining Monitor

Write-Host "Monitoring quantum mining progress..."

$startTime = Get-Date
$startBlock = ./geth.exe --datadir ./qdata attach --exec "eth.blockNumber" 2>$null
if ($startBlock) {
    $startBlock = [int]$startBlock.Trim()
    Write-Host "Starting at block: $startBlock"
} else {
    Write-Host "Could not connect to blockchain"
    exit 1
}

$previousBlock = $startBlock
$previousTime = $startTime

for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 10
    
    $currentTime = Get-Date
    $currentBlock = ./geth.exe --datadir ./qdata attach --exec "eth.blockNumber" 2>$null
    
    if ($currentBlock) {
        $currentBlock = [int]$currentBlock.Trim()
        
        if ($currentBlock -gt $previousBlock) {
            $timeDiff = ($currentTime - $previousTime).TotalSeconds
            $blocksDiff = $currentBlock - $previousBlock
            $avgBlockTime = $timeDiff / $blocksDiff
            
            Write-Host "Block $currentBlock - Time: $($avgBlockTime.ToString('F1'))s"
            
            $previousBlock = $currentBlock
            $previousTime = $currentTime
        } else {
            $waitTime = ($currentTime - $previousTime).TotalSeconds
            Write-Host "Waiting... ($($waitTime.ToString('F1'))s since last block)"
        }
    }
}

$endTime = Get-Date
$endBlock = ./geth.exe --datadir ./qdata attach --exec "eth.blockNumber" 2>$null
if ($endBlock) {
    $endBlock = [int]$endBlock.Trim()
    $totalTime = ($endTime - $startTime).TotalSeconds
    $totalBlocks = $endBlock - $startBlock
    
    if ($totalBlocks -gt 0) {
        $avgBlockTime = $totalTime / $totalBlocks
        Write-Host "Summary: $totalBlocks blocks in $($totalTime.ToString('F1'))s"
        Write-Host "Average block time: $($avgBlockTime.ToString('F1'))s (target: 12s)"
    }
}

Write-Host "Monitoring complete" 