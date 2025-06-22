# Monitor Quantum Blockchain Progress

Write-Host "Monitoring Quantum Blockchain Progress..."

$startBlock = ./geth.exe --datadir ./qdata attach --exec "eth.blockNumber" 2>$null
$startBlock = $startBlock.Trim()
Write-Host "Starting block: $startBlock"

for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 5
    
    $currentBlock = ./geth.exe --datadir ./qdata attach --exec "eth.blockNumber" 2>$null
    $currentBlock = $currentBlock.Trim()
    
    $blockDiff = [int]$currentBlock - [int]$startBlock
    $blocksPerSecond = $blockDiff / (5 * $i)
    
    Write-Host "Check $i - Block: $currentBlock, Mined: +$blockDiff blocks, Speed: $blocksPerSecond blocks/sec"
}

Write-Host "Monitoring complete!" 