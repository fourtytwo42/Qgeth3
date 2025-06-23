# Mining Monitor - Quantum-Geth v0.9 BareBones+Halving
# Monitors mining progress and performance metrics

param(
    [string]$nodeUrl = "http://localhost:8545",
    [int]$refreshInterval = 10,    # seconds between updates
    [int]$retargetBlocks = 100     # ASERT-Q retargets every 100 blocks
)

Write-Host "*** QUANTUM-GETH v0.9 BareBones+Halving MINING MONITOR ***" -ForegroundColor Cyan
Write-Host "Real-Time Mining Performance | ASERT-Q Difficulty | 48 Quantum Puzzles" -ForegroundColor Green
Write-Host ""

function Get-HexToDecimal($hexValue) {
    if ($hexValue -match "^0x([0-9a-fA-F]+)$") {
        return [Convert]::ToInt64($matches[1], 16)
    }
    return 0
}

function Get-WeiToQGC($weiValue) {
    $wei = [decimal]$weiValue
    return [math]::Round($wei / 1000000000000000000, 6)
}

function Get-JsonRpcCall($method, $params = @()) {
    $body = @{
        jsonrpc = "2.0"
        method = $method
        params = $params
        id = 1
    } | ConvertTo-Json -Depth 10

    try {
        $response = Invoke-RestMethod -Uri $nodeUrl -Method Post -Body $body -ContentType "application/json"
        return $response.result
    } catch {
        return $null
    }
}

function Get-AverageBlockTime {
    param([int]$blockCount = 10)
    
    $latestBlockHex = Get-JsonRpcCall "eth_blockNumber"
    if (-not $latestBlockHex) { return $null }
    
    $latestBlock = Get-HexToDecimal $latestBlockHex
    if ($latestBlock -lt $blockCount) { return $null }
    
    $startBlockHex = "0x" + [Convert]::ToString($latestBlock - $blockCount + 1, 16)
    $endBlockHex = $latestBlockHex
    
    $startBlock = Get-JsonRpcCall "eth_getBlockByNumber" @($startBlockHex, $false)
    $endBlock = Get-JsonRpcCall "eth_getBlockByNumber" @($endBlockHex, $false)
    
    if ($startBlock -and $endBlock) {
        $startTime = Get-HexToDecimal $startBlock.timestamp
        $endTime = Get-HexToDecimal $endBlock.timestamp
        $timeDiff = $endTime - $startTime
        return [math]::Round($timeDiff / ($blockCount - 1), 2)
    }
    
    return $null
}

function Get-MiningEfficiency {
    param([decimal]$currentHashrate, [decimal]$avgBlockTime, [int]$targetBlockTime = 12)
    
    if ($avgBlockTime -le 0 -or $currentHashrate -le 0) { return "N/A" }
    
    # Calculate theoretical vs actual performance
    $theoreticalBlocksPerHour = 3600 / $targetBlockTime
    $actualBlocksPerHour = 3600 / $avgBlockTime
    $efficiency = ($actualBlocksPerHour / $theoreticalBlocksPerHour) * 100
    
    return [math]::Round($efficiency, 1)
}

function Show-V09Status {
    Write-Host "=" * 80 -ForegroundColor Yellow
    Write-Host "QUANTUM-GETH v0.9 BareBones+Halving MINING STATUS" -ForegroundColor Cyan
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "=" * 80 -ForegroundColor Yellow
    
    # Basic node info
    $blockNumber = Get-JsonRpcCall "eth_blockNumber"
    $hashRate = Get-JsonRpcCall "eth_hashrate"
    $mining = Get-JsonRpcCall "eth_mining"
    $syncing = Get-JsonRpcCall "eth_syncing"
    $peerCount = Get-JsonRpcCall "net_peerCount"
    
    if ($blockNumber) {
        $currentBlock = Get-HexToDecimal $blockNumber
        $currentEpoch = [math]::Floor($currentBlock / 600000)
        $currentSubsidy = 50 / [math]::Pow(2, $currentEpoch)
        
        Write-Host "BLOCKCHAIN STATUS:" -ForegroundColor Green
        Write-Host "  Current Block: #$currentBlock" -ForegroundColor White
        Write-Host "  Current Epoch: $currentEpoch" -ForegroundColor White
        Write-Host "  Current Block Subsidy: $currentSubsidy QGC" -ForegroundColor Cyan
        Write-Host "  Mining Active: $(if ($mining -eq $true) { 'YES' } else { 'NO' })" -ForegroundColor $(if ($mining -eq $true) { 'Green' } else { 'Red' })
        Write-Host "  Peer Count: $(Get-HexToDecimal $peerCount)" -ForegroundColor White
        
        if ($syncing -ne $false) {
            Write-Host "  Syncing: YES (startingBlock: $(Get-HexToDecimal $syncing.startingBlock), currentBlock: $(Get-HexToDecimal $syncing.currentBlock), highestBlock: $(Get-HexToDecimal $syncing.highestBlock))" -ForegroundColor Yellow
        } else {
            Write-Host "  Syncing: NO (fully synchronized)" -ForegroundColor Green
        }
    }
    
    Write-Host ""
    
    # Mining performance and difficulty
    if ($mining -eq $true) {
        Write-Host "MINING PERFORMANCE:" -ForegroundColor Green
        
        # Get current difficulty
        $latestBlock = Get-JsonRpcCall "eth_getBlockByNumber" @("latest", $false)
        $difficulty = if ($latestBlock) { Get-HexToDecimal $latestBlock.difficulty } else { 0 }
        
        # Calculate blocks to next retarget
        $blocksToRetarget = $retargetBlocks - ($currentBlock % $retargetBlocks)
        
        # Get average block time
        $avgBlockTime = Get-AverageBlockTime 10
        $avgBlockTimeStr = if ($avgBlockTime) { "$avgBlockTime seconds" } else { "N/A" }
        
        # Get mining efficiency
        $currentHashrate = Get-HexToDecimal $hashRate
        $efficiency = Get-MiningEfficiency $currentHashrate $avgBlockTime
        
        Write-Host "  Hash Rate: $currentHashrate H/s" -ForegroundColor White
        Write-Host "  Current Difficulty: $difficulty" -ForegroundColor White
        Write-Host "  Blocks to Next Retarget: $blocksToRetarget" -ForegroundColor Yellow
        Write-Host "  Average Block Time (last 10): $avgBlockTimeStr" -ForegroundColor White
        Write-Host "  Target Block Time: 12 seconds" -ForegroundColor Gray
        Write-Host "  Mining Efficiency: $efficiency%" -ForegroundColor $(if ($efficiency -ne "N/A" -and [decimal]$efficiency -gt 90) { 'Green' } elseif ($efficiency -ne "N/A" -and [decimal]$efficiency -gt 70) { 'Yellow' } else { 'Red' })
        
        # Try to get quantum-specific stats
        $quantumStats = Get-JsonRpcCall "qmpow_getQuantumStats"
        if ($quantumStats) {
            Write-Host ""
            Write-Host "QUANTUM MINING STATS:" -ForegroundColor Magenta
            
            if ($quantumStats.mining) {
                $qHashrate = $quantumStats.mining.hashrate
                $qThreads = $quantumStats.mining.threads
                $qActive = $quantumStats.mining.isActive
                
                Write-Host "  Quantum Puzzles/sec: $qHashrate" -ForegroundColor White
                Write-Host "  Mining Threads: $qThreads" -ForegroundColor White
                Write-Host "  Quantum Mining Active: $qActive" -ForegroundColor $(if ($qActive -eq $true) { 'Green' } else { 'Red' })
            }
            
            if ($quantumStats.difficulty) {
                $puzzlesPerBlock = $quantumStats.difficulty.puzzlesPerBlock
                $qubitsPerPuzzle = $quantumStats.difficulty.qubitsPerPuzzle
                $tgatesPerPuzzle = $quantumStats.difficulty.tgatesPerPuzzle
                $totalComplexity = $quantumStats.difficulty.totalComplexity
                
                Write-Host "  Puzzles per Block: $puzzlesPerBlock" -ForegroundColor White
                Write-Host "  Qubits per Puzzle: $qubitsPerPuzzle" -ForegroundColor White
                Write-Host "  T-gates per Puzzle: $tgatesPerPuzzle" -ForegroundColor White
                Write-Host "  Total Quantum Complexity: $totalComplexity" -ForegroundColor Cyan
            }
            
            if ($quantumStats.quantum) {
                $securityBits = $quantumStats.quantum.effectiveSecurityBits
                $stateSpace = $quantumStats.quantum.stateSpaceSize
                
                Write-Host "  Effective Security: $securityBits bits" -ForegroundColor Green
                Write-Host "  Quantum State Space: 2^$qubitsPerPuzzle = $stateSpace states" -ForegroundColor Gray
            }
        } else {
            Write-Host ""
            Write-Host "QUANTUM STATS: Not available (RPC method not responding)" -ForegroundColor Red
        }
        
        # Calculate estimated rewards
        if ($currentSubsidy -gt 0 -and $avgBlockTime) {
            $blocksPerHour = 3600 / $avgBlockTime
            $qgcPerHour = $currentSubsidy * $blocksPerHour
            Write-Host ""
            Write-Host "ESTIMATED REWARDS:" -ForegroundColor Green
            Write-Host "  QGC/hour (solo mining): $([math]::Round($qgcPerHour, 4)) QGC" -ForegroundColor Cyan
            Write-Host "  QGC/day (solo mining): $([math]::Round($qgcPerHour * 24, 2)) QGC" -ForegroundColor Cyan
        }
    }
    
    Write-Host ""
    
    # Latest block details
    Write-Host "LATEST BLOCK DETAILS:" -ForegroundColor Green
    if ($latestBlock) {
        $txCount = if ($latestBlock.transactions) { $latestBlock.transactions.Count } else { 0 }
        $blockTimestamp = Get-HexToDecimal $latestBlock.timestamp
        $blockTime = (Get-Date "1970-01-01 00:00:00").AddSeconds($blockTimestamp).ToString("yyyy-MM-dd HH:mm:ss")
        
        Write-Host "  Block #$(Get-HexToDecimal $latestBlock.number)" -ForegroundColor White
        Write-Host "  Timestamp: $blockTime" -ForegroundColor White
        Write-Host "  Transaction Count: $txCount" -ForegroundColor White
        Write-Host "  Gas Used: $(Get-HexToDecimal $latestBlock.gasUsed)" -ForegroundColor White
        Write-Host "  Gas Limit: $(Get-HexToDecimal $latestBlock.gasLimit)" -ForegroundColor White
        Write-Host "  Miner: $($latestBlock.miner)" -ForegroundColor White
        
        # Check for quantum blob in extraData
        if ($latestBlock.extraData -and $latestBlock.extraData.Length -gt 10) {
            $quantumBlobSize = ($latestBlock.extraData.Length - 2) / 2
            Write-Host "  Quantum Blob: Present ($quantumBlobSize bytes)" -ForegroundColor Cyan
        } else {
            Write-Host "  Quantum Blob: Not detected" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "v0.9 BareBones+Halving: Real Quantum Mining | ASERT-Q Difficulty | 48 Sequential Puzzles" -ForegroundColor Magenta
    Write-Host "Press Ctrl+C to stop monitoring..." -ForegroundColor Gray
}

# Main monitoring loop
Write-Host "Connecting to Quantum-Geth node at $nodeUrl..." -ForegroundColor Yellow
Write-Host "Refresh interval: $refreshInterval seconds" -ForegroundColor Gray
Write-Host ""

while ($true) {
    try {
        Show-V09Status
        Start-Sleep $refreshInterval
        Clear-Host
    } catch {
        Write-Host "Error connecting to node: $_" -ForegroundColor Red
        Write-Host "Retrying in $refreshInterval seconds..." -ForegroundColor Yellow
        Start-Sleep $refreshInterval
    }
} 