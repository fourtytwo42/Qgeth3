# Mining Monitor - Quantum-Geth v0.9 BareBones+Halving with ASERT-Q Tracking
# Monitors mining progress, performance metrics, and ASERT-Q multiplier values

param(
    [string]$nodeUrl = "http://localhost:8545",
    [int]$refreshInterval = 10,    # seconds between updates
    [int]$retargetBlocks = 100,    # ASERT-Q retargets every 100 blocks
    [string]$logFile = "qdata_quantum\geth\geth.log"  # Path to geth log file
)

Write-Host "*** QUANTUM-GETH v0.9 BareBones+Halving MINING MONITOR ***" -ForegroundColor Cyan
Write-Host "Real-Time Mining Performance | ASERT-Q Difficulty | 48 Quantum Puzzles" -ForegroundColor Green
Write-Host "Enhanced with ASERT-Q Multiplier Tracking" -ForegroundColor Yellow
Write-Host ""

# Global variables to track ASERT-Q data
$global:LastASERTData = @{
    BlockNumber = 0
    Multiplier = 100
    ActualBlockTime = 12
    BaseDifficulty = 0.0005
    EffectiveDifficulty = 0.000005
    LastUpdate = (Get-Date)
}

function Get-HexToDecimal($hexValue) {
    if ($hexValue -match "^0x([0-9a-fA-F]+)$") {
        try {
            return [Convert]::ToInt64($matches[1], 16)
        } catch {
            # Handle very large numbers that exceed Int64
            $bigInt = [System.Numerics.BigInteger]::Parse($matches[1], [System.Globalization.NumberStyles]::HexNumber)
            return $bigInt.ToString()
        }
    }
    return 0
}

function Format-LargeNumber($number) {
    if ($number -is [string]) {
        # Already a string from BigInteger conversion
        return $number
    }
    
    if ($number -ge 1000000000) {
        $billions = [math]::Round($number / 1000000000, 2)
        return "$billions B"
    } elseif ($number -ge 1000000) {
        $millions = [math]::Round($number / 1000000, 2)
        return "$millions M"
    } elseif ($number -ge 1000) {
        $thousands = [math]::Round($number / 1000, 2)
        return "$thousands K"
    } else {
        return $number.ToString()
    }
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

function Parse-ASERTQData {
    if (-not (Test-Path $logFile)) {
        return $null
    }
    
    try {
        # Get recent log entries (last 100 lines to catch recent ASERT-Q activity)
        $logLines = Get-Content $logFile -Tail 100 -ErrorAction SilentlyContinue
        
        # Look for ASERT-Q multiplier calculations
        $asertLines = $logLines | Where-Object { $_ -match "🎯 ASERT-Q Multiplier calculated" }
        $targetLines = $logLines | Where-Object { $_ -match "🎯 ASERT-Q Target calculated" }
        
        if ($asertLines -and $asertLines.Count -gt 0) {
            # Parse the most recent ASERT-Q multiplier line
            $latestLine = $asertLines[-1]
            
            # Extract values using regex
            if ($latestLine -match "actualBlockTime=(\d+).*multiplier=(\d+).*blockNumber=(\d+)") {
                $global:LastASERTData.ActualBlockTime = [int]$matches[1]
                $global:LastASERTData.Multiplier = [int]$matches[2]
                $global:LastASERTData.BlockNumber = [int]$matches[3]
                $global:LastASERTData.LastUpdate = Get-Date
            }
        }
        
        if ($targetLines -and $targetLines.Count -gt 0) {
            # Parse the most recent target calculation
            $latestTargetLine = $targetLines[-1]
            
            # Extract base difficulty
            if ($latestTargetLine -match "difficulty=([0-9.]+)") {
                $global:LastASERTData.BaseDifficulty = [decimal]$matches[1]
            }
            
            # Calculate effective difficulty
            if ($global:LastASERTData.Multiplier -gt 0 -and $global:LastASERTData.BaseDifficulty -gt 0) {
                $global:LastASERTData.EffectiveDifficulty = $global:LastASERTData.BaseDifficulty / $global:LastASERTData.Multiplier
            }
        }
        
        return $global:LastASERTData
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
    
    # Parse ASERT-Q data from logs
    $asertData = Parse-ASERTQData
    
    # Basic node info
    $blockNumber = Get-JsonRpcCall "eth_blockNumber"
    $mining = Get-JsonRpcCall "eth_mining"
    
    if ($blockNumber) {
        $currentBlock = Get-HexToDecimal $blockNumber
        
        Write-Host "BLOCKCHAIN STATUS:" -ForegroundColor Green
        Write-Host "  Current Block: #$currentBlock" -ForegroundColor White
        Write-Host "  Mining Active: $(if ($mining -eq $true) { 'YES' } else { 'NO' })" -ForegroundColor $(if ($mining -eq $true) { 'Green' } else { 'Red' })
    }
    
    Write-Host ""
    
    # ASERT-Q Difficulty Status
    if ($asertData) {
        Write-Host "ASERT-Q DIFFICULTY ADJUSTMENT:" -ForegroundColor Cyan
        
        $minutesAgo = [math]::Round(((Get-Date) - $asertData.LastUpdate).TotalMinutes, 1)
        $dataAge = if ($minutesAgo -lt 1) { "Just now" } else { "$minutesAgo minutes ago" }
        
        Write-Host "  Base Difficulty: $($asertData.BaseDifficulty)" -ForegroundColor White
        Write-Host "  ASERT-Q Multiplier: $($asertData.Multiplier)x" -ForegroundColor Yellow
        Write-Host "  Effective Difficulty: $([math]::Round($asertData.EffectiveDifficulty, 9))" -ForegroundColor Green
        Write-Host "  Last Block Time: $($asertData.ActualBlockTime) seconds" -ForegroundColor White
        Write-Host "  Target Block Time: 12 seconds" -ForegroundColor Gray
        
        # Calculate adjustment direction
        $adjustment = if ($asertData.ActualBlockTime -gt 12) { "EASIER" } elseif ($asertData.ActualBlockTime -lt 12) { "HARDER" } else { "STABLE" }
        $adjustColor = if ($adjustment -eq "EASIER") { 'Green' } elseif ($adjustment -eq "HARDER") { 'Red' } else { 'Yellow' }
        Write-Host "  Adjustment Direction: $adjustment" -ForegroundColor $adjustColor
        Write-Host "  Data Age: $dataAge" -ForegroundColor Gray
        
        Write-Host ""
    }
    
    # Mining performance and difficulty
    if ($mining -eq $true) {
        # Get current difficulty
        $latestBlock = Get-JsonRpcCall "eth_getBlockByNumber" @("latest", $false)
        $difficulty = if ($latestBlock -and $latestBlock.difficulty) { 
            Get-HexToDecimal $latestBlock.difficulty 
        } else { 
            "N/A" 
        }
        
        # Calculate blocks to next retarget
        $blocksToRetarget = $retargetBlocks - ($currentBlock % $retargetBlocks)
        
        # Get average block time
        $avgBlockTime = Get-AverageBlockTime 10
        $avgBlockTimeStr = if ($avgBlockTime) { "$avgBlockTime seconds" } else { "N/A" }
        
        # Format difficulty for display
        $difficultyStr = if ($difficulty -ne "N/A") { Format-LargeNumber $difficulty } else { "N/A" }
        
        Write-Host "MINING PERFORMANCE:" -ForegroundColor Green
        Write-Host "  Current Difficulty: $difficultyStr" -ForegroundColor White
        Write-Host "  Blocks to Next Retarget: $blocksToRetarget" -ForegroundColor Yellow
        Write-Host "  Average Block Time (last 10): $avgBlockTimeStr" -ForegroundColor White
        
        # Calculate mining efficiency based on block times
        if ($avgBlockTime -and $avgBlockTime -gt 0) {
            $efficiency = [math]::Round((12 / $avgBlockTime) * 100, 1)
            $efficiencyColor = if ($efficiency -gt 90) { 'Green' } elseif ($efficiency -gt 70) { 'Yellow' } else { 'Red' }
            Write-Host "  Mining Efficiency: $efficiency% (vs 12s target)" -ForegroundColor $efficiencyColor
        }
        
        # Try to get quantum-specific stats
        $quantumStats = Get-JsonRpcCall "qmpow_getQuantumStats"
        if ($quantumStats) {
            Write-Host ""
            Write-Host "QUANTUM MINING STATS:" -ForegroundColor Magenta
            
            if ($quantumStats.mining) {
                $qHashrate = $quantumStats.mining.hashrate
                $qThreads = $quantumStats.mining.threads
                $qActive = $quantumStats.mining.isActive
                
                # Calculate mining efficiency based on puzzles/sec vs theoretical
                $theoreticalPuzzlesPerSec = 48 / 12  # 48 puzzles per 12 second target = 4 puzzles/sec
                $efficiency = if ($qHashrate -gt 0) { 
                    [math]::Round(($qHashrate / $theoreticalPuzzlesPerSec) * 100, 1) 
                } else { "N/A" }
                
                Write-Host "  Quantum Puzzles/sec: $qHashrate" -ForegroundColor White
                Write-Host "  Mining Threads: $qThreads" -ForegroundColor White
                Write-Host "  Mining Efficiency: $efficiency%" -ForegroundColor $(if ($efficiency -ne "N/A" -and [decimal]$efficiency -gt 90) { 'Green' } elseif ($efficiency -ne "N/A" -and [decimal]$efficiency -gt 70) { 'Yellow' } else { 'Red' })
            }
            
            if ($quantumStats.difficulty) {
                $puzzlesPerBlock = $quantumStats.difficulty.puzzlesPerBlock
                $qubitsPerPuzzle = $quantumStats.difficulty.qubitsPerPuzzle
                $tgatesPerPuzzle = $quantumStats.difficulty.tgatesPerPuzzle
                $securityBits = if ($quantumStats.quantum) { $quantumStats.quantum.effectiveSecurityBits } else { "N/A" }
                
                Write-Host "  Puzzles per Block: $puzzlesPerBlock" -ForegroundColor White
                Write-Host "  Quantum Security: $qubitsPerPuzzle qubits, $tgatesPerPuzzle T-gates ($securityBits bits)" -ForegroundColor Cyan
            }
        } else {
            Write-Host ""
            Write-Host "QUANTUM STATS: Not available (RPC method not responding)" -ForegroundColor Red
        }
        
        # Latest block info (condensed)
        if ($latestBlock) {
            $blockTimestamp = Get-HexToDecimal $latestBlock.timestamp
            $blockTime = (Get-Date "1970-01-01 00:00:00").AddSeconds($blockTimestamp).ToString("HH:mm:ss")
            $quantumBlobSize = if ($latestBlock.extraData -and $latestBlock.extraData.Length -gt 10) { 
                ($latestBlock.extraData.Length - 2) / 2 
            } else { 0 }
            
            Write-Host ""
            Write-Host "LATEST BLOCK:" -ForegroundColor Green
            Write-Host "  Block #$(Get-HexToDecimal $latestBlock.number) at $blockTime | Quantum Blob: $quantumBlobSize bytes" -ForegroundColor White
        }
    } else {
        Write-Host "MINING: Inactive" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "v0.9 BareBones+Halving: Real Quantum Mining | ASERT-Q Difficulty | 48 Sequential Puzzles" -ForegroundColor Magenta
    Write-Host "ASERT-Q: Bitcoin-style exponential adjustment with granular multipliers" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop monitoring..." -ForegroundColor Gray
}

# Main monitoring loop
Write-Host "Connecting to Quantum-Geth node at $nodeUrl..." -ForegroundColor Yellow
Write-Host "Refresh interval: $refreshInterval seconds" -ForegroundColor Gray
Write-Host "Log file: $logFile" -ForegroundColor Gray
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