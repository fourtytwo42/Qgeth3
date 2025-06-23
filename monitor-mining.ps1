# Mining Monitor - Quantum-Geth v0.9 Simplified
# Monitors mining progress with streamlined difficulty system

param(
    [string]$nodeUrl = "http://localhost:8545",
    [int]$refreshInterval = 10,    # seconds between updates
    [string]$logFile = "qdata_quantum\geth.log"  # Path to geth log file
)

Write-Host "*** QUANTUM-GETH v0.9 MINING MONITOR (Simplified) ***" -ForegroundColor Cyan
Write-Host "Streamlined Bitcoin-style difficulty retargeting every 100 blocks" -ForegroundColor Green
Write-Host ""

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

function Get-DifficultyFromLogs {
    if (-not (Test-Path $logFile)) {
        return 0.5  # Default
    }
    
    try {
        # Get recent log entries
        $logLines = Get-Content $logFile -Tail 100 -ErrorAction SilentlyContinue
        
        # Look for the new simplified difficulty logs
        $difficultyLines = $logLines | Where-Object { $_ -match "Difficulty maintained|Difficulty retargeted" }
        
        if ($difficultyLines -and $difficultyLines.Count -gt 0) {
            $latestLine = $difficultyLines[-1]
            
            # Extract difficulty value (new format: difficulty=0.5)
            if ($latestLine -match "difficulty=([0-9.]+)") {
                return [decimal]$matches[1]
            }
        }
        
        # Fallback: look for Quantum difficulty calculation logs
        $calcLines = $logLines | Where-Object { $_ -match "Quantum difficulty calculation" }
        if ($calcLines -and $calcLines.Count -gt 0) {
            $latestCalcLine = $calcLines[-1]
            if ($latestCalcLine -match "parentDifficulty=([0-9.]+)") {
                return [decimal]$matches[1]
            }
        }
        
        return 0.5  # Default fallback
    } catch {
        return 0.5  # Default fallback
    }
}

function Get-NextRetargetFromLogs {
    if (-not (Test-Path $logFile)) {
        return $null
    }
    
    try {
        # Get recent log entries
        $logLines = Get-Content $logFile -Tail 50 -ErrorAction SilentlyContinue
        
        # Look for nextRetarget info in the logs
        $retargetLines = $logLines | Where-Object { $_ -match "nextRetarget=(\d+)" }
        
        if ($retargetLines -and $retargetLines.Count -gt 0) {
            $latestLine = $retargetLines[-1]
            if ($latestLine -match "nextRetarget=(\d+)") {
                return [int]$matches[1]
            }
        }
        
        return $null
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

function Show-MiningStatus {
    Write-Host "=" * 80 -ForegroundColor Yellow
    Write-Host "QUANTUM-GETH v0.9 MINING STATUS (Simplified)" -ForegroundColor Cyan
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "=" * 80 -ForegroundColor Yellow
    
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
    
    # Mining performance and difficulty
    if ($mining -eq $true) {
        # Get difficulty from logs (most reliable)
        $difficulty = Get-DifficultyFromLogs
        
        # Get next retarget from logs or calculate it
        $nextRetarget = Get-NextRetargetFromLogs
        if (-not $nextRetarget) {
            # Calculate next retarget: ((floor(blockNumber / 100) + 1) * 100)
            $retargetPeriod = [math]::Floor($currentBlock / 100)
            $nextRetarget = ($retargetPeriod + 1) * 100
        }
        $blocksToRetarget = $nextRetarget - $currentBlock
        
        # Get average block time
        $avgBlockTime = Get-AverageBlockTime 10
        $avgBlockTimeStr = if ($avgBlockTime) { "$avgBlockTime seconds" } else { "N/A" }
        
        Write-Host "DIFFICULTY SYSTEM:" -ForegroundColor Green
        Write-Host "  Current Difficulty: $difficulty" -ForegroundColor White
        Write-Host "  Retarget Interval: 100 blocks" -ForegroundColor Gray
        Write-Host "  Next Retarget Block: $nextRetarget" -ForegroundColor Cyan
        Write-Host "  Blocks to Next Retarget: $blocksToRetarget" -ForegroundColor Yellow
        Write-Host "  Target Block Time: 12 seconds" -ForegroundColor Gray
        Write-Host "  Average Block Time (last 10): $avgBlockTimeStr" -ForegroundColor White
        
        # Calculate mining efficiency based on block times
        if ($avgBlockTime -and $avgBlockTime -gt 0) {
            $efficiency = [math]::Round((12 / $avgBlockTime) * 100, 1)
            $efficiencyColor = if ($efficiency -gt 90) { 'Green' } elseif ($efficiency -gt 70) { 'Yellow' } else { 'Red' }
            Write-Host "  Mining Efficiency: $efficiency% (vs 12s target)" -ForegroundColor $efficiencyColor
            
            # Show if difficulty should increase/decrease
            if ($avgBlockTime -lt 10) {
                Write-Host "  Difficulty Trend: Should INCREASE (blocks too fast)" -ForegroundColor Red
            } elseif ($avgBlockTime -gt 14) {
                Write-Host "  Difficulty Trend: Should DECREASE (blocks too slow)" -ForegroundColor Green
            } else {
                Write-Host "  Difficulty Trend: Should MAINTAIN (good timing)" -ForegroundColor Yellow
            }
        }
        
        Write-Host ""
        Write-Host "QUANTUM MINING STATS:" -ForegroundColor Magenta
        
        # Calculate theoretical puzzles/sec based on block times
        $theoreticalPuzzlesPerSec = if ($avgBlockTime -and $avgBlockTime -gt 0) {
            [math]::Round(48 / $avgBlockTime, 2)
        } else {
            0
        }
        
        Write-Host "  Quantum Puzzles/sec: $theoreticalPuzzlesPerSec" -ForegroundColor White
        Write-Host "  Puzzles per Block: 48" -ForegroundColor White
        Write-Host "  Quantum Security: 16 qubits, 8192 T-gates" -ForegroundColor Cyan
        
        # Latest block info
        $latestBlock = Get-JsonRpcCall "eth_getBlockByNumber" @("latest", $false)
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
    Write-Host "Simplified Bitcoin-style difficulty retargeting every 100 blocks" -ForegroundColor Magenta
    Write-Host "Press Ctrl+C to stop monitoring..." -ForegroundColor Gray
}

# Main monitoring loop
Write-Host "Connecting to Quantum-Geth node at $nodeUrl..." -ForegroundColor Yellow
Write-Host "Refresh interval: $refreshInterval seconds" -ForegroundColor Gray
Write-Host "Log file: $logFile" -ForegroundColor Gray
Write-Host ""

while ($true) {
    try {
        Show-MiningStatus
        Start-Sleep $refreshInterval
        Clear-Host
    } catch {
        Write-Host "Error connecting to node: $_" -ForegroundColor Red
        Write-Host "Retrying in $refreshInterval seconds..." -ForegroundColor Yellow
        Start-Sleep $refreshInterval
    }
} 