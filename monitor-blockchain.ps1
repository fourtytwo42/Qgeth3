#!/usr/bin/env pwsh
# Quantum-Geth Blockchain Monitor
# Shows current blockchain status, difficulty retargeting info, and block timing analysis

param(
    [string]$nodeUrl = "http://localhost:8545",
    [int]$refreshSeconds = 5,
    [switch]$once = $false
)

function Get-BlockInfo {
    param([string]$blockNumber)
    
    try {
        $body = @{
            jsonrpc = "2.0"
            method = "eth_getBlockByNumber"
            params = @($blockNumber, $false)
            id = 1
        } | ConvertTo-Json
        
        $response = Invoke-RestMethod -Uri $nodeUrl -Method Post -ContentType "application/json" -Body $body
        return $response.result
    } catch {
        return $null
    }
}

function Convert-HexToDecimal {
    param([string]$hexValue)
    if ([string]::IsNullOrEmpty($hexValue)) { return 0 }
    return [Convert]::ToInt64($hexValue, 16)
}

function Format-Duration {
    param([double]$seconds)
    
    if ($seconds -lt 60) {
        return "{0:F1}s" -f $seconds
    } elseif ($seconds -lt 3600) {
        return "{0:F1}m" -f ($seconds / 60)
    } else {
        return "{0:F1}h" -f ($seconds / 3600)
    }
}

function Show-BlockchainStatus {
    Write-Host ""
    Write-Host "🔗 QUANTUM-GETH BLOCKCHAIN MONITOR" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Gray
    
    # Get latest block
    $latestBlock = Get-BlockInfo "latest"
    if (-not $latestBlock) {
        Write-Host " Unable to connect to quantum-geth node at $nodeUrl" -ForegroundColor Red
        return
    }
    
    $currentBlockNumber = Convert-HexToDecimal $latestBlock.number
    $currentDifficulty = Convert-HexToDecimal $latestBlock.difficulty
    $currentTimeUtcstamp = Convert-HexToDecimal $latestBlock.timestamp
    
    # Calculate retarget info (every 100 blocks)
    $retargetInterval = 100
    $blocksToRetarget = $retargetInterval - ($currentBlockNumber % $retargetInterval)
    $nextRetargetBlock = $currentBlockNumber + $blocksToRetarget
    
    # Get last 10 blocks for timing analysis
    $blockTimes = @()
    $maxBlocks = [Math]::Min(10, $currentBlockNumber)
    
    if ($maxBlocks -gt 1) {
        for ($i = 0; $i -lt $maxBlocks; $i++) {
            $blockNum = $currentBlockNumber - $i
            $block = Get-BlockInfo ("0x" + $blockNum.ToString("X"))
            if ($block) {
                $timestamp = Convert-HexToDecimal $block.timestamp
                $blockTimes += $timestamp
            }
        }
    }
    
    # Calculate average block time
    $avgBlockTime = 0
    $minBlockTime = 0
    $maxBlockTime = 0
    if ($blockTimes.Count -gt 1) {
        $timeDiffs = @()
        for ($i = 0; $i -lt ($blockTimes.Count - 1); $i++) {
            $diff = $blockTimes[$i] - $blockTimes[$i + 1]
            if ($diff -gt 0) { $timeDiffs += $diff }
        }
        
        if ($timeDiffs.Count -gt 0) {
            $avgBlockTime = ($timeDiffs | Measure-Object -Average).Average
            $minBlockTime = ($timeDiffs | Measure-Object -Minimum).Minimum
            $maxBlockTime = ($timeDiffs | Measure-Object -Maximum).Maximum
        }
    }
    
    # Current time - fix timezone issue by using UTC for both
    $currentTimeUtcUtc = [DateTimeOffset]::UtcNow
    $lastBlockTimeUtcUtc = [DateTimeOffset]::FromUnixTimeSeconds($currentTimeUtcstamp).UtcDateTime
    $timeSinceLastBlock = ($currentTimeUtcUtc - $lastBlockTimeUtcUtc).TotalSeconds
    
    # Ensure positive time difference
    $timeSinceLastBlock = [Math]::Abs($timeSinceLastBlock)
    
    # Display information
    Write-Host " CURRENT STATUS" -ForegroundColor Yellow
    Write-Host "  Block Number: " -NoNewline -ForegroundColor Gray
    Write-Host $currentBlockNumber -ForegroundColor White
    
    Write-Host "  Difficulty: " -NoNewline -ForegroundColor Gray
    Write-Host ("{0:F6}" -f ($currentDifficulty / 1000000.0)) -ForegroundColor White -NoNewline
    Write-Host " (Raw: $currentDifficulty)" -ForegroundColor Gray
    
    Write-Host "  Last Block: " -NoNewline -ForegroundColor Gray
    Write-Host (Format-Duration $timeSinceLastBlock) -ForegroundColor $(if ($timeSinceLastBlock -gt 30) { "Red" } elseif ($timeSinceLastBlock -gt 15) { "Yellow" } else { "Green" }) -NoNewline
    Write-Host " ago" -ForegroundColor Gray
    
    Write-Host ""
    Write-Host " DIFFICULTY RETARGETING" -ForegroundColor Yellow
    Write-Host "  Retarget Interval: " -NoNewline -ForegroundColor Gray
    Write-Host "$retargetInterval blocks" -ForegroundColor White
    
    Write-Host "  Blocks to Retarget: " -NoNewline -ForegroundColor Gray
    Write-Host $blocksToRetarget -ForegroundColor $(if ($blocksToRetarget -le 5) { "Red" } elseif ($blocksToRetarget -le 20) { "Yellow" } else { "Green" })
    
    Write-Host "  Next Retarget Block: " -NoNewline -ForegroundColor Gray
    Write-Host $nextRetargetBlock -ForegroundColor White
    
    Write-Host ""
    Write-Host "  BLOCK TIMING ANALYSIS" -ForegroundColor Yellow
    Write-Host "  Target Block Time: " -NoNewline -ForegroundColor Gray
    Write-Host "12.0s" -ForegroundColor White -NoNewline
    Write-Host " (ASERT-Q algorithm)" -ForegroundColor Gray
    
    if ($avgBlockTime -gt 0) {
        $avgColor = if ($avgBlockTime -lt 8) { "Red" } elseif ($avgBlockTime -lt 10 -or $avgBlockTime -gt 15) { "Yellow" } else { "Green" }
        Write-Host "  Average Block Time: " -NoNewline -ForegroundColor Gray
        Write-Host (Format-Duration $avgBlockTime) -ForegroundColor $avgColor -NoNewline
        Write-Host " (last $($timeDiffs.Count) blocks)" -ForegroundColor Gray
        
        Write-Host "  Min/Max Block Time: " -NoNewline -ForegroundColor Gray
        Write-Host (Format-Duration $minBlockTime) -ForegroundColor Cyan -NoNewline
        Write-Host " / " -NoNewline -ForegroundColor Gray
        Write-Host (Format-Duration $maxBlockTime) -ForegroundColor Cyan
        
        # Calculate deviation from target
        $targetTime = 12.0
        $deviation = (($avgBlockTime - $targetTime) / $targetTime) * 100
        $deviationColor = if ([Math]::Abs($deviation) -lt 10) { "Green" } elseif ([Math]::Abs($deviation) -lt 25) { "Yellow" } else { "Red" }
        Write-Host "  Target Deviation: " -NoNewline -ForegroundColor Gray
        Write-Host ("{0:+0.1;-0.1}%" -f $deviation) -ForegroundColor $deviationColor
    } else {
        Write-Host "  Average Block Time: " -NoNewline -ForegroundColor Gray
        Write-Host "Calculating..." -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "  QUANTUM MINING INFO" -ForegroundColor Yellow
    Write-Host "  Quantum Puzzles: " -NoNewline -ForegroundColor Gray
    Write-Host "48 per block" -ForegroundColor White
    
    Write-Host "  Circuit Complexity: " -NoNewline -ForegroundColor Gray
    Write-Host "16 qubits  8192 T-gates" -ForegroundColor White
    
    Write-Host "  Security Level: " -NoNewline -ForegroundColor Gray
    Write-Host "1,152-bit aggregate" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "Updated: " -NoNewline -ForegroundColor Gray
    Write-Host (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -ForegroundColor White
    
    if (-not $once) {
        Write-Host ""
        Write-Host "Press Ctrl+C to stop monitoring..." -ForegroundColor Gray
        Write-Host "=" * 60 -ForegroundColor Gray
    }
}

# Main execution
try {
    if ($once) {
        Show-BlockchainStatus
    } else {
        Write-Host " Starting Quantum-Geth Blockchain Monitor..." -ForegroundColor Green
        Write-Host " Node URL: $nodeUrl" -ForegroundColor Cyan
        Write-Host " Refresh Rate: ${refreshSeconds}s" -ForegroundColor Cyan
        Write-Host ""
        
        while ($true) {
            Clear-Host
            Show-BlockchainStatus
            Start-Sleep $refreshSeconds
        }
    }
} catch [System.Management.Automation.PipelineStoppedException] {
    Write-Host ""
    Write-Host " Monitoring stopped by user." -ForegroundColor Yellow
} catch {
    Write-Host ""
    Write-Host " Error: $_" -ForegroundColor Red
} finally {
    Write-Host ""
}

