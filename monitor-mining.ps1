# Monitor Quantum-Geth Mining Performance
# Real-time quantum mining metrics

param(
    [switch]$Once,  # Run once and exit instead of continuous monitoring
    [int]$Interval = 5  # Update interval in seconds
)

# Global variables to track performance metrics
$global:LastBlockNumber = 0
$global:LastBlockTime = Get-Date
$global:LastQNonce = 0
$global:BlockTimes = @()
$global:PuzzleTimes = @()

function Show-QuantumStats {
    Write-Host ""
    Write-Host "=== Quantum Mining Status ===" -ForegroundColor Cyan
    Write-Host "Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    
    # Check if geth is running
    $gethProcess = Get-Process geth -ErrorAction SilentlyContinue
    if ($gethProcess) {
        Write-Host "Status: MINING" -ForegroundColor Green
        
        try {
            # Get current block number
            $blockNumberResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8545" -Method POST -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' -TimeoutSec 3
            $blockNumber = [Convert]::ToInt32($blockNumberResponse.result, 16)
            
            # Get current block info
            $currentBlockResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8545" -Method POST -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"eth_getBlockByNumber","params":["latest", false],"id":1}' -TimeoutSec 3
            
            # Display block information
            Write-Host "Block Number: $blockNumber" -ForegroundColor White
            
            # Handle difficulty display properly
            if ($currentBlockResponse.result -and $currentBlockResponse.result.difficulty) {
                $difficultyHex = $currentBlockResponse.result.difficulty
                $difficulty = [Convert]::ToInt64($difficultyHex, 16)
                
                if ($difficulty -ge 1000000) {
                    $difficultyDisplay = "$([math]::Round($difficulty / 1000000, 3))M"
                } elseif ($difficulty -ge 1000) {
                    $difficultyDisplay = "$([math]::Round($difficulty / 1000, 3))K"
                } else {
                    $difficultyDisplay = "$difficulty"
                }
                Write-Host "Difficulty: $difficultyDisplay" -ForegroundColor White
            } else {
                Write-Host "Difficulty: Unknown (RLP Error)" -ForegroundColor Yellow
            }
            
            # Show timestamp and calculate block time
            if ($currentBlockResponse.result -and $currentBlockResponse.result.timestamp) {
                $timestampHex = $currentBlockResponse.result.timestamp
                $timestamp = [Convert]::ToInt64($timestampHex, 16)
                $blockDateTime = [DateTimeOffset]::FromUnixTimeSeconds($timestamp)
                $blockTime = $blockDateTime.ToString("HH:mm:ss")
                Write-Host "Last Block: $blockTime" -ForegroundColor White
                
                # Calculate current block time (time since last block)
                $currentTime = Get-Date
                $timeSinceBlock = ($currentTime - $blockDateTime.DateTime).TotalSeconds
                if ($timeSinceBlock -ge 0 -and $timeSinceBlock -lt 3600) { # Only show if reasonable (< 1 hour)
                    Write-Host "Block Age: $([math]::Round($timeSinceBlock, 1))s" -ForegroundColor Cyan
                }
                
                # Track block timing for rate calculations
                if ($blockNumber -ne $global:LastBlockNumber) {
                    $blockInterval = ($currentTime - $global:LastBlockTime).TotalSeconds
                    if ($global:LastBlockNumber -gt 0 -and $blockInterval -gt 0 -and $blockInterval -lt 300) {
                        $global:BlockTimes += $blockInterval
                        # Keep only last 10 block times for moving average
                        if ($global:BlockTimes.Count -gt 10) {
                            $global:BlockTimes = $global:BlockTimes[-10..-1]
                        }
                        Write-Host "New Block! Interval: $([math]::Round($blockInterval, 1))s" -ForegroundColor Green
                    }
                    $global:LastBlockNumber = $blockNumber
                    $global:LastBlockTime = $currentTime
                }
            }
            
            # Get quantum-specific metrics from latest block
            if ($currentBlockResponse.result) {
                $block = $currentBlockResponse.result
                
                # Try to get quantum fields (these might be in extraData or custom fields)
                $qbits = 12  # Default from v0.9-rc3-hw0 spec
                $puzzles = 48  # Fixed puzzle count
                
                # Display quantum mining configuration
                Write-Host "Quantum Config: ${qbits} qubits, ${puzzles} puzzles" -ForegroundColor Magenta
                
                # Calculate average block time and rates
                if ($global:BlockTimes.Count -gt 0) {
                    $avgBlockTime = ($global:BlockTimes | Measure-Object -Average).Average
                    Write-Host "Avg Block Time: $([math]::Round($avgBlockTime, 1))s (from $($global:BlockTimes.Count) blocks)" -ForegroundColor Cyan
                    
                    # Calculate puzzles per second
                    $puzzlesPerSecond = $puzzles / $avgBlockTime
                    Write-Host "Puzzles/sec: $([math]::Round($puzzlesPerSecond, 2))" -ForegroundColor Green
                    
                    # Calculate theoretical vs actual performance
                    $theoreticalTime = 6.0  # Typical quantum computation time from logs
                    $efficiency = ($theoreticalTime / $avgBlockTime) * 100
                    Write-Host "Mining Efficiency: $([math]::Round($efficiency, 1))%" -ForegroundColor $(if($efficiency -gt 90) { "Green" } elseif($efficiency -gt 70) { "Yellow" } else { "Red" })
                } else {
                    Write-Host "Tracking: Waiting for next block to calculate rates..." -ForegroundColor Yellow
                }
                
                # Try to extract QNonce from block (this would be in quantum blob)
                if ($block.nonce) {
                    try {
                        $nonceHex = $block.nonce
                        $currentNonce = [Convert]::ToInt64($nonceHex, 16)
                        Write-Host "Current Nonce: $currentNonce" -ForegroundColor Gray
                        
                        # Calculate nonce rate (attempts per second)
                        if ($global:LastQNonce -gt 0 -and $global:BlockTimes.Count -gt 0) {
                            $nonceIncrease = $currentNonce - $global:LastQNonce
                            $timeInterval = ($global:BlockTimes | Measure-Object -Average).Average
                            if ($timeInterval -gt 0 -and $nonceIncrease -gt 0) {
                                $noncePerSecond = $nonceIncrease / $timeInterval
                                Write-Host "Nonce/sec: $([math]::Round($noncePerSecond, 2))" -ForegroundColor Green
                            } else {
                                # Quantum mining typically uses QNonce64, estimate based on attempts
                                $estimatedNonceRate = 1.0 / $timeInterval
                                Write-Host "Est. Nonce/sec: $([math]::Round($estimatedNonceRate, 3)) (quantum attempts)" -ForegroundColor Yellow
                            }
                        }
                        $global:LastQNonce = $currentNonce
                    } catch {
                        # If nonce parsing fails, show what we can
                        Write-Host "Nonce: Unable to parse from block" -ForegroundColor Yellow
                        if ($global:BlockTimes.Count -gt 0) {
                            $avgBlockTime = ($global:BlockTimes | Measure-Object -Average).Average
                            # Quantum mining typically tries 1 nonce per puzzle solution
                            $estimatedNonceRate = 1.0 / $avgBlockTime
                            Write-Host "Est. Nonce/sec: $([math]::Round($estimatedNonceRate, 3)) (quantum rate)" -ForegroundColor Yellow
                        }
                    }
                } else {
                    Write-Host "Nonce: Not available in block data" -ForegroundColor Yellow
                }
                
                # Show data collection status
                Write-Host "Data Points: $($global:BlockTimes.Count) block intervals collected" -ForegroundColor Gray
            }
            
        } catch {
            Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        }
        
    } else {
        Write-Host "Status: NOT RUNNING" -ForegroundColor Red
        # Reset tracking variables when not mining
        $global:LastBlockNumber = 0
        $global:BlockTimes = @()
        $global:PuzzleTimes = @()
    }
    
    Write-Host "=============================" -ForegroundColor Cyan
}

# Main execution
if ($Once) {
    # Run once and exit
    Show-QuantumStats
} else {
    # Continuous monitoring
    Write-Host "Quantum Mining Monitor Started (Ctrl+C to exit)" -ForegroundColor Green
    Write-Host "Tracking: Block Time | Puzzles/sec | Nonce/sec | Efficiency" -ForegroundColor Gray
    Write-Host "Note: Rate calculations appear after observing multiple blocks" -ForegroundColor Yellow
    
    try {
        while ($true) {
            Show-QuantumStats
            Start-Sleep $Interval
        }
    } catch [System.Management.Automation.PipelineStoppedException] {
        Write-Host ""
        Write-Host "Monitor stopped by user" -ForegroundColor Yellow
    }
} 