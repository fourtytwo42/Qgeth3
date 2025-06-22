# Monitor Quantum Mining
# Provides real-time monitoring of quantum mining performance
# Shows puzzles per second instead of hashes, current block info, and mining status

param(
    [int]$refreshInterval = 5,
    [string]$rpcUrl = "http://localhost:8545",
    [int]$displayLines = 20
)

Write-Host "*** QUANTUM MINING MONITOR ***" -ForegroundColor Cyan
Write-Host "Monitoring quantum mining performance..." -ForegroundColor Green
Write-Host "Refresh interval: $refreshInterval seconds" -ForegroundColor Yellow
Write-Host "RPC URL: $rpcUrl" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
Write-Host ""

# Initialize variables for tracking
$lastBlockNumber = -1  # Will be set on first run
$lastPuzzleCount = 0
$lastTimestamp = Get-Date
$blocksMined = 0
$totalPuzzlesSolved = 0
$startTime = Get-Date
$initialBlockNumber = -1  # Track starting block number

function Get-BlockchainInfo {
    try {
        # Get current block number
        $blockNumberHex = Invoke-RestMethod -Uri $rpcUrl -Method Post -ContentType "application/json" -Body (@{
            jsonrpc = "2.0"
            method = "eth_blockNumber"
            params = @()
            id = 1
        } | ConvertTo-Json) -ErrorAction Stop
        
        $currentBlock = [Convert]::ToInt64($blockNumberHex.result, 16)
        
        # Get latest block details
        $latestBlock = Invoke-RestMethod -Uri $rpcUrl -Method Post -ContentType "application/json" -Body (@{
            jsonrpc = "2.0"
            method = "eth_getBlockByNumber"
            params = @("latest", $false)
            id = 2
        } | ConvertTo-Json) -ErrorAction Stop
        
        # Get mining status
        $miningStatus = Invoke-RestMethod -Uri $rpcUrl -Method Post -ContentType "application/json" -Body (@{
            jsonrpc = "2.0"
            method = "eth_mining"
            params = @()
            id = 3
        } | ConvertTo-Json) -ErrorAction Stop
        
        return @{
            BlockNumber = $currentBlock
            BlockHash = $latestBlock.result.hash
            BlockTimestamp = [Convert]::ToInt64($latestBlock.result.timestamp, 16)
            Difficulty = [Convert]::ToInt64($latestBlock.result.difficulty, 16)
            IsMining = $miningStatus.result
            Success = $true
        }
    }
    catch {
        return @{
            Success = $false
            Error = $_.Exception.Message
        }
    }
}

function Format-Duration {
    param($seconds)
    $hours = [math]::Floor($seconds / 3600)
    $minutes = [math]::Floor(($seconds % 3600) / 60)
    $secs = [math]::Floor($seconds % 60)
    return "{0:00}h {1:00}m {2:00}s" -f $hours, $minutes, $secs
}

function Display-Stats {
    param($info, $puzzlesPerSecond, $blocksMinedThisSession, $totalRuntime)
    
    Clear-Host
    Write-Host "*** QUANTUM MINING MONITOR ***" -ForegroundColor Cyan
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
    
    if ($info.Success) {
        # Current blockchain status
        Write-Host "=== BLOCKCHAIN STATUS ===" -ForegroundColor Green
        Write-Host "Current Block:     #$($info.BlockNumber)" -ForegroundColor White
        Write-Host "Block Hash:        $($info.BlockHash)" -ForegroundColor Gray
        Write-Host "Difficulty:        $($info.Difficulty)" -ForegroundColor Yellow
        Write-Host "Mining Status:     $(if($info.IsMining) {'ACTIVE'} else {'INACTIVE'})" -ForegroundColor $(if($info.IsMining) {'Green'} else {'Red'})
        Write-Host ""
        
        # Mining performance
        Write-Host "=== QUANTUM MINING PERFORMANCE ===" -ForegroundColor Magenta
        Write-Host "Puzzles/Second:    $([math]::Round($puzzlesPerSecond, 2)) puzzles/sec" -ForegroundColor Cyan
        Write-Host "Security Level:    1,152-bit (48 quantum puzzles)" -ForegroundColor Yellow
        Write-Host "Blocks Mined:      $blocksMinedThisSession this session" -ForegroundColor Green
        Write-Host "Runtime:           $(Format-Duration $totalRuntime)" -ForegroundColor White
        Write-Host ""
        
        # Bitcoin-style features
        Write-Host "=== BITCOIN-STYLE FEATURES ===" -ForegroundColor Blue
        Write-Host "Consensus:         QMPoW (Quantum Proof-of-Work)" -ForegroundColor Gray
        Write-Host "Nonce Style:       Bitcoin-style iteration (0-4B)" -ForegroundColor Gray
        Write-Host "Target Validation: Bitcoin-style difficulty" -ForegroundColor Gray
        Write-Host "Retargeting:       Every 100 blocks" -ForegroundColor Gray
        Write-Host ""
        
        if ($blocksMinedThisSession -gt 0 -and $totalRuntime -gt 0) {
            $avgBlockTime = $totalRuntime / $blocksMinedThisSession
            $puzzlesPerMinute = ($blocksMinedThisSession * 48) / ($totalRuntime / 60)
            $blocksPerMinute = $blocksMinedThisSession / ($totalRuntime / 60)
            Write-Host "Average Block Time: $([math]::Round($avgBlockTime, 1)) seconds" -ForegroundColor Cyan
            Write-Host "Blocks per Minute:  $([math]::Round($blocksPerMinute, 1)) blocks/min" -ForegroundColor Cyan
            Write-Host "Puzzles per Minute: $([math]::Round($puzzlesPerMinute, 1)) puzzles/min" -ForegroundColor Cyan
        }
        
        if ($info.IsMining) {
            Write-Host ""
            Write-Host "QUANTUM MINING ACTIVE - Bitcoin-style quantum blocks being mined!" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "Mining paused or not connected" -ForegroundColor Yellow
        }
    } else {
        Write-Host "=== CONNECTION ERROR ===" -ForegroundColor Red
        Write-Host "Cannot connect to geth node" -ForegroundColor Red
        Write-Host "Error: $($info.Error)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Make sure geth is running with RPC enabled:" -ForegroundColor Yellow
        Write-Host "  .\start-geth-mining.ps1 -threads 1" -ForegroundColor Gray
    }
}

# Main monitoring loop
while ($true) {
    $info = Get-BlockchainInfo
    $currentTime = Get-Date
    $totalRuntime = ($currentTime - $startTime).TotalSeconds
    
    if ($info.Success) {
        # Initialize starting block number on first run
        if ($initialBlockNumber -eq -1) {
            $initialBlockNumber = $info.BlockNumber
            $lastBlockNumber = $info.BlockNumber
        }
        
        # Check if new block was mined since we started monitoring
        if ($info.BlockNumber -gt $lastBlockNumber) {
            $newBlocks = $info.BlockNumber - $lastBlockNumber
            $blocksMined += $newBlocks
            $lastBlockNumber = $info.BlockNumber
            
            # Estimate puzzles solved (48 puzzles per block)
            $totalPuzzlesSolved += ($newBlocks * 48)
        }
        
        # Calculate total blocks mined this session
        $totalBlocksThisSession = $info.BlockNumber - $initialBlockNumber
        
        # Calculate puzzles per second
        $puzzlesPerSecond = if ($totalRuntime -gt 0) { $totalPuzzlesSolved / $totalRuntime } else { 0 }
        
        Display-Stats $info $puzzlesPerSecond $totalBlocksThisSession $totalRuntime
    } else {
        Display-Stats $info 0 $blocksMined $totalRuntime
    }
    
    Start-Sleep -Seconds $refreshInterval
} 