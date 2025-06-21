#!/usr/bin/env pwsh
# Simple Quantum Blockchain Monitor
# Works with standard APIs while quantum APIs are being fixed

param(
    [string]$RpcUrl = "http://localhost:8545",
    [int]$RefreshInterval = 5
)

function Invoke-RpcCall {
    param([string]$Method, [array]$Params = @())
    
    $body = @{
        jsonrpc = "2.0"
        method = $Method
        params = $Params
        id = 1
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-RestMethod -Uri $RpcUrl -Method Post -Body $body -ContentType "application/json"
        return $response.result
    } catch {
        return $null
    }
}

function Get-BlockTime {
    param([int]$BlockNumber)
    
    if ($BlockNumber -le 1) { return 0 }
    
    $currentBlock = Invoke-RpcCall "eth_getBlockByNumber" @("0x$($BlockNumber.ToString('X'))", $false)
    $prevBlock = Invoke-RpcCall "eth_getBlockByNumber" @("0x$(($BlockNumber-1).ToString('X'))", $false)
    
    if ($currentBlock -and $prevBlock) {
        $currentTime = [Convert]::ToInt64($currentBlock.timestamp, 16)
        $prevTime = [Convert]::ToInt64($prevBlock.timestamp, 16)
        return $currentTime - $prevTime
    }
    return 0
}

$lastBlockNumber = 0
$blockTimes = @()

while ($true) {
    Clear-Host
    
    Write-Host "🚀 Quantum Blockchain Monitor (Simplified)" -ForegroundColor Cyan
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
    
    # Get current block
    $blockHex = Invoke-RpcCall "eth_blockNumber"
    if ($blockHex) {
        $blockNumber = [Convert]::ToInt64($blockHex, 16)
        Write-Host "📦 Current Block: $blockNumber" -ForegroundColor Green
        
        # Calculate hashrate if we have a new block
        if ($blockNumber -gt $lastBlockNumber -and $lastBlockNumber -gt 0) {
            $blockTime = Get-BlockTime $blockNumber
            if ($blockTime -gt 0) {
                $blockTimes += $blockTime
                # Keep only last 10 block times
                if ($blockTimes.Count -gt 10) {
                    $blockTimes = $blockTimes[-10..-1]
                }
            }
        }
        $lastBlockNumber = $blockNumber
        
        # Get block details
        $block = Invoke-RpcCall "eth_getBlockByNumber" @("latest", $false)
        if ($block) {
            $difficulty = [Convert]::ToInt64($block.difficulty, 16)
            $gasUsed = [Convert]::ToInt64($block.gasUsed, 16)
            $gasLimit = [Convert]::ToInt64($block.gasLimit, 16)
            
            Write-Host ""
            Write-Host "📊 Block Information" -ForegroundColor Yellow
            Write-Host "====================" -ForegroundColor Yellow
            Write-Host "Difficulty: $difficulty" -ForegroundColor White
            Write-Host "Gas Used: $gasUsed / $gasLimit" -ForegroundColor White
            Write-Host "Transactions: $($block.transactions.Count)" -ForegroundColor White
        }
        
        # Show block timing statistics
        if ($blockTimes.Count -gt 0) {
            $avgBlockTime = ($blockTimes | Measure-Object -Average).Average
            $minBlockTime = ($blockTimes | Measure-Object -Minimum).Minimum
            $maxBlockTime = ($blockTimes | Measure-Object -Maximum).Maximum
            
            # Estimate quantum hashrate (assuming 300 puzzles per block)
            $puzzlesPerBlock = 300
            $puzzlesPerSecond = $puzzlesPerBlock / $avgBlockTime
            
            Write-Host ""
            Write-Host "⚡ Mining Performance" -ForegroundColor Yellow
            Write-Host "====================" -ForegroundColor Yellow
            Write-Host "Avg Block Time: $($avgBlockTime.ToString('F2'))s" -ForegroundColor White
            Write-Host "Min Block Time: $($minBlockTime.ToString('F2'))s" -ForegroundColor White
            Write-Host "Max Block Time: $($maxBlockTime.ToString('F2'))s" -ForegroundColor White
            Write-Host "Est. Quantum Hashrate: $($puzzlesPerSecond.ToString('F2')) puzzles/sec" -ForegroundColor White
            Write-Host "Blocks Sampled: $($blockTimes.Count)" -ForegroundColor Gray
        }
        
        # Show quantum parameters
        Write-Host ""
        Write-Host "⚛️  Quantum Parameters (Current Config)" -ForegroundColor Red
        Write-Host "=======================================" -ForegroundColor Red
        Write-Host "QBits: 12 (4,096 quantum states)" -ForegroundColor White
        Write-Host "T-Gates: 40 (high complexity)" -ForegroundColor White
        Write-Host "Puzzles per Block: 300 (L_net)" -ForegroundColor White
        Write-Host "Retarget Period: 50 blocks" -ForegroundColor White
        Write-Host "Target Block Time: 12 seconds" -ForegroundColor White
        
        # Show mining status
        $mining = Invoke-RpcCall "eth_mining"
        $hashrate = Invoke-RpcCall "eth_hashrate"
        if ($mining -ne $null) {
            Write-Host ""
            Write-Host "🔨 Mining Status" -ForegroundColor Green
            Write-Host "================" -ForegroundColor Green
            Write-Host "Mining: $(if($mining) {'✅ Active'} else {'❌ Stopped'})" -ForegroundColor White
            if ($hashrate) {
                $hashrateVal = [Convert]::ToInt64($hashrate, 16)
                Write-Host "Reported Hashrate: $hashrateVal H/s" -ForegroundColor White
            }
        }
    } else {
        Write-Host "❌ Cannot connect to geth - make sure it's running!" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Press Ctrl+C to exit. Refreshing in ${RefreshInterval}s..." -ForegroundColor Gray
    
    Start-Sleep -Seconds $RefreshInterval
}