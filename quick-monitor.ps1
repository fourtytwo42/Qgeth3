#!/usr/bin/env pwsh
# Quick Quantum Blockchain Monitor with Block Times

Write-Host "🚀 Quantum Blockchain Monitor" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

$lastBlockNum = 0
$lastBlockTime = 0

while ($true) {
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    # Get current block
    $body = '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -Body $body -ContentType "application/json"
        $blockNum = [Convert]::ToInt64($response.result, 16)
        
        # Get current block details
        $blockBody = "{`"jsonrpc`":`"2.0`",`"method`":`"eth_getBlockByNumber`",`"params`":[`"latest`",false],`"id`":1}"
        $blockResponse = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -Body $blockBody -ContentType "application/json"
        $difficulty = [Convert]::ToInt64($blockResponse.result.difficulty, 16)
        $currentBlockTime = [Convert]::ToInt64($blockResponse.result.timestamp, 16)
        
        # Calculate block time if we have a new block
        $blockTimeDisplay = ""
        if ($blockNum -gt $lastBlockNum -and $lastBlockTime -gt 0) {
            $blockTimeDiff = $currentBlockTime - $lastBlockTime
            $blockTimeDisplay = " | Block Time: $($blockTimeDiff)s"
        }
        
        # Update tracking variables
        if ($blockNum -gt $lastBlockNum) {
            $lastBlockNum = $blockNum
            $lastBlockTime = $currentBlockTime
        }
        
        # Get mining status
        $miningBody = '{"jsonrpc":"2.0","method":"eth_mining","params":[],"id":1}'
        $miningResponse = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -Body $miningBody -ContentType "application/json"
        $isMining = $miningResponse.result
        
        Write-Host "[$timestamp] Block: $blockNum | Difficulty: $difficulty | Mining: $(if($isMining){'✅'}else{'❌'})$blockTimeDisplay" -ForegroundColor Green
        
    } catch {
        Write-Host "[$timestamp] ❌ Connection failed" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 2
} 