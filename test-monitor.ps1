#!/usr/bin/env pwsh
# Simple test monitor

Write-Host "Testing monitor script..." -ForegroundColor Green

function Invoke-RpcCall {
    param([string]$Method, [array]$Params = @())
    
    $body = @{
        jsonrpc = "2.0"
        method = $Method
        params = $Params
        id = 1
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -Body $body -ContentType "application/json"
        return $response.result
    } catch {
        Write-Host "RPC call failed: $Method" -ForegroundColor Red
        return $null
    }
}

# Test basic RPC call
$blockNumber = Invoke-RpcCall "eth_blockNumber"
if ($blockNumber) {
    $blockNum = [Convert]::ToInt64($blockNumber, 16)
    Write-Host "Current Block: $blockNum" -ForegroundColor Green
} else {
    Write-Host "Cannot connect to geth - make sure it's running!" -ForegroundColor Red
}

# Test quantum APIs
$miningStats = Invoke-RpcCall "qmpow_getMiningStats"
if ($miningStats) {
    Write-Host "Mining Stats API working!" -ForegroundColor Green
    Write-Host "Hashrate: $($miningStats.hashrate) puzzles/sec" -ForegroundColor White
} else {
    Write-Host "Quantum APIs not available" -ForegroundColor Red
} 