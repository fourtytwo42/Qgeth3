#!/usr/bin/env pwsh
# Quantum-Geth v0.9-rc3-hw0 Monitor Script
# Tests and monitors the v0.9-rc3-hw0 quantum blockchain

Write-Host "*** QUANTUM-GETH v0.9-rc3-hw0 MONITOR ***" -ForegroundColor Cyan
Write-Host "Testing v0.9-rc3-hw0 monitor script..." -ForegroundColor Green

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
Write-Host "Testing basic connectivity..." -ForegroundColor Yellow
$blockNumber = Invoke-RpcCall "eth_blockNumber"
if ($blockNumber) {
    $blockNum = [Convert]::ToInt64($blockNumber, 16)
    Write-Host "Current Block: $blockNum" -ForegroundColor Green
} else {
    Write-Host "Cannot connect to geth - make sure it is running!" -ForegroundColor Red
    exit 1
}

# Test quantum APIs
Write-Host "Testing quantum APIs..." -ForegroundColor Yellow
$miningStats = Invoke-RpcCall "qmpow_getMiningStats"
if ($miningStats) {
    Write-Host "Mining Stats API working!" -ForegroundColor Green
    Write-Host "   Hashrate: $($miningStats.hashrate) puzzles/sec" -ForegroundColor White
} else {
    Write-Host "Quantum APIs not available (normal if not mining)" -ForegroundColor Yellow
}

# Get latest block with v0.9-rc3-hw0 quantum fields
Write-Host "Testing v0.9-rc3-hw0 block structure..." -ForegroundColor Yellow
$latestBlock = Invoke-RpcCall "eth_getBlockByNumber" @("latest", $true)
if ($latestBlock) {
    Write-Host "Latest Block Retrieved!" -ForegroundColor Green
    Write-Host "   Block Number: $([Convert]::ToInt64($latestBlock.number, 16))" -ForegroundColor White
    Write-Host "   Block Hash: $($latestBlock.hash)" -ForegroundColor White
    Write-Host "   Difficulty: $([Convert]::ToInt64($latestBlock.difficulty, 16))" -ForegroundColor White
    
    # Check for v0.9-rc3-hw0 quantum fields
    if ($latestBlock.epoch -ne $null) {
        Write-Host "   v0.9-rc3-hw0 Quantum Fields Detected:" -ForegroundColor Magenta
        Write-Host "      Epoch: $([Convert]::ToInt64($latestBlock.epoch, 16))" -ForegroundColor Gray
        Write-Host "      QBits: $([Convert]::ToInt64($latestBlock.qbits, 16))" -ForegroundColor Gray
        Write-Host "      TCount: $([Convert]::ToInt64($latestBlock.tcount, 16))" -ForegroundColor Gray
        Write-Host "      LNet: $([Convert]::ToInt64($latestBlock.lnet, 16))" -ForegroundColor Gray
        Write-Host "      QNonce64: $([Convert]::ToInt64($latestBlock.qnonce64, 16))" -ForegroundColor Gray
        Write-Host "      ExtraNonce32: $($latestBlock.extranonce32)" -ForegroundColor Gray
        Write-Host "      OutcomeRoot: $($latestBlock.outcomeroot)" -ForegroundColor Gray
        Write-Host "      BranchNibbles: $($latestBlock.branchnibbles)" -ForegroundColor Gray
        Write-Host "      GateHash: $($latestBlock.gatehash)" -ForegroundColor Gray
        Write-Host "      ProofRoot: $($latestBlock.proofroot)" -ForegroundColor Gray
        Write-Host "      AttestMode: $([Convert]::ToInt64($latestBlock.attestmode, 16))" -ForegroundColor Gray
    } else {
        Write-Host "   Genesis block - no quantum fields yet" -ForegroundColor Yellow
    }
} else {
    Write-Host "Failed to retrieve latest block" -ForegroundColor Red
}

Write-Host ""
Write-Host "v0.9-rc3-hw0 monitor test completed!" -ForegroundColor Green 