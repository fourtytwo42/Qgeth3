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
        if ($response.error) {
            Write-Host "RPC Error: $($response.error.message)" -ForegroundColor Red
            return $null
        }
        return $response.result
    } catch {
        Write-Host "RPC call failed: $Method - $($_.Exception.Message)" -ForegroundColor Red
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

# Test mining status
Write-Host "Testing mining status..." -ForegroundColor Yellow
$isMining = Invoke-RpcCall "eth_mining"
if ($isMining -ne $null) {
    if ($isMining) {
        Write-Host "Mining: ACTIVE" -ForegroundColor Green
        
        # Get hashrate if mining
        $hashrate = Invoke-RpcCall "eth_hashrate"
        if ($hashrate) {
            $hashrateNum = [Convert]::ToInt64($hashrate, 16)
            Write-Host "   Hashrate: $hashrateNum H/s" -ForegroundColor White
        }
    } else {
        Write-Host "Mining: INACTIVE" -ForegroundColor Yellow
    }
} else {
    Write-Host "Mining status unavailable" -ForegroundColor Yellow
}

# Test quantum APIs
Write-Host "Testing quantum APIs..." -ForegroundColor Yellow
$miningStats = Invoke-RpcCall "qmpow_getMiningStats"
if ($miningStats) {
    Write-Host "QMPoW Mining Stats API working!" -ForegroundColor Green
    if ($miningStats.hashrate) {
        Write-Host "   Quantum Hashrate: $($miningStats.hashrate) attempts/sec" -ForegroundColor White
    }
    if ($miningStats.totalBlocks) {
        Write-Host "   Total Quantum Blocks: $($miningStats.totalBlocks)" -ForegroundColor White
    }
} else {
    Write-Host "QMPoW APIs not available (normal if not mining)" -ForegroundColor Yellow
}

# Get latest block with v0.9-rc3-hw0 quantum fields
Write-Host "Testing v0.9-rc3-hw0 block structure..." -ForegroundColor Yellow
$latestBlock = Invoke-RpcCall "eth_getBlockByNumber" @("latest", $true)
if ($latestBlock) {
    Write-Host "Latest Block Retrieved!" -ForegroundColor Green
    Write-Host "   Block Number: $([Convert]::ToInt64($latestBlock.number, 16))" -ForegroundColor White
    Write-Host "   Block Hash: $($latestBlock.hash)" -ForegroundColor White
    Write-Host "   Difficulty: $([Convert]::ToInt64($latestBlock.difficulty, 16))" -ForegroundColor White
    Write-Host "   Gas Used: $([Convert]::ToInt64($latestBlock.gasUsed, 16))" -ForegroundColor White
    Write-Host "   Timestamp: $([Convert]::ToInt64($latestBlock.timestamp, 16))" -ForegroundColor White
    
    # Check for v0.9-rc3-hw0 quantum fields (these are virtual fields populated from QBlob)
    if ($latestBlock.epoch -ne $null) {
        Write-Host "   v0.9-rc3-hw0 Quantum Fields Detected:" -ForegroundColor Magenta
        Write-Host "      Epoch: $([Convert]::ToInt64($latestBlock.epoch, 16))" -ForegroundColor Gray
        Write-Host "      QBits: $([Convert]::ToInt64($latestBlock.qBits, 16))" -ForegroundColor Gray
        Write-Host "      TCount: $([Convert]::ToInt64($latestBlock.tCount, 16))" -ForegroundColor Gray
        Write-Host "      LNet: $([Convert]::ToInt64($latestBlock.lNet, 16))" -ForegroundColor Gray
        Write-Host "      QNonce64: $([Convert]::ToInt64($latestBlock.qNonce64, 16))" -ForegroundColor Gray
        if ($latestBlock.extraNonce32) {
            Write-Host "      ExtraNonce32: $($latestBlock.extraNonce32)" -ForegroundColor Gray
        }
        if ($latestBlock.outcomeRoot) {
            Write-Host "      OutcomeRoot: $($latestBlock.outcomeRoot)" -ForegroundColor Gray
        }
        if ($latestBlock.branchNibbles) {
            Write-Host "      BranchNibbles: $($latestBlock.branchNibbles)" -ForegroundColor Gray
        }
        if ($latestBlock.gateHash) {
            Write-Host "      GateHash: $($latestBlock.gateHash)" -ForegroundColor Gray
        }
        if ($latestBlock.proofRoot) {
            Write-Host "      ProofRoot: $($latestBlock.proofRoot)" -ForegroundColor Gray
        }
        if ($latestBlock.attestMode -ne $null) {
            Write-Host "      AttestMode: $([Convert]::ToInt64($latestBlock.attestMode, 16))" -ForegroundColor Gray
        }
    } else {
        Write-Host "   No quantum fields detected (genesis block or pre-quantum)" -ForegroundColor Yellow
    }
    
    # Check for quantum blob
    if ($latestBlock.qBlob) {
        Write-Host "   Quantum Blob Present: $($latestBlock.qBlob.Length) bytes" -ForegroundColor Cyan
    } else {
        Write-Host "   No quantum blob found" -ForegroundColor Yellow
    }
} else {
    Write-Host "Failed to retrieve latest block" -ForegroundColor Red
}

# Check peer count
Write-Host "Testing network status..." -ForegroundColor Yellow
$peerCount = Invoke-RpcCall "net_peerCount"
if ($peerCount -ne $null) {
    $peers = [Convert]::ToInt64($peerCount, 16)
    Write-Host "Network Peers: $peers" -ForegroundColor Green
} else {
    Write-Host "Peer count unavailable" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "v0.9-rc3-hw0 monitor test completed!" -ForegroundColor Green
Write-Host "Monitor script updated for current quantum-geth implementation." -ForegroundColor Cyan 