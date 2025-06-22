#!/usr/bin/env pwsh
# Quantum-Geth v0.9-rc3-hw0 Monitor
# Unified, Branch-Serial Quantum Proof-of-Work — Canonical-Compile Edition

param(
    [int]$refreshSeconds = 5,
    [string]$rpcUrl = "http://localhost:8545"
)

Write-Host "*** QUANTUM-GETH v0.9-rc3-hw0 MONITOR ***" -ForegroundColor Cyan
Write-Host "Monitoring quantum blockchain with epochic glide and canonical compiler" -ForegroundColor Green
Write-Host "Refresh interval: $refreshSeconds seconds" -ForegroundColor Yellow
Write-Host "RPC URL: $rpcUrl" -ForegroundColor Yellow
Write-Host ""

function Invoke-RpcCall {
    param([string]$Method, [array]$Params = @())
    
    $body = @{
        jsonrpc = "2.0"
        method = $Method
        params = $Params
        id = 1
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-RestMethod -Uri $rpcUrl -Method Post -Body $body -ContentType "application/json"
        return $response.result
    } catch {
        return $null
    }
}

function Show-QuantumStatus {
    Write-Host "=== QUANTUM-GETH v0.9-rc3-hw0 STATUS ===" -ForegroundColor Cyan
    
    # Get basic blockchain info
    $blockNumber = Invoke-RpcCall "eth_blockNumber"
    $isMining = Invoke-RpcCall "eth_mining"
    $hashrate = Invoke-RpcCall "eth_hashrate"
    
    if ($blockNumber) {
        $blockNum = [Convert]::ToInt64($blockNumber, 16)
        Write-Host "Current Block: $blockNum" -ForegroundColor Green
        
        # Calculate v0.9-rc3-hw0 parameters
        $epoch = [Math]::Floor($blockNum / 50000)
        $glideSteps = [Math]::Floor($blockNum / 12500)
        $currentQBits = 12 + $glideSteps
        
        Write-Host "Epoch: $epoch (⌊$blockNum / 50,000⌋)" -ForegroundColor Yellow
        Write-Host "QBits: $currentQBits (12 + $glideSteps glide steps)" -ForegroundColor Yellow
        Write-Host "TCount: 4,096 (constant)" -ForegroundColor Yellow
        Write-Host "LNet: 48 puzzles (constant, 1,152-bit security)" -ForegroundColor Yellow
    } else {
        Write-Host "❌ Cannot connect to geth node!" -ForegroundColor Red
        return
    }
    
    # Mining status
    if ($isMining) {
        Write-Host "Mining: ACTIVE ⛏️" -ForegroundColor Green
        if ($hashrate) {
            $hashrateDecimal = [Convert]::ToInt64($hashrate, 16)
            Write-Host "Hashrate: $hashrateDecimal attempts/sec" -ForegroundColor Green
        }
    } else {
        Write-Host "Mining: INACTIVE" -ForegroundColor Red
    }
    
    # Get latest block details
    $latestBlock = Invoke-RpcCall "eth_getBlockByNumber" @("latest", $true)
    if ($latestBlock) {
        Write-Host ""
        Write-Host "=== LATEST BLOCK QUANTUM FIELDS ===" -ForegroundColor Magenta
        
        if ($latestBlock.epoch) {
            $epochValue = [Convert]::ToInt64($latestBlock.epoch, 16)
            Write-Host "Epoch: $epochValue" -ForegroundColor White
        }
        
        if ($latestBlock.qBits) {
            $qbitsValue = [Convert]::ToInt64($latestBlock.qBits, 16)
            Write-Host "QBits: $qbitsValue" -ForegroundColor White
        }
        
        if ($latestBlock.tCount) {
            $tcountValue = [Convert]::ToInt64($latestBlock.tCount, 16)
            Write-Host "TCount: $tcountValue" -ForegroundColor White
        }
        
        if ($latestBlock.lNet) {
            $lnetValue = [Convert]::ToInt64($latestBlock.lNet, 16)
            Write-Host "LNet: $lnetValue puzzles" -ForegroundColor White
        }
        
        if ($latestBlock.qNonce64) {
            $qnonceValue = [Convert]::ToInt64($latestBlock.qNonce64, 16)
            Write-Host "QNonce64: $qnonceValue" -ForegroundColor White
        }
        
        if ($latestBlock.outcomeRoot) {
            $outcomeRoot = $latestBlock.outcomeRoot.Substring(0, 10) + "..."
            Write-Host "OutcomeRoot: $outcomeRoot" -ForegroundColor White
        }
        
        if ($latestBlock.gateHash) {
            $gateHash = $latestBlock.gateHash.Substring(0, 10) + "..."
            Write-Host "GateHash: $gateHash" -ForegroundColor White
        }
        
        if ($latestBlock.proofRoot) {
            $proofRoot = $latestBlock.proofRoot.Substring(0, 10) + "..."
            Write-Host "ProofRoot: $proofRoot" -ForegroundColor White
        }
        
        if ($latestBlock.attestMode) {
            $attestValue = [Convert]::ToInt64($latestBlock.attestMode, 16)
            $attestName = if ($attestValue -eq 0) { "Dilithium" } else { "Unknown" }
            Write-Host "AttestMode: $attestValue ($attestName)" -ForegroundColor White
        }
        
        # Block timing
        if ($latestBlock.timestamp) {
            $timestamp = [Convert]::ToInt64($latestBlock.timestamp, 16)
            $blockTime = [DateTimeOffset]::FromUnixTimeSeconds($timestamp).ToString("yyyy-MM-dd HH:mm:ss")
            Write-Host "Block Time: $blockTime" -ForegroundColor Gray
        }
        
        # Difficulty
        if ($latestBlock.difficulty) {
            $difficulty = [Convert]::ToInt64($latestBlock.difficulty, 16)
            Write-Host "Difficulty: $difficulty" -ForegroundColor Gray
        }
    }
    
    # Get quantum mining stats
    $miningStats = Invoke-RpcCall "qmpow_getMiningStats"
    if ($miningStats) {
        Write-Host ""
        Write-Host "=== QUANTUM MINING STATS ===" -ForegroundColor Magenta
        Write-Host "Hashrate: $($miningStats.hashrate) $($miningStats.hashrateUnit)" -ForegroundColor White
        Write-Host "Threads: $($miningStats.threads)" -ForegroundColor White
        Write-Host "Target Block Time: $($miningStats.targetBlockTime)s" -ForegroundColor White
        Write-Host "Retarget Period: $($miningStats.retargetPeriod) blocks" -ForegroundColor White
        
        if ($miningStats.quantumComplexity) {
            $complexity = $miningStats.quantumComplexity
            Write-Host "Quantum State Space: 2^$($complexity.qubits) = $($complexity.stateSpace)" -ForegroundColor Gray
            Write-Host "T-Gates per Puzzle: $($complexity.tgates)" -ForegroundColor Gray
        }
    }
    
    Write-Host ""
    Write-Host "Last updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
}

# Main monitoring loop
try {
    while ($true) {
        Clear-Host
        Show-QuantumStatus
        Start-Sleep -Seconds $refreshSeconds
    }
} catch {
    Write-Host ""
    Write-Host "Monitoring stopped." -ForegroundColor Yellow
} 