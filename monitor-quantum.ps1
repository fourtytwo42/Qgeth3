#!/usr/bin/env pwsh
# Quantum Blockchain Monitor
# Displays real-time hashrate, difficulty, and network statistics

param(
    [string]$RpcUrl = "http://localhost:8545",
    [int]$RefreshInterval = 5
)

Write-Host "🚀 Quantum Blockchain Monitor" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "RPC URL: $RpcUrl" -ForegroundColor Gray
Write-Host "Refresh: ${RefreshInterval}s" -ForegroundColor Gray
Write-Host ""

function Invoke-RpcCall {
    param(
        [string]$Method,
        [array]$Params = @()
    )
    
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

function Format-Number {
    param([double]$Number)
    
    if ($Number -ge 1000000) {
        return "{0:N2}M" -f ($Number / 1000000)
    } elseif ($Number -ge 1000) {
        return "{0:N2}K" -f ($Number / 1000)
    } else {
        return "{0:N2}" -f $Number
    }
}

while ($true) {
    Clear-Host
    
    Write-Host "🚀 Quantum Blockchain Monitor" -ForegroundColor Cyan
    Write-Host "==============================" -ForegroundColor Cyan
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
    
    # Get block number
    $blockNumber = Invoke-RpcCall "eth_blockNumber"
    if ($blockNumber) {
        $blockNum = [Convert]::ToInt64($blockNumber, 16)
        Write-Host "📦 Current Block: $blockNum" -ForegroundColor Green
    }
    
    # Get mining stats
    $miningStats = Invoke-RpcCall "qmpow_getMiningStats"
    if ($miningStats) {
        Write-Host ""
        Write-Host "⚡ Mining Statistics" -ForegroundColor Yellow
        Write-Host "===================" -ForegroundColor Yellow
        
        $hashrate = [double]$miningStats.hashrate
        $threads = $miningStats.threads
        
        Write-Host "Hashrate: $(Format-Number $hashrate) puzzles/sec" -ForegroundColor White
        Write-Host "Threads:  $threads" -ForegroundColor White
        
        if ($miningStats.difficulty) {
            $difficulty = $miningStats.difficulty
            Write-Host ""
            Write-Host "🔧 Difficulty Settings" -ForegroundColor Magenta
            Write-Host "======================" -ForegroundColor Magenta
            Write-Host "Puzzles per Block: $($difficulty.puzzlesPerBlock)" -ForegroundColor White
            Write-Host "Qubits per Puzzle: $($difficulty.qubitsPerPuzzle)" -ForegroundColor White
            Write-Host "T-gates per Puzzle: $($difficulty.tgatesPerPuzzle)" -ForegroundColor White
            Write-Host "Total Complexity: $(Format-Number $difficulty.totalComplexity)" -ForegroundColor White
        }
        
        Write-Host ""
        Write-Host "⏱️  Timing" -ForegroundColor Blue
        Write-Host "==========" -ForegroundColor Blue
        Write-Host "Target Block Time: $($miningStats.targetBlockTime)s" -ForegroundColor White
        Write-Host "Retarget Period: $($miningStats.retargetPeriod) blocks" -ForegroundColor White
    }
    
    # Get network stats
    $networkStats = Invoke-RpcCall "qmpow_getNetworkStats"
    if ($networkStats) {
        Write-Host ""
        Write-Host "🌐 Network Statistics" -ForegroundColor Cyan
        Write-Host "=====================" -ForegroundColor Cyan
        
        $theoreticalHashrate = [double]$networkStats.theoreticalHashrate
        Write-Host "Current Difficulty: $($networkStats.currentDifficulty)" -ForegroundColor White
        Write-Host "Theoretical Hashrate: $(Format-Number $theoreticalHashrate) puzzles/sec" -ForegroundColor White
        
        if ($networkStats.quantumComplexity) {
            $qc = $networkStats.quantumComplexity
            Write-Host ""
            Write-Host "⚛️  Quantum Complexity" -ForegroundColor Red
            Write-Host "======================" -ForegroundColor Red
            Write-Host "Qubits: $($qc.qubits)" -ForegroundColor White
            Write-Host "T-gates: $($qc.tgates)" -ForegroundColor White
            Write-Host "State Space: $(Format-Number $qc.stateSpace) states" -ForegroundColor White
        }
    }
    
    # Get quantum params
    $quantumParams = Invoke-RpcCall "qmpow_getQuantumParams" @("latest")
    if ($quantumParams) {
        Write-Host ""
        Write-Host "📊 Quantum Parameters" -ForegroundColor Green
        Write-Host "=====================" -ForegroundColor Green
        Write-Host "QBits: $($quantumParams.qbits)" -ForegroundColor White
        Write-Host "TCount: $($quantumParams.tcount)" -ForegroundColor White
        Write-Host "LNet: $($quantumParams.lnet)" -ForegroundColor White
        Write-Host "Epoch Length: $($quantumParams.epochLen)" -ForegroundColor White
    }
    
    Write-Host ""
    Write-Host "Press Ctrl+C to exit. Refreshing in ${RefreshInterval}s..." -ForegroundColor Gray
    
    Start-Sleep -Seconds $RefreshInterval
} 