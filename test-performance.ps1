#!/usr/bin/env pwsh
# Performance Test Script for Optimized Quantum Miner
# Tests the miner for 60 seconds and reports key performance metrics

param(
    [string]$NodeUrl = "http://localhost:8545",
    [string]$Coinbase = "0x0000000000000000000000000000000000000001",
    [int]$Threads = 4,
    [int]$TestDuration = 60
)

Write-Host "🚀 Quantum Miner Performance Test" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host "Node URL: $NodeUrl"
Write-Host "Threads: $Threads"
Write-Host "Test Duration: $TestDuration seconds"
Write-Host ""

# Build the miner if needed
if (-not (Test-Path "quantum-miner.exe")) {
    Write-Host "📦 Building quantum miner..." -ForegroundColor Yellow
    go build -o quantum-miner.exe quantum-miner/main.go
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to build quantum miner" -ForegroundColor Red
        exit 1
    }
}

# Start the miner
Write-Host "🏁 Starting performance test..." -ForegroundColor Green
$startTime = Get-Date

$minerArgs = @(
    "-coinbase", $Coinbase,
    "-node", $NodeUrl,
    "-threads", $Threads,
    "-log", "true"
)

$minerProcess = Start-Process -FilePath ".\quantum-miner.exe" -ArgumentList $minerArgs -PassThru -NoNewWindow

Write-Host "⏱️  Test running for $TestDuration seconds..." -ForegroundColor Yellow

# Wait for test duration
Start-Sleep -Seconds $TestDuration

# Stop the miner
Write-Host "🛑 Stopping miner..." -ForegroundColor Yellow
$minerProcess.Kill()
$minerProcess.WaitForExit(5000)

$endTime = Get-Date
$actualDuration = ($endTime - $startTime).TotalSeconds

Write-Host ""
Write-Host "📊 Performance Test Results" -ForegroundColor Green
Write-Host "============================" -ForegroundColor Green
Write-Host "Test Duration: $actualDuration seconds"

# Find the most recent log file
$logFiles = Get-ChildItem -Filter "quantum-miner-*.log" | Sort-Object LastWriteTime -Descending
if ($logFiles.Count -gt 0) {
    $logFile = $logFiles[0].FullName
    Write-Host "📝 Analyzing log file: $($logFiles[0].Name)"
    
    # Analyze performance metrics from log
    $logContent = Get-Content $logFile
    
    # Count iterations completed
    $iterationLines = $logContent | Select-String "(\d+) iterations.*?(\d+\.\d+) iter/sec"
    if ($iterationLines.Count -gt 0) {
        $iterationRates = @()
        foreach ($line in $iterationLines) {
            if ($line -match "(\d+\.\d+) iter/sec") {
                $iterationRates += [double]$matches[1]
            }
        }
        
        if ($iterationRates.Count -gt 0) {
            $avgIterationRate = ($iterationRates | Measure-Object -Average).Average
            $maxIterationRate = ($iterationRates | Measure-Object -Maximum).Maximum
            
            Write-Host "🔄 Average Iteration Rate: $($avgIterationRate.ToString('F2')) iter/sec" -ForegroundColor Cyan
            Write-Host "⚡ Peak Iteration Rate: $($maxIterationRate.ToString('F2')) iter/sec" -ForegroundColor Cyan
            
            # Expected improvement
            if ($avgIterationRate -gt 0.5) {
                Write-Host "✅ PERFORMANCE IMPROVED: Achieving >0.5 iter/sec (vs ~0.1 previously)" -ForegroundColor Green
            } else {
                Write-Host "⚠️  Performance may need further optimization" -ForegroundColor Yellow
            }
        }
    }
    
    # Count slow iteration warnings
    $slowIterations = ($logContent | Select-String "Very slow iteration took").Count
    Write-Host "⚠️  Slow Iterations (>5s): $slowIterations"
    
    # Count solutions found
    $solutions = ($logContent | Select-String "Solution accepted").Count
    Write-Host "🎯 Solutions Found: $solutions"
    
    # Check work change detection
    $workChanges = ($logContent | Select-String "NEW WORK DETECTED").Count
    Write-Host "🔄 Work Changes Detected: $workChanges"
    
    # Check if threads are switching to new work
    $workSwitches = ($logContent | Select-String "Successfully switched to new work").Count
    Write-Host "✅ Successful Work Switches: $workSwitches"
    
    # Check stale work detection
    $staleWork = ($logContent | Select-String "Stale work detected").Count
    Write-Host "🚫 Stale Work Detected: $staleWork"
    
    # Check work fetcher performance
    $workFetcher = ($logContent | Select-String "Work fetcher started \(200ms interval\)").Count
    if ($workFetcher -gt 0) {
        Write-Host "✅ Fast work fetcher (200ms): ACTIVE" -ForegroundColor Green
    } else {
        Write-Host "❌ Fast work fetcher not detected" -ForegroundColor Red
    }
    
    # Check for specific performance improvements
    $workGetWarnings = ($logContent | Select-String "Slow getWork\(\) took").Count
    Write-Host "📦 Work Get Warnings (should be 0): $workGetWarnings"
    
    if ($workGetWarnings -eq 0) {
        Write-Host "✅ Work package pool optimization: SUCCESS" -ForegroundColor Green
    } else {
        Write-Host "❌ Work package pool still has issues" -ForegroundColor Red
    }
    
    # Calculate stale work efficiency
    if ($staleWork -gt 0 -and $workChanges -gt 0) {
        $staleWorkRate = ($staleWork / $workChanges) * 100
        Write-Host "📊 Stale Work Rate: $($staleWorkRate.ToString('F1'))% (lower is better)" -ForegroundColor Yellow
        
        if ($staleWorkRate -lt 20) {
            Write-Host "✅ Stale work rate: GOOD (<20%)" -ForegroundColor Green
        } elseif ($staleWorkRate -lt 50) {
            Write-Host "⚠️  Stale work rate: MODERATE (20-50%)" -ForegroundColor Yellow
        } else {
            Write-Host "❌ Stale work rate: HIGH (>50%)" -ForegroundColor Red
        }
    }
    
} else {
    Write-Host "❌ No log file found - logging may be disabled" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎯 Performance Targets:" -ForegroundColor Yellow
Write-Host "   • Iteration Rate: >1.0 iter/sec (target: 5-10 iter/sec)"
Write-Host "   • Work Change Detection: >0 (should detect new blocks)"
Write-Host "   • Work Switch Success: >80% of work changes"
Write-Host "   • Stale Work Rate: <20% (indicates fast work switching)"
Write-Host "   • Work Fetcher: 200ms interval (vs 2s previously)"
Write-Host "   • Work Get Warnings: 0 (indicates no pool thrashing)"

Write-Host ""
Write-Host "✅ Performance test completed!" -ForegroundColor Green
Write-Host ""
Write-Host "🔧 Key Optimizations Applied:" -ForegroundColor Cyan
Write-Host "   1. Work fetcher interval: 2s → 200ms (10x faster)"
Write-Host "   2. Work change detection: 10,000 → 1,000 qnonces (10x more frequent)"
Write-Host "   3. Work package pool: Fixed thrashing issue"
Write-Host "   4. Stale work detection: Immediate abandonment of old work"
Write-Host "   5. Buffered logging: Reduced I/O blocking" 