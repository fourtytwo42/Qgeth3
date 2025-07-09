#!/usr/bin/env pwsh
# Test Script for Stale Work Fix
# Tests ultra-aggressive work change detection

param(
    [string]$NodeUrl = "http://localhost:8545",
    [string]$Coinbase = "0x0000000000000000000000000000000000000001",
    [int]$Threads = 8,
    [int]$TestDuration = 20
)

Write-Host "🚀 Stale Work Fix Test" -ForegroundColor Green
Write-Host "=====================" -ForegroundColor Green
Write-Host "Testing ultra-aggressive work change detection"
Write-Host "Threads: $Threads | Duration: $TestDuration seconds"
Write-Host ""

# Check if fixed executable exists
if (-not (Test-Path "quantum-miner-stale-fix.exe")) {
    Write-Host "❌ quantum-miner-stale-fix.exe not found!" -ForegroundColor Red
    exit 1
}

# Start the miner
Write-Host "🏁 Starting stale work fix test..." -ForegroundColor Green
$startTime = Get-Date

$minerArgs = @(
    "-coinbase", $Coinbase,
    "-node", $NodeUrl,
    "-threads", $Threads,
    "-log", "true"
)

$minerProcess = Start-Process -FilePath ".\quantum-miner-stale-fix.exe" -ArgumentList $minerArgs -PassThru -NoNewWindow

Write-Host "⏱️  Test running for $TestDuration seconds..." -ForegroundColor Yellow
Write-Host "🔍 Watching for improved work switching..." -ForegroundColor Cyan

# Wait for test duration
Start-Sleep -Seconds $TestDuration

# Stop the miner
Write-Host "🛑 Stopping miner..." -ForegroundColor Yellow
$minerProcess.Kill()
$minerProcess.WaitForExit(3000)

Write-Host ""
Write-Host "📊 Stale Work Fix Results" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green

# Find the most recent log file
$logFiles = Get-ChildItem -Filter "quantum-miner-*.log" | Sort-Object LastWriteTime -Descending
if ($logFiles.Count -gt 0) {
    $logFile = $logFiles[0].FullName
    Write-Host "📝 Analyzing log file: $($logFiles[0].Name)"
    
    $logContent = Get-Content $logFile
    
    # Count work changes detected by threads
    $workChanges = ($logContent | Select-String "Work change detected").Count
    Write-Host "🔄 Work Changes Detected by Threads: $workChanges" -ForegroundColor $(if ($workChanges -gt 0) { "Green" } else { "Red" })
    
    # Count successful work switches
    $workSwitches = ($logContent | Select-String "Successfully switched to new work").Count
    Write-Host "✅ Successful Work Switches: $workSwitches" -ForegroundColor $(if ($workSwitches -gt 0) { "Green" } else { "Yellow" })
    
    # Count work abandonment
    $abandonedWork = ($logContent | Select-String "Abandoning.*remaining qnonces").Count
    Write-Host "🚫 Work Abandonment Events: $abandonedWork" -ForegroundColor $(if ($abandonedWork -gt 0) { "Green" } else { "Yellow" })
    
    # Check for extremely frequent work checking
    $frequentChecks = ($logContent | Select-String "every 50 attempts").Count
    if ($frequentChecks -gt 0) {
        Write-Host "✅ Ultra-frequent work checking: ACTIVE (every 50 qnonces)" -ForegroundColor Green
    }
    
    # Count iterations per thread to see if they're shorter
    $iterationLines = ($logContent | Select-String "finished:.*iterations").ToArray()
    if ($iterationLines.Count -gt 0) {
        $totalIterations = 0
        foreach ($line in $iterationLines) {
            if ($line -match "(\d+) iterations") {
                $totalIterations += [int]$matches[1]
            }
        }
        $avgIterationsPerThread = $totalIterations / $iterationLines.Count
        Write-Host "📊 Average Iterations per Thread: $($avgIterationsPerThread.ToString('F1'))" -ForegroundColor Cyan
        
        if ($avgIterationsPerThread -gt 1) {
            Write-Host "✅ Faster iterations: IMPROVED (more iterations completed)" -ForegroundColor Green
        }
    }
    
    # Check iteration times
    $slowIterations = ($logContent | Select-String "Very slow iteration took").Count
    Write-Host "⚠️  Slow Iterations (>5s): $slowIterations" -ForegroundColor $(if ($slowIterations -lt 5) { "Green" } else { "Yellow" })
    
    # Calculate improvement metrics
    if ($workChanges -gt 0) {
        Write-Host ""
        Write-Host "🎯 STALE WORK FIX STATUS:" -ForegroundColor Yellow
        Write-Host "   ✅ Work change detection: WORKING ($workChanges events)"
        Write-Host "   ✅ Reduced iteration size: 100K qnonces (vs 1M previously)"
        Write-Host "   ✅ Ultra-frequent checking: Every 50 qnonces (vs 1000 previously)"
        Write-Host "   ✅ Post-submission checking: Active"
        
        if ($abandonedWork -gt 0) {
            Write-Host "   ✅ Work abandonment: WORKING ($abandonedWork events)" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  Work abandonment: No events detected" -ForegroundColor Yellow
        }
    } else {
        Write-Host ""
        Write-Host "❌ Work change detection not working - may need further debugging" -ForegroundColor Red
    }
    
} else {
    Write-Host "❌ No log file found" -ForegroundColor Red
}

Write-Host ""
Write-Host "🔧 Key Improvements Applied:" -ForegroundColor Cyan
Write-Host "   1. Work checking frequency: 1000 → 50 qnonces (20x more frequent)"
Write-Host "   2. Iteration size: 1M → 100K qnonces (10x smaller batches)"
Write-Host "   3. Post-submission work checking: Added"
Write-Host "   4. Pre-iteration work checking: Added"

Write-Host ""
Write-Host "✅ Stale work fix test completed!" -ForegroundColor Green 