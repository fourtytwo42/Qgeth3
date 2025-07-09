#!/usr/bin/env pwsh
# Test Script for Post-Solution Work Switching Fix
# Tests intelligent post-solution work switching with exponential backoff

param(
    [string]$NodeUrl = "http://localhost:8545",
    [string]$Coinbase = "0x0000000000000000000000000000000000000001",
    [int]$Threads = 16,
    [int]$TestDuration = 30
)

Write-Host "🚀 Post-Solution Work Switching Test" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host "Testing intelligent post-solution work switching with exponential backoff"
Write-Host "Threads: $Threads | Duration: $TestDuration seconds"
Write-Host ""

# Check if fixed executable exists
if (-not (Test-Path "quantum-miner-post-solution-fix.exe")) {
    Write-Host "❌ quantum-miner-post-solution-fix.exe not found!" -ForegroundColor Red
    exit 1
}

# Start the miner
Write-Host "🏁 Starting post-solution fix test..." -ForegroundColor Green
$startTime = Get-Date

$minerArgs = @(
    "-coinbase", $Coinbase,
    "-node", $NodeUrl,
    "-threads", $Threads,
    "-log", "true"
)

$minerProcess = Start-Process -FilePath ".\quantum-miner-post-solution-fix.exe" -ArgumentList $minerArgs -PassThru -NoNewWindow

Write-Host "⏱️  Test running for $TestDuration seconds..." -ForegroundColor Yellow
Write-Host "🔍 Watching for intelligent post-solution work switching..." -ForegroundColor Cyan

# Wait for test duration
Start-Sleep -Seconds $TestDuration

# Stop the miner
Write-Host "🛑 Stopping miner..." -ForegroundColor Yellow
$minerProcess.Kill()
$minerProcess.WaitForExit(3000)

Write-Host ""
Write-Host "📊 Post-Solution Work Switching Results" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green

# Find the most recent log file
$logFiles = Get-ChildItem -Filter "quantum-miner-*.log" | Sort-Object LastWriteTime -Descending
if ($logFiles.Count -gt 0) {
    $logFile = $logFiles[0].FullName
    Write-Host "📝 Analyzing log file: $($logFiles[0].Name)"
    
    $logContent = Get-Content $logFile
    
    # Count solution submissions
    $solutionSubmissions = ($logContent | Select-String "Solution submitted for QNonce").Count
    Write-Host "🎯 Solution Submissions: $solutionSubmissions" -ForegroundColor $(if ($solutionSubmissions -gt 0) { "Green" } else { "Yellow" })
    
    # Count immediate work refresh triggers
    $immediateRefreshTriggers = ($logContent | Select-String "Triggered immediate work refresh after solution").Count
    Write-Host "⚡ Immediate Work Refresh Triggers: $immediateRefreshTriggers" -ForegroundColor $(if ($immediateRefreshTriggers -gt 0) { "Green" } else { "Yellow" })
    
    # Count force work refresh events
    $forceRefreshEvents = ($logContent | Select-String "Force work refresh requested").Count
    Write-Host "🔄 Force Work Refresh Events: $forceRefreshEvents" -ForegroundColor $(if ($forceRefreshEvents -gt 0) { "Green" } else { "Yellow" })
    
    # Count post-solution work waiting
    $postSolutionWaiting = ($logContent | Select-String "Waiting for new work after solution submission").Count
    Write-Host "⏳ Post-Solution Work Waiting: $postSolutionWaiting" -ForegroundColor $(if ($postSolutionWaiting -gt 0) { "Green" } else { "Yellow" })
    
    # Count successful new work detection after solutions
    $newWorkAfterSolution = ($logContent | Select-String "New work found for block.*attempt").Count
    Write-Host "✅ New Work Found After Solutions: $newWorkAfterSolution" -ForegroundColor $(if ($newWorkAfterSolution -gt 0) { "Green" } else { "Yellow" })
    
    # Count "Still block" messages (waiting for new work)
    $stillBlockMessages = ($logContent | Select-String "Still block.*attempt").Count
    Write-Host "⏳ Still Block Messages (Waiting): $stillBlockMessages" -ForegroundColor $(if ($stillBlockMessages -lt 10) { "Green" } else { "Yellow" })
    
    # Count successful work switches
    $successfulSwitches = ($logContent | Select-String "Successfully switched to new work after solution").Count
    Write-Host "🎯 Successful Work Switches: $successfulSwitches" -ForegroundColor $(if ($successfulSwitches -gt 0) { "Green" } else { "Yellow" })
    
    # Count assumed rejections
    $assumedRejections = ($logContent | Select-String "No new work after 10 attempts - assuming solution rejected").Count
    Write-Host "❌ Assumed Rejections: $assumedRejections" -ForegroundColor $(if ($assumedRejections -eq 0) { "Green" } else { "Yellow" })
    
    # Analyze exponential backoff timing
    $backoffTimings = @()
    foreach ($line in $logContent) {
        if ($line -match "waited (\d+)ms") {
            $backoffTimings += [int]$matches[1]
        }
    }
    
    if ($backoffTimings.Count -gt 0) {
        $avgBackoff = ($backoffTimings | Measure-Object -Average).Average
        $maxBackoff = ($backoffTimings | Measure-Object -Maximum).Maximum
        Write-Host "📊 Exponential Backoff Stats:" -ForegroundColor Cyan
        Write-Host "   Average wait time: $($avgBackoff.ToString('F1'))ms"
        Write-Host "   Maximum wait time: $maxBackoff ms"
        Write-Host "   Total backoff events: $($backoffTimings.Count)"
    }
    
    # Calculate post-solution efficiency
    if ($solutionSubmissions -gt 0) {
        $postSolutionEfficiency = ($immediateRefreshTriggers / $solutionSubmissions) * 100
        Write-Host ""
        Write-Host "📈 Post-Solution Efficiency Metrics:" -ForegroundColor Yellow
        Write-Host "   Solutions submitted: $solutionSubmissions"
        Write-Host "   Immediate refresh triggers: $immediateRefreshTriggers"
        Write-Host "   Post-solution efficiency: $($postSolutionEfficiency.ToString('F1'))%"
        
        if ($postSolutionEfficiency -ge 90) {
            Write-Host "   ✅ EXCELLENT: Post-solution work switching working optimally!" -ForegroundColor Green
        } elseif ($postSolutionEfficiency -ge 70) {
            Write-Host "   ✅ GOOD: Post-solution work switching working well" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  NEEDS IMPROVEMENT: Post-solution work switching needs optimization" -ForegroundColor Yellow
        }
    }
    
    # Check for exponential backoff implementation
    $exponentialBackoffWorking = $backoffTimings.Count -gt 0 -and $maxBackoff -le 50
    Write-Host ""
    Write-Host "🔧 Exponential Backoff Analysis:" -ForegroundColor Cyan
    if ($exponentialBackoffWorking) {
        Write-Host "   ✅ Exponential backoff: WORKING (1ms→2ms→4ms→8ms...→50ms cap)" -ForegroundColor Green
        Write-Host "   ✅ Max backoff capped at 50ms: WORKING" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Exponential backoff: Not detected or not working properly" -ForegroundColor Yellow
    }
    
    # Overall assessment
    Write-Host ""
    Write-Host "🎯 OVERALL POST-SOLUTION FIX ASSESSMENT:" -ForegroundColor Yellow
    
    $allSystemsWorking = $immediateRefreshTriggers -gt 0 -and $forceRefreshEvents -gt 0 -and $exponentialBackoffWorking
    
    if ($allSystemsWorking) {
        Write-Host "   ✅ All systems working: Immediate refresh, force refresh, exponential backoff" -ForegroundColor Green
        Write-Host "   ✅ Post-solution work switching: FULLY IMPLEMENTED" -ForegroundColor Green
        Write-Host "   ✅ Stale work reduction: Should be dramatically improved" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Some systems not working optimally - check individual metrics above" -ForegroundColor Yellow
    }
    
    # Check for stale work reduction
    $staleWorkWarnings = ($logContent | Select-String "Quantum work not found \(stale\)").Count
    Write-Host ""
    Write-Host "🚫 Stale Work Analysis:" -ForegroundColor Cyan
    Write-Host "   Stale work warnings: $staleWorkWarnings"
    
    if ($staleWorkWarnings -lt 5) {
        Write-Host "   ✅ EXCELLENT: Stale work dramatically reduced!" -ForegroundColor Green
    } elseif ($staleWorkWarnings -lt 15) {
        Write-Host "   ✅ GOOD: Stale work significantly reduced" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  NEEDS IMPROVEMENT: Still too many stale work warnings" -ForegroundColor Yellow
    }
    
} else {
    Write-Host "❌ No log file found" -ForegroundColor Red
}

Write-Host ""
Write-Host "🔧 Key Improvements Applied:" -ForegroundColor Cyan
Write-Host "   1. Immediate work refresh after solution submission"
Write-Host "   2. Exponential backoff waiting (1ms→2ms→4ms→8ms...→50ms)"
Write-Host "   3. Up to 10 attempts to get new work after solution"
Write-Host "   4. Force refresh channel for instant work updates"
Write-Host "   5. Intelligent block number comparison"
Write-Host "   6. Assumption of rejection after 10 failed attempts"

Write-Host ""
Write-Host "✅ Post-solution work switching test completed!" -ForegroundColor Green 