#!/usr/bin/env pwsh

# Test Enhanced Quantum Miner
# This script tests the new thread management and memory pooling features

Write-Host "🧪 Testing Enhanced Quantum Miner with Thread Management" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Check if geth is running
$gethProcess = Get-Process -Name "geth" -ErrorAction SilentlyContinue
if (-not $gethProcess) {
    Write-Host "❌ Geth is not running. Please start quantum-geth first." -ForegroundColor Red
    Write-Host "   Run: .\geth.exe --mine --miner.threads=0 --http --http.api=eth,net,web3,qmpow" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Geth is running (PID: $($gethProcess.Id))" -ForegroundColor Green

# Build the enhanced miner
Write-Host "`n🔨 Building enhanced quantum miner..." -ForegroundColor Yellow
Set-Location quantum-miner

try {
    go build -o quantum-miner-enhanced.exe .
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    Write-Host "✅ Build successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Build failed: $_" -ForegroundColor Red
    exit 1
}

# Test with conservative settings first
Write-Host "`n🚀 Starting enhanced miner with conservative settings..." -ForegroundColor Yellow
Write-Host "   - 8 threads (limited to 4 concurrent)" -ForegroundColor Cyan
Write-Host "   - Memory pooling enabled" -ForegroundColor Cyan
Write-Host "   - Staggered thread execution" -ForegroundColor Cyan
Write-Host "   - Enhanced thread monitoring" -ForegroundColor Cyan

# Start the miner with logging
$minerArgs = @(
    "-coinbase", "0x1234567890123456789012345678901234567890"
    "-threads", "8"
    "-gpu"
    "-url", "http://localhost:8545"
    "-log"
)

Write-Host "`nStarting: .\quantum-miner-enhanced.exe $($minerArgs -join ' ')" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the miner and view results`n" -ForegroundColor Yellow

# Monitor system resources while mining
$job = Start-Job -ScriptBlock {
    param($minerPath, $args)
    & $minerPath @args
} -ArgumentList ".\quantum-miner-enhanced.exe", $minerArgs

# Monitor for 60 seconds
$startTime = Get-Date
$duration = 60 # seconds
$monitorInterval = 5 # seconds

Write-Host "📊 Monitoring system resources for $duration seconds..." -ForegroundColor Cyan

while ((Get-Date) -lt $startTime.AddSeconds($duration) -and $job.State -eq "Running") {
    Start-Sleep $monitorInterval
    
    # Check memory usage
    $memory = Get-WmiObject -Class Win32_OperatingSystem
    $totalMem = [math]::Round($memory.TotalVisibleMemorySize / 1MB, 2)
    $freeMem = [math]::Round($memory.FreePhysicalMemory / 1MB, 2)
    $usedMem = [math]::Round($totalMem - $freeMem, 2)
    $memPercent = [math]::Round(($usedMem / $totalMem) * 100, 1)
    
    # Check disk activity
    $disk = Get-Counter "\PhysicalDisk(_Total)\% Disk Time" -SampleInterval 1 -MaxSamples 1
    $diskUsage = [math]::Round($disk.CounterSamples[0].CookedValue, 1)
    
    $elapsed = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 0)
    Write-Host "[$elapsed s] Memory: $usedMem GB / $totalMem GB ($memPercent%) | Disk: $diskUsage%" -ForegroundColor Gray
    
    # Alert if disk usage is high (potential swapping)
    if ($diskUsage -gt 85) {
        Write-Host "⚠️  HIGH DISK ACTIVITY DETECTED! Possible memory swapping." -ForegroundColor Red
    }
}

# Stop the miner
Write-Host "`n🛑 Stopping miner..." -ForegroundColor Yellow
Stop-Job $job -Force
Remove-Job $job -Force

# Check if log file was created
$logFile = "quantum-miner.log"
if (Test-Path $logFile) {
    Write-Host "`n📋 Analyzing log file..." -ForegroundColor Cyan
    
    # Count thread abort messages
    $abortCount = (Select-String -Path $logFile -Pattern "Aborting stale work" | Measure-Object).Count
    $stuckCount = (Select-String -Path $logFile -Pattern "stuck for" | Measure-Object).Count
    $memoryPoolMessages = (Select-String -Path $logFile -Pattern "Memory pool" | Measure-Object).Count
    $threadManagementMessages = (Select-String -Path $logFile -Pattern "Thread management" | Measure-Object).Count
    
    Write-Host "📊 Test Results:" -ForegroundColor Green
    Write-Host "   - Thread aborts: $abortCount" -ForegroundColor White
    Write-Host "   - Stuck threads: $stuckCount" -ForegroundColor White
    Write-Host "   - Memory pool messages: $memoryPoolMessages" -ForegroundColor White
    Write-Host "   - Thread management messages: $threadManagementMessages" -ForegroundColor White
    
    if ($stuckCount -eq 0) {
        Write-Host "✅ SUCCESS: No stuck threads detected!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  WARNING: $stuckCount stuck thread events detected" -ForegroundColor Yellow
    }
    
    if ($abortCount -lt 10) {
        Write-Host "✅ SUCCESS: Low thread abort count ($abortCount)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  WARNING: High thread abort count ($abortCount)" -ForegroundColor Yellow
    }
    
    # Show last few lines of log
    Write-Host "`n📄 Last 10 lines of log:" -ForegroundColor Cyan
    Get-Content $logFile -Tail 10 | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
    
} else {
    Write-Host "❌ No log file found. Miner may not have started properly." -ForegroundColor Red
}

Write-Host "`n✅ Enhanced miner test completed!" -ForegroundColor Green
Write-Host "🔍 Check quantum-miner.log for detailed results" -ForegroundColor Cyan

Set-Location .. 