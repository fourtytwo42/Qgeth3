# Quiet Quantum Miner - Shows only the dashboard report
param(
    [string]$coinbase = "0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A",
    [int]$threads = 4,
    [switch]$gpu
)

Write-Host "🚀 Starting Quiet Quantum Miner..." -ForegroundColor Cyan
Write-Host "   Coinbase: $coinbase" -ForegroundColor White
Write-Host "   Threads: $threads" -ForegroundColor White
if ($gpu) {
    Write-Host "   Mode: GPU Accelerated" -ForegroundColor Green
} else {
    Write-Host "   Mode: CPU Only" -ForegroundColor Yellow
}
Write-Host ""

# Build the command
$minerArgs = @(
    "-coinbase", $coinbase,
    "-threads", $threads.ToString()
)

if ($gpu) {
    $minerArgs += "-gpu"
}

# Run the miner and filter output to show only the dashboard
& "quantum-miner/quantum-miner.exe" @minerArgs 2>&1 | ForEach-Object {
    $line = $_.ToString()
    
    # Show dashboard lines (lines that start with │ or ┌ or ├ or └)
    if ($line -match "^[│┌├└┐┘┤┬┴┼─]" -or 
        $line -match "QUANTUM.*MINER" -or
        $line -match "Last Update:" -or
        $line -match "Press Ctrl\+C" -or
        $line -match "FINAL.*REPORT" -or
        $line -match "═══════════" -or
        $line -match "🏁|📊|🎮|⏱️|⚡|🧮|🎯|💻|👋|💎") {
        Write-Host $line
    }
    
    # Show important status messages
    elseif ($line -match "Connected to:|Building|built successfully|Mining stopped|Shutdown") {
        Write-Host $line -ForegroundColor Gray
    }
} 