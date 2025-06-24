# Quantum-Geth CPU Miner (Local)
param(
    [string]$Coinbase = "",
    [string]$NodeURL = "http://localhost:8545",
    [int]$Threads = 1,
    [switch]$Help
)

$MinerExecutable = "quantum-gpu-miner.exe"

if ($Help) {
    Write-Host "Quantum-Geth CPU Miner (Local)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Description:" -ForegroundColor Yellow
    Write-Host "  High-performance quantum mining with CPU-based simulation" -ForegroundColor White
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\run-cpu-miner.ps1 -Coinbase <address> [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Yellow
    Write-Host "  -Coinbase <address>    Coinbase address for mining rewards (required)" -ForegroundColor White
    Write-Host "  -NodeURL <url>         Quantum-Geth node URL (default: http://localhost:8545)" -ForegroundColor White
    Write-Host "  -Threads <number>      Number of CPU mining threads (default: 1)" -ForegroundColor White
    Write-Host "  -Help                  Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run-cpu-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A" -ForegroundColor Green
    Write-Host "  .\run-cpu-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 4" -ForegroundColor Green
    Write-Host "  .\run-cpu-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -NodeURL http://192.168.1.100:8545" -ForegroundColor Green
    Write-Host ""
    Write-Host "Features:" -ForegroundColor Yellow
    Write-Host "  * CPU quantum simulation (0.36 puzzles/sec)" -ForegroundColor Green
    Write-Host "  * Real quantum circuit simulation (16-qubit, 8192 T-gates)" -ForegroundColor Green
    Write-Host "  * Multi-threaded mining support" -ForegroundColor Green
    Write-Host "  * No additional dependencies required" -ForegroundColor Green
    Write-Host ""
    Write-Host "Performance Comparison:" -ForegroundColor Yellow
    Write-Host "  CPU Mining: 0.36 puzzles/sec (this miner)" -ForegroundColor White
    Write-Host "  GPU Mining: 0.45 puzzles/sec (use .\run-gpu-miner.ps1)" -ForegroundColor White
    Write-Host ""
    Write-Host "Note:" -ForegroundColor Yellow
    Write-Host "  This script uses quantum-gpu-miner.exe in CPU mode (without -gpu flag)" -ForegroundColor White
    Write-Host "  For GPU acceleration, use .\run-gpu-miner.ps1" -ForegroundColor White
    Write-Host ""
    exit 0
}

Write-Host "Quantum-Geth CPU Miner" -ForegroundColor Cyan
Write-Host "CPU-based quantum circuit mining" -ForegroundColor Blue

if ($Coinbase -eq "") {
    Write-Host "ERROR: Coinbase address required!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage examples:" -ForegroundColor Yellow
    Write-Host "  .\run-cpu-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A" -ForegroundColor White
    Write-Host "  .\run-cpu-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 4" -ForegroundColor White
    Write-Host "  .\run-cpu-miner.ps1 -Help" -ForegroundColor White
    Write-Host ""
    Write-Host "Tip: For better performance, try GPU mining with .\run-gpu-miner.ps1" -ForegroundColor Yellow
    exit 1
}

if ($Coinbase -notmatch "^0x[0-9a-fA-F]{40}$") {
    Write-Host "ERROR: Invalid coinbase address format!" -ForegroundColor Red
    Write-Host "Expected format: 0x followed by 40 hex characters" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $MinerExecutable)) {
    Write-Host "ERROR: Quantum miner executable not found: $MinerExecutable" -ForegroundColor Red
    Write-Host "Please build the quantum-gpu-miner first." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To build the miner:" -ForegroundColor Yellow
    Write-Host "  go build -o quantum-gpu-miner.exe ." -ForegroundColor White
    Write-Host ""
    Write-Host "Or use the build script:" -ForegroundColor Yellow
    Write-Host "  .\build.ps1 cpu" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "CPU Mining Configuration:" -ForegroundColor Green
Write-Host "   Coinbase: $Coinbase" -ForegroundColor White
Write-Host "   Node URL: $NodeURL" -ForegroundColor White
Write-Host "   CPU Threads: $Threads" -ForegroundColor White
Write-Host "   Mining Mode: CPU only (no -gpu flag)" -ForegroundColor White
Write-Host "   Quantum Puzzles: 48 per block" -ForegroundColor White
Write-Host "   Circuit Size: 16 qubits, 8192 T-gates" -ForegroundColor White
Write-Host ""

# Note: No -gpu flag used, so it defaults to CPU mode
$MinerArgs = @("-coinbase", $Coinbase, "-node", $NodeURL, "-threads", $Threads)

Write-Host "Starting CPU quantum miner..." -ForegroundColor Blue
Write-Host ""

try {
    & ".\$MinerExecutable" @MinerArgs
} catch {
    Write-Host "ERROR: Failed to start CPU miner: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check quantum-geth node is running at $NodeURL" -ForegroundColor White
    Write-Host "  2. Verify coinbase address format" -ForegroundColor White
    Write-Host "  3. Ensure quantum-gpu-miner.exe is built and accessible" -ForegroundColor White
    Write-Host "  4. Consider GPU mining for better performance: .\run-gpu-miner.ps1" -ForegroundColor White
    Write-Host "  5. Try building: go build -o quantum-gpu-miner.exe ." -ForegroundColor White
    exit 1
}
