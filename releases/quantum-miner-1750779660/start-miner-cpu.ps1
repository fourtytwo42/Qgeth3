# Quantum-Miner CPU Launcher (PowerShell)
param(
    [string]$Coinbase = "",
    [int]$Threads = 1,
    [string]$NodeURL = "http://localhost:8545",
    [switch]$Help
)

if ($Help -or $Coinbase -eq "") {
    Write-Host "Quantum-Miner CPU Launcher" -ForegroundColor Cyan
    Write-Host "Usage: .\start-miner-cpu.ps1 -Coinbase <address> [options]" -ForegroundColor Yellow
    Write-Host "  -Coinbase: Mining reward address (required)" -ForegroundColor White
    Write-Host "  -Threads: CPU mining threads (default: 1)" -ForegroundColor White
    Write-Host "  -NodeURL: Quantum-Geth node URL (default: http://localhost:8545)" -ForegroundColor White
    Write-Host "Example: .\start-miner-cpu.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A" -ForegroundColor Green
    if ($Coinbase -eq "") { exit 1 }
    exit 0
}

Write-Host "Starting Quantum-Miner (CPU Mode)..." -ForegroundColor Blue
Write-Host "Coinbase: $Coinbase" -ForegroundColor Green
Write-Host "Threads: $Threads" -ForegroundColor Green
Write-Host "Node URL: $NodeURL" -ForegroundColor Green
Write-Host ""

& ".\quantum-miner.exe" -coinbase "$Coinbase" -threads $Threads -node "$NodeURL"
