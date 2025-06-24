# Quantum-Geth Standalone Miner - Solo Mining Script
param(
    [string]$Coinbase = "",
    [string]$NodeURL = "http://localhost:8545",
    [int]$Threads = [System.Environment]::ProcessorCount,
    [int]$Intensity = 1
)

$MinerExecutable = "quantum-miner.exe"

Write-Host " Starting Quantum-Geth Standalone Miner (Solo Mode)" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $MinerExecutable)) {
    Write-Host " Miner executable not found: $MinerExecutable" -ForegroundColor Red
    Write-Host "Please run build-windows.ps1 first to compile the miner." -ForegroundColor Yellow
    exit 1
}

if ($Coinbase -eq "") {
    Write-Host " Coinbase address is required for solo mining!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage examples:" -ForegroundColor Yellow
    Write-Host "  .\run-solo-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A" -ForegroundColor White
    Write-Host "  .\run-solo-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 8" -ForegroundColor White
    exit 1
}

if ($Coinbase -notmatch "^0x[0-9a-fA-F]{40}$") {
    Write-Host " Invalid coinbase address format!" -ForegroundColor Red
    Write-Host "Expected format: 0x followed by 40 hex characters" -ForegroundColor Yellow
    exit 1
}

Write-Host " Mining Configuration:" -ForegroundColor Green
Write-Host "  Mode: Solo Mining" -ForegroundColor White
Write-Host "  Coinbase: $Coinbase" -ForegroundColor White
Write-Host "  Node URL: $NodeURL" -ForegroundColor White
Write-Host "  Threads: $Threads" -ForegroundColor White
Write-Host ""

Write-Host " Starting quantum miner..." -ForegroundColor Blue

$MinerArgs = @(
    "-coinbase", $Coinbase
    "-node", $NodeURL
    "-threads", $Threads
)

try {
    & ".\$MinerExecutable" @MinerArgs
} catch {
    Write-Host " Failed to start miner: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
