# Quantum-Geth Standalone Miner
param(
    [string]$Coinbase = "",
    [string]$NodeURL = "http://localhost:8545",
    [int]$Threads = 1
)

$MinerExecutable = "quantum-miner.exe"

Write-Host "🚀 Quantum-Geth Standalone Miner" -ForegroundColor Cyan

if ($Coinbase -eq "") {
    Write-Host "❌ Coinbase address required!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage examples:" -ForegroundColor Yellow
    Write-Host "  .\run-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A" -ForegroundColor White
    Write-Host "  .\run-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 8" -ForegroundColor White
    Write-Host "  .\run-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -NodeURL http://192.168.1.100:8545" -ForegroundColor White
    exit 1
}

if ($Coinbase -notmatch "^0x[0-9a-fA-F]{40}$") {
    Write-Host "❌ Invalid coinbase address format!" -ForegroundColor Red
    Write-Host "Expected format: 0x followed by 40 hex characters" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $MinerExecutable)) {
    Write-Host "❌ Miner executable not found: $MinerExecutable" -ForegroundColor Red
    Write-Host "Please ensure quantum-miner.exe is compiled and available." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "📋 Configuration:" -ForegroundColor Green
Write-Host "   💰 Coinbase: $Coinbase" -ForegroundColor White
Write-Host "   🌐 Node URL: $NodeURL" -ForegroundColor White
Write-Host "   🧵 Threads: $Threads" -ForegroundColor White
Write-Host ""

$MinerArgs = @("-coinbase", $Coinbase, "-node", $NodeURL, "-threads", $Threads)

Write-Host "🚀 Starting miner..." -ForegroundColor Blue
try {
    & ".\$MinerExecutable" @MinerArgs
} catch {
    Write-Host "❌ Failed to start miner: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
