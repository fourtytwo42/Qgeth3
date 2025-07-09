#!/usr/bin/env pwsh

Write-Host "========================================"
Write-Host "  Q Geth Quantum Miner - Node Recovery Test"
Write-Host "========================================"
Write-Host ""
Write-Host "Testing new features:"
Write-Host "- Solution rate limiting (500ms minimum between submissions)"
Write-Host "- Quantum-geth node recovery (detects 'no mining work available yet')"
Write-Host "- Intelligent work refresh after solutions"
Write-Host ""

# Build the quantum miner
Write-Host "Building quantum miner..."
cd quantum-miner
$buildResult = go build -o quantum-miner.exe .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to build quantum miner" -ForegroundColor Red
    Read-Host "Press Enter to continue..."
    exit 1
}

# Run the quantum miner with logging
Write-Host ""
Write-Host "Starting quantum miner with node recovery features..."
Write-Host ""

$minerArgs = @(
    "-coinbase", "0x742d35Cc6634C0532925a3b8D186aaD9a5C9B9b5",
    "-url", "http://127.0.0.1:8545",
    "-threads", "4",
    "-log", "quantum-miner-node-recovery.log"
)

& .\quantum-miner.exe @minerArgs

Write-Host ""
Write-Host "Quantum miner stopped."
Read-Host "Press Enter to continue..." 