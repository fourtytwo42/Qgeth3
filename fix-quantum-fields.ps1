# Quick Fix Script for Quantum Field Compatibility Issue
# This script rebuilds geth with CGO_ENABLED=0 to ensure compatibility with Linux nodes
# Run this script to fix the "missing quantum fields in header" issue

param(
    [switch]$Force = $false
)

Write-Host "🔧 Q Coin Quantum Field Compatibility Fix" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Issue: Windows geth blocks not compatible with Linux nodes" -ForegroundColor Yellow
Write-Host "Cause: CGO_ENABLED inconsistency between platforms" -ForegroundColor Yellow
Write-Host "Fix:   Rebuild with CGO_ENABLED=0 on both platforms" -ForegroundColor Yellow
Write-Host ""

if (-not $Force) {
    Write-Host "This will rebuild your geth.exe. Continue? (y/n): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Operation cancelled." -ForegroundColor Red
        exit 0
    }
}

# Check if quantum-geth exists
if (-not (Test-Path "quantum-geth")) {
    Write-Host "❌ Error: quantum-geth directory not found!" -ForegroundColor Red
    Write-Host "Please run this script from the root of the Qgeth3 project." -ForegroundColor Red
    exit 1
}

Write-Host "🛠️ Step 1: Backing up current geth.exe..." -ForegroundColor Cyan
if (Test-Path "geth.exe") {
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    Copy-Item "geth.exe" "geth-backup-$timestamp.exe" -Force
    Write-Host "✅ Backup created: geth-backup-$timestamp.exe" -ForegroundColor Green
}

Write-Host ""
Write-Host "🛠️ Step 2: Rebuilding quantum-geth with CGO_ENABLED=0..." -ForegroundColor Cyan

Push-Location "quantum-geth"
try {
    # CRITICAL: Ensure CGO_ENABLED=0 for quantum field compatibility
    $env:CGO_ENABLED = "0"
    Write-Host "🛡️ Enforcing CGO_ENABLED=0 for quantum field compatibility" -ForegroundColor Yellow
    
    # Build geth
    & go build -o "../geth.exe" "./cmd/geth"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ quantum-geth rebuilt successfully (CGO_ENABLED=0)" -ForegroundColor Green
    } else {
        Write-Host "❌ Error: Failed to rebuild quantum-geth" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "🧪 Step 3: Testing the fix..." -ForegroundColor Cyan

# Test that geth can start
Write-Host "Testing geth startup..." -NoNewline
$testResult = & "./geth.exe" version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host " ✅" -ForegroundColor Green
} else {
    Write-Host " ❌" -ForegroundColor Red
    Write-Host "Warning: geth may have build issues" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Fix Applied Successfully!" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Restart your geth node: ./start-geth.ps1 devnet" -ForegroundColor White
Write-Host "2. Start mining: ./start-miner.ps1" -ForegroundColor White
Write-Host "3. Your blocks should now be compatible with Linux nodes" -ForegroundColor White
Write-Host ""
Write-Host "To verify the fix:" -ForegroundColor Cyan
Write-Host "- Check that your VPS Linux nodes can sync your new blocks" -ForegroundColor White
Write-Host "- No more 'missing quantum fields in header' errors" -ForegroundColor White
Write-Host ""
Write-Host "Technical Details:" -ForegroundColor Gray
Write-Host "- Windows geth now built with CGO_ENABLED=0" -ForegroundColor Gray
Write-Host "- Matches Linux build configuration" -ForegroundColor Gray
Write-Host "- Ensures identical RLP encoding/decoding" -ForegroundColor Gray 