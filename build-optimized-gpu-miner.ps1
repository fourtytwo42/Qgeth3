#!/usr/bin/env pwsh

Write-Host "🚀 Building OPTIMIZED Quantum GPU Miner" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green

# Get timestamp
$timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

# Set build variables
$env:CGO_ENABLED = "1"
$env:GOOS = "linux"
$env:GOARCH = "amd64"

# Create releases directory
$releasesDir = "releases"
if (!(Test-Path $releasesDir)) {
    New-Item -ItemType Directory -Path $releasesDir
}

# Create release directory
$releaseDir = Join-Path $releasesDir "quantum-gpu-miner-optimized-$timestamp"
New-Item -ItemType Directory -Path $releaseDir -Force

Write-Host "📦 Release directory: $releaseDir" -ForegroundColor Yellow

# Build the miner
Write-Host "🔨 Building quantum-gpu-miner (Linux binary)..." -ForegroundColor Yellow
Set-Location "quantum-miner"

# Build with optimizations
$buildCmd = "go build -ldflags `"-s -w -X main.Version=optimized-$timestamp`" -o quantum-gpu-miner ."
Write-Host "Command: $buildCmd" -ForegroundColor Cyan
Invoke-Expression $buildCmd

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed!"
    exit 1
}

Write-Host "✅ Build successful!" -ForegroundColor Green

# Copy binary to release directory
Copy-Item "quantum-gpu-miner" (Join-Path ".." $releaseDir "quantum-gpu-miner") -Force

# Copy Python scripts
$pythonDir = Join-Path ".." $releaseDir "pkg" "quantum"
New-Item -ItemType Directory -Path $pythonDir -Force -Recurse
Copy-Item "pkg\quantum\qiskit_gpu.py" (Join-Path $pythonDir "qiskit_gpu.py") -Force

# Copy requirements
Copy-Item "pkg\quantum\requirements-gpu.txt" (Join-Path $pythonDir "requirements-gpu.txt") -Force

Write-Host "📋 Created optimized GPU miner in: $releaseDir" -ForegroundColor Green
Write-Host "🎯 Key optimizations:" -ForegroundColor Yellow
Write-Host "   • Reduced logging (every 100th batch instead of every batch)" -ForegroundColor White
Write-Host "   • Faster timeouts (5s instead of 15s)" -ForegroundColor White
Write-Host "   • No artificial delays in fallback simulation" -ForegroundColor White
Write-Host "   • Increased WSL2 concurrency (4 instead of 2)" -ForegroundColor White
Write-Host "   • Real quantum circuits (no CPU approximation)" -ForegroundColor White

Set-Location ".."
Write-Host "🚀 Ready to test! Run from: $releaseDir" -ForegroundColor Green 