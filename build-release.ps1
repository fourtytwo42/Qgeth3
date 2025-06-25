# Q Coin Build Script - Creates timestamped releases
# Usage: ./build-release.ps1 [component]
# Components: geth, miner, both (default: both)

param(
    [Parameter(Position=0)]
    [ValidateSet("geth", "miner", "both")]
    [string]$Component = "both"
)

Write-Host "Building Q Coin Release..." -ForegroundColor Cyan
Write-Host ""

$timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

# Build geth
if ($Component -eq "geth" -or $Component -eq "both") {
    Write-Host "Building quantum-geth..." -ForegroundColor Yellow
    
    if (-not (Test-Path "quantum-geth")) {
        Write-Host "quantum-geth directory not found!" -ForegroundColor Red
        exit 1
    }
    
    # Build to regular location first
    Push-Location "quantum-geth"
    try {
        # Disable CGO for geth (not needed on Windows)
        $env:CGO_ENABLED = "0"
        & go build -o "build/bin/geth.exe" "./cmd/geth"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "quantum-geth built successfully" -ForegroundColor Green
            
            # Create timestamped release
            $releaseDir = "../releases/quantum-geth-$timestamp"
            New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
            Copy-Item "build/bin/geth.exe" "$releaseDir/geth.exe" -Force
            
            # Create release info
            @"
# Quantum-Geth Release $timestamp
Built: $(Get-Date)
Component: quantum-geth
Version: Latest
"@ | Out-File -FilePath "$releaseDir/README.md" -Encoding UTF8
            
            Write-Host "Created release: $releaseDir" -ForegroundColor Green
        } else {
            Write-Host "quantum-geth build failed!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host ""
}

# Build miner
if ($Component -eq "miner" -or $Component -eq "both") {
    Write-Host "Building quantum-miner..." -ForegroundColor Yellow
    
    if (-not (Test-Path "quantum-miner")) {
        Write-Host "quantum-miner directory not found!" -ForegroundColor Red
        exit 1
    }
    
    # Build to regular location first
    Push-Location "quantum-miner"
    try {
        # Disable CGO for Windows miner (uses CuPy instead of native CUDA)
        $env:CGO_ENABLED = "0"
        & go build -o "quantum-miner.exe" "."
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "quantum-miner built successfully" -ForegroundColor Green
            
            # Create timestamped release
            $releaseDir = "../releases/quantum-miner-$timestamp"
            New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
            Copy-Item "quantum-miner.exe" "$releaseDir/quantum-miner.exe" -Force
            
            # Copy pkg directory if it exists
            if (Test-Path "pkg") {
                Copy-Item "pkg" "$releaseDir/pkg" -Recurse -Force
            }
            
            # Create release scripts
            @"
# Quantum-Miner GPU Launcher (PowerShell)
param(
    [string]`$Coinbase = "",
    [int]`$Threads = 1,
    [int]`$GpuId = 0,
    [string]`$NodeURL = "http://localhost:8545"
)

if (`$Coinbase -eq "") {
    Write-Host "Usage: .\start-miner-gpu.ps1 -Coinbase <address>" -ForegroundColor Yellow
    exit 1
}

& ".\quantum-miner.exe" -gpu -coinbase "`$Coinbase" -threads `$Threads -gpu-id `$GpuId -node "`$NodeURL"
"@ | Out-File -FilePath "$releaseDir/start-miner-gpu.ps1" -Encoding UTF8

            @"
# Quantum-Miner CPU Launcher (PowerShell)
param(
    [string]`$Coinbase = "",
    [int]`$Threads = 4,
    [string]`$NodeURL = "http://localhost:8545"
)

if (`$Coinbase -eq "") {
    Write-Host "Usage: .\start-miner-cpu.ps1 -Coinbase <address>" -ForegroundColor Yellow
    exit 1
}

& ".\quantum-miner.exe" -coinbase "`$Coinbase" -threads `$Threads -node "`$NodeURL"
"@ | Out-File -FilePath "$releaseDir/start-miner-cpu.ps1" -Encoding UTF8

            # Create release info
            @"
# Quantum-Miner Release $timestamp
Built: $(Get-Date)
Component: quantum-miner
Version: Latest

## Usage
- GPU Mining: .\start-miner-gpu.ps1 -Coinbase <address>
- CPU Mining: .\start-miner-cpu.ps1 -Coinbase <address>
"@ | Out-File -FilePath "$releaseDir/README.md" -Encoding UTF8
            
            Write-Host "Created release: $releaseDir" -ForegroundColor Green
        } else {
            Write-Host "quantum-miner build failed!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host ""
}

Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "" 