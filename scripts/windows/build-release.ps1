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

# Get absolute path to releases directory before changing directories
$releasesPath = (Resolve-Path "../../releases").Path

# Build geth
if ($Component -eq "geth" -or $Component -eq "both") {
    Write-Host "Building quantum-geth..." -ForegroundColor Yellow
    
    if (-not (Test-Path "../../quantum-geth")) {
        Write-Host "quantum-geth directory not found!" -ForegroundColor Red
        exit 1
    }
    
    # Build to regular location first
    Push-Location "../../quantum-geth"
    try {
        # CRITICAL: Always use CGO_ENABLED=0 for geth to ensure compatibility
        # This ensures Windows and Linux builds have identical quantum field handling
        $env:CGO_ENABLED = "0"
        Write-Host "ENFORCING: CGO_ENABLED=0 for geth build (quantum field compatibility)" -ForegroundColor Yellow
        & go build -o "geth.exe" "./cmd/geth"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "quantum-geth built successfully (CGO_ENABLED=0)" -ForegroundColor Green
            
            # Create timestamped release
            $releaseDir = "$releasesPath\quantum-geth-$timestamp"
            New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
            Copy-Item "geth.exe" "$releaseDir\geth.exe" -Force
            
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
    
    if (-not (Test-Path "../../quantum-miner")) {
        Write-Host "quantum-miner directory not found!" -ForegroundColor Red
        exit 1
    }
    
    # Build to regular location first
    Push-Location "../../quantum-miner"
    try {
        # Use CGO_ENABLED=0 for Windows miner (uses CuPy instead of native CUDA)
        $env:CGO_ENABLED = "0"
        Write-Host "INFO: Using CGO_ENABLED=0 for Windows miner (CuPy GPU support)" -ForegroundColor Cyan
        & go build -o "quantum-miner.exe" "."
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "quantum-miner built successfully" -ForegroundColor Green
            
            # Create timestamped release
            $releaseDir = "$releasesPath\quantum-miner-$timestamp"
            New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
            Copy-Item "quantum-miner.exe" "$releaseDir\quantum-miner.exe" -Force
            
            # Copy pkg directory if it exists
            if (Test-Path "pkg") {
                Copy-Item "pkg" "$releaseDir\pkg" -Recurse -Force
            }
            
            # Create release scripts
            $gpuScript = @'
# Quantum-Miner GPU Launcher (PowerShell)
param(
    [string]$Coinbase = "",
    [int]$Threads = 1,
    [int]$GpuId = 0,
    [string]$NodeURL = "http://localhost:8545"
)

if ($Coinbase -eq "") {
    Write-Host "Usage: .\start-miner-gpu.ps1 -Coinbase [address]" -ForegroundColor Yellow
    exit 1
}

& ".\quantum-miner.exe" -gpu -coinbase "$Coinbase" -threads $Threads -gpu-id $GpuId -node "$NodeURL"
'@
            $gpuScript | Out-File -FilePath "$releaseDir/start-miner-gpu.ps1" -Encoding UTF8

            $cpuScript = @'
# Quantum-Miner CPU Launcher (PowerShell)
param(
    [string]$Coinbase = "",
    [int]$Threads = 4,
    [string]$NodeURL = "http://localhost:8545"
)

if ($Coinbase -eq "") {
    Write-Host "Usage: .\start-miner-cpu.ps1 -Coinbase [address]" -ForegroundColor Yellow
    exit 1
}

& ".\quantum-miner.exe" -coinbase "$Coinbase" -threads $Threads -node "$NodeURL"
'@
            $cpuScript | Out-File -FilePath "$releaseDir/start-miner-cpu.ps1" -Encoding UTF8

            # Create release info
            $readmeLines = @(
                "# Quantum-Miner Release $timestamp",
                "Built: $(Get-Date)",
                "Component: quantum-miner", 
                "Version: Latest",
                "",
                "## Usage",
                "* GPU Mining: .\start-miner-gpu.ps1 -Coinbase [address]",
                "* CPU Mining: .\start-miner-cpu.ps1 -Coinbase [address]"
            )
            $readmeLines -join "`r`n" | Out-File -FilePath "$releaseDir/README.md" -Encoding UTF8
            
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