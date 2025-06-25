# Q Coin Build Script
# Usage: ./build-release.ps1 [component] [options]
# Components: geth, miner, both (default: both)

param(
    [Parameter(Position=0)]
    [ValidateSet("geth", "miner", "both")]
    [string]$Component = "both",
    
    [switch]$Clean,
    [switch]$Help
)

if ($Help) {
    Write-Host "Q Coin Build Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: ./build-release.ps1 [component] [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Components:" -ForegroundColor Yellow
    Write-Host "  geth      - Build quantum-geth only"
    Write-Host "  miner     - Build quantum-miner only"
    Write-Host "  both      - Build both geth and miner [DEFAULT]"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -clean    - Clean build directories before building"
    Write-Host "  -help     - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ./build-release.ps1           # Build both geth and miner"
    Write-Host "  ./build-release.ps1 geth      # Build geth only"
    Write-Host "  ./build-release.ps1 -clean    # Clean build and build both"
    exit 0
}

Write-Host "🔨 Q Coin Build Script" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Clean if requested
if ($Clean) {
    Write-Host "🧹 Cleaning build directories..." -ForegroundColor Yellow
    
    if (Test-Path "quantum-geth\build") {
        Remove-Item -Recurse -Force "quantum-geth\build"
        Write-Host "   Cleaned quantum-geth\build" -ForegroundColor Gray
    }
    
    if (Test-Path "quantum-miner\quantum-miner.exe") {
        Remove-Item -Force "quantum-miner\quantum-miner.exe"
        Write-Host "   Cleaned quantum-miner.exe" -ForegroundColor Gray
    }
    
    Write-Host "✅ Clean completed" -ForegroundColor Green
    Write-Host ""
}

# Build geth
if ($Component -eq "geth" -or $Component -eq "both") {
    Write-Host "🔨 Building quantum-geth..." -ForegroundColor Yellow
    
    if (-not (Test-Path "quantum-geth")) {
        Write-Host "❌ quantum-geth directory not found!" -ForegroundColor Red
        exit 1
    }
    
    Push-Location "quantum-geth"
    try {
        $env:CGO_ENABLED = "1"
        & go build -o "build\bin\geth.exe" ".\cmd\geth"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ quantum-geth built successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ quantum-geth build failed!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host ""
}

# Build miner
if ($Component -eq "miner" -or $Component -eq "both") {
    Write-Host "🔨 Building quantum-miner..." -ForegroundColor Yellow
    
    if (-not (Test-Path "quantum-miner")) {
        Write-Host "❌ quantum-miner directory not found!" -ForegroundColor Red
        exit 1
    }
    
    Push-Location "quantum-miner"
    try {
        $env:CGO_ENABLED = "1"
        & go build -o "quantum-miner.exe" "."
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ quantum-miner built successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ quantum-miner build failed!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host ""
}

Write-Host "🎉 Build completed successfully!" -ForegroundColor Green

# Show built files
Write-Host ""
Write-Host "Built files:" -ForegroundColor Cyan
if ($Component -eq "geth" -or $Component -eq "both") {
    if (Test-Path "quantum-geth\build\bin\geth.exe") {
        $gethSize = (Get-Item "quantum-geth\build\bin\geth.exe").Length / 1MB
        Write-Host "  quantum-geth\build\bin\geth.exe ($([math]::Round($gethSize, 1)) MB)" -ForegroundColor White
    }
}
if ($Component -eq "miner" -or $Component -eq "both") {
    if (Test-Path "quantum-miner\quantum-miner.exe") {
        $minerSize = (Get-Item "quantum-miner\quantum-miner.exe").Length / 1MB
        Write-Host "  quantum-miner\quantum-miner.exe ($([math]::Round($minerSize, 1)) MB)" -ForegroundColor White
    }
} 