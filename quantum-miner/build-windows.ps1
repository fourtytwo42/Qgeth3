# Build script for Quantum-Geth Standalone Miner (Windows)
param(
    [string]$OutputName = "quantum-miner.exe",
    [string]$Version = "1.0.0",
    [switch]$Clean = $false
)

Write-Host " Building Quantum-Geth Standalone Miner for Windows..." -ForegroundColor Cyan
Write-Host "Version: $Version" -ForegroundColor Green
Write-Host ""

if ($Clean) {
    Write-Host " Cleaning previous builds..." -ForegroundColor Yellow
    Remove-Item -Path "*.exe" -ErrorAction SilentlyContinue
    Remove-Item -Path "quantum_solver.py" -ErrorAction SilentlyContinue
}

$env:GOOS = "windows"
$env:GOARCH = "amd64"
$env:CGO_ENABLED = "0"

$BuildTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$GitCommit = try { git rev-parse --short HEAD } catch { "unknown" }

Write-Host "  Compiling Go binary..." -ForegroundColor Blue
Write-Host "  Target: windows/amd64" -ForegroundColor Gray
Write-Host "  Output: $OutputName" -ForegroundColor Gray

try {
    go build -ldflags "-s -w" -o $OutputName .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host " Build successful!" -ForegroundColor Green
        
        $FileInfo = Get-Item $OutputName
        Write-Host ""
        Write-Host " Binary Information:" -ForegroundColor Cyan
        Write-Host "  File: $($FileInfo.Name)" -ForegroundColor White
        Write-Host "  Size: $([math]::Round($FileInfo.Length / 1MB, 2)) MB" -ForegroundColor White
        
        Write-Host ""
        Write-Host " Build complete! Ready to mine quantum blocks." -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage examples:" -ForegroundColor Yellow
        Write-Host "  .\$OutputName -coinbase 0x... -node http://localhost:8545" -ForegroundColor White
        Write-Host "  .\$OutputName -version" -ForegroundColor White
        
    } else {
        Write-Host " Build failed!" -ForegroundColor Red
        exit 1
    }
    
} catch {
    Write-Host " Build error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
